// peresmeshnik-loopback: the call app's sound through a WASAPI process loopback, as raw PCM on stdout
// (Windows 10 2004 / build 19041+). The Windows counterpart of native/macos/tap.
//
//   peresmeshnik-loopback check            one line "windows=<build> loopback=yes|no"; exit 0, 5 too old
//   peresmeshnik-loopback console          one line "console=0|1": does this process own a console window
//   peresmeshnik-loopback list             the audio sessions, one JSON object per line: pid, name, playing, exe
//   peresmeshnik-loopback tap --pid N [--tree]   process loopback of pid (and, with --tree, its process tree)
//   peresmeshnik-loopback volume --pid N [--set L] [--tree]   get or set the sessions' volume (one-shot)
//   peresmeshnik-loopback mute --pid N [--tree]   hold pid's (tree's) sessions at 0, new ones too (mute-only; the
//   peresmeshnik-loopback watch            device-change notifications, one JSON line each (default-device change)
//
// tap: a WASAPI loopback that captures only what process `pid` (with --tree, it and its children) plays, mixed to
// mono. stdout: one line "peresmeshnik-loopback 1 rate=<Hz> channels=1 format=f32le", then mono float32
// little-endian at that rate. It lives while stdin is open. The process loopback is taken BEFORE the app's audio
// session volume, so lowering that volume (volume --set 0, the mute spike) silences the speakers but not us. An app
// that has not played yet gives silence until it does. Nothing polls: a WASAPI event, a wait on the process handle,
// a blocked read of stdin.
// Exit: 0 stdin or stdout closed (or the process ended, --pid), 2 usage, 3 no such process, 4 access denied,
// 5 Windows older than 10 2004 (build 19041), 6 a WASAPI/COM error, 7 the mix format changed: start again, the new
// header has the new rate. Messages go to stderr.
//
// Build (no project file): native/windows/loopback/build.cmd native/windows/loopback <out.exe>  (cl via vswhere).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <mmdeviceapi.h>
#include <Audiopolicy.h>
#include <Functiondiscoverykeys_devpkey.h>
#include <wrl/implements.h>
#include <tlhelp32.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <io.h>
#include <fcntl.h>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using Microsoft::WRL::RuntimeClass;
using Microsoft::WRL::RuntimeClassFlags;
using Microsoft::WRL::ClassicCom;
using Microsoft::WRL::FtmBase;

enum Code { OK = 0, USAGE = 2, NO_PROCESS = 3, ACCESS = 4, OLD_WINDOWS = 5, WASAPI = 6, RESTART = 7 };

static void say(const char* fmt, ...) {
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "peresmeshnik-loopback: %s\n", buf);
    fflush(stderr);
}

[[noreturn]] static void quit(Code code, const char* text = nullptr) {
    if (text) say("%s", text);
    exit(code);
}

static bool hr_ok(HRESULT hr, const char* what) {
    if (SUCCEEDED(hr)) return true;
    say("%s: 0x%08lx", what, (unsigned long)hr);
    return false;
}

// The build at which process loopback (AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK) exists: Windows 10 2004.
static const DWORD MIN_BUILD = 19041;

static DWORD os_build() {
    // RtlGetVersion: GetVersionEx lies without a manifest; the real build number is what we need.
    typedef LONG(WINAPI * RtlGetVersionFn)(PRTL_OSVERSIONINFOW);
    RTL_OSVERSIONINFOW vi = { sizeof(vi) };
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    auto fn = nt ? (RtlGetVersionFn)GetProcAddress(nt, "RtlGetVersion") : nullptr;
    if (fn) fn(&vi);
    return vi.dwBuildNumber;
}

static std::string wide_to_utf8(const wchar_t* w) {
    if (!w || !*w) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

// The image name (basename of the exe) of a process id, and whether it is still alive.
static std::string exe_name(DWORD pid) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return {};
    wchar_t path[MAX_PATH]; DWORD n = MAX_PATH;
    std::string name;
    if (QueryFullProcessImageNameW(h, 0, path, &n)) {
        std::wstring p(path, n);
        size_t slash = p.find_last_of(L"\\/");
        name = wide_to_utf8((slash == std::wstring::npos ? p : p.substr(slash + 1)).c_str());
    }
    CloseHandle(h);
    return name;
}

static bool pid_alive(DWORD pid) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return false;
    DWORD code = 0;
    bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
}

// ---- process loopback capture -------------------------------------------------------------------

// ActivateAudioInterfaceAsync is asynchronous: the completion handler holds the IAudioClient it made.
struct ActivateHandler
    : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IActivateAudioInterfaceCompletionHandler> {
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HRESULT activate_hr = E_FAIL;
    ComPtr<IAudioClient> client;
    STDMETHOD(ActivateCompleted)(IActivateAudioInterfaceAsyncOperation* op) override {
        HRESULT hr = S_OK; ComPtr<IUnknown> unk;
        op->GetActivateResult(&hr, &unk);
        activate_hr = hr;
        if (SUCCEEDED(hr)) unk.As(&client);
        SetEvent(done);
        return S_OK;
    }
    ~ActivateHandler() { if (done) CloseHandle(done); }
};

static ComPtr<IAudioClient> activate_loopback(DWORD pid, HRESULT* out_hr) {
    AUDIOCLIENT_ACTIVATION_PARAMS params = {};
    params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    params.ProcessLoopbackParams.TargetProcessId = pid;
    // INCLUDE_TARGET_PROCESS_TREE captures the app and its children (a browser's tabs, a call app's helpers);
    // EXCLUDE would capture everything but the app, which is not what we want.
    params.ProcessLoopbackParams.ProcessLoopbackMode = PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;

    PROPVARIANT pv = {};
    pv.vt = VT_BLOB;
    pv.blob.cbSize = sizeof(params);
    pv.blob.pBlobData = reinterpret_cast<BYTE*>(&params);

    auto handler = Microsoft::WRL::Make<ActivateHandler>();
    ComPtr<IActivateAudioInterfaceAsyncOperation> op;
    HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient), &pv,
                                             handler.Get(), &op);
    if (FAILED(hr)) { *out_hr = hr; return nullptr; }
    WaitForSingleObject(handler->done, INFINITE);
    *out_hr = handler->activate_hr;
    return handler->client;
}

static WAVEFORMATEX mono_float(DWORD rate) {
    WAVEFORMATEX f = {};
    f.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    f.nChannels = 1;
    f.nSamplesPerSec = rate;
    f.wBitsPerSample = 32;
    f.nBlockAlign = f.nChannels * f.wBitsPerSample / 8;
    f.nAvgBytesPerSec = f.nSamplesPerSec * f.nBlockAlign;
    f.cbSize = 0;
    return f;
}

static std::atomic<bool> g_stdin_open{true};
static HANDLE g_stdin_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);  // set when stdin closes (manual-reset)

// Block on stdin: it closing (the parent is done, or asks us to drop the loopback) ends the capture. Setting the
// event wakes the waiters at once, so no one polls on a timer.
static DWORD WINAPI watch_stdin(LPVOID) {
    char buf[256];
    while (true) {
        DWORD n = 0;
        HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
        if (!ReadFile(in, buf, sizeof(buf), &n, nullptr) || n == 0) break;
    }
    g_stdin_open = false;
    SetEvent(g_stdin_event);
    return 0;
}

static int run_tap(DWORD pid, DWORD rate) {
    if (!pid_alive(pid)) quit(NO_PROCESS, "the process is not running");
    HRESULT hr = S_OK;
    ComPtr<IAudioClient> client = activate_loopback(pid, &hr);
    if (!client) {
        if (hr == E_ACCESSDENIED) quit(ACCESS, "access denied to the process loopback");
        char b[96]; snprintf(b, sizeof(b), "could not activate the process loopback: 0x%08lx", (unsigned long)hr);
        quit(WASAPI, b);
    }
    WAVEFORMATEX fmt = mono_float(rate);
    // A process loopback is a virtual endpoint: it must be shared, event-driven, with the loopback flag. Its mix is
    // produced at the format we ask for (mono f32 here); AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM lets it convert the rate.
    DWORD flags = AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
                  AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 2000000 /*200 ms*/, 0, &fmt, nullptr);
    if (FAILED(hr)) {
        char b[96]; snprintf(b, sizeof(b), "IAudioClient::Initialize: 0x%08lx", (unsigned long)hr);
        quit(WASAPI, b);
    }
    HANDLE evt = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!hr_ok(client->SetEventHandle(evt), "SetEventHandle")) quit(WASAPI);
    ComPtr<IAudioCaptureClient> capture;
    if (!hr_ok(client->GetService(IID_PPV_ARGS(&capture)), "GetService(IAudioCaptureClient)")) quit(WASAPI);
    if (!hr_ok(client->Start(), "IAudioClient::Start")) quit(WASAPI);

    // stdout binary; the header the Python side parses, then the PCM stream.
    _setmode(_fileno(stdout), _O_BINARY);
    printf("peresmeshnik-loopback 1 rate=%lu channels=1 format=f32le\n", (unsigned long)rate);
    fflush(stdout);
    say("tapping %lu at %lu Hz (tree)", (unsigned long)pid, (unsigned long)rate);

    CreateThread(nullptr, 0, watch_stdin, nullptr, 0, nullptr);
    HANDLE proc = OpenProcess(SYNCHRONIZE, FALSE, pid);
    HANDLE waits[3]; DWORD n_waits = 0;
    waits[n_waits++] = evt;                              // 0: an audio buffer is ready (WASAPI event)
    waits[n_waits++] = g_stdin_event;                    // 1: stdin closed (stop)
    DWORD proc_idx = 0;
    if (proc) { proc_idx = n_waits; waits[n_waits++] = proc; }  // the process ended

    int code = OK;
    while (g_stdin_open) {
        DWORD w = WaitForMultipleObjects(n_waits, waits, FALSE, INFINITE);  // no timer wakeups during the call
        if (w == WAIT_OBJECT_0 + 1) break;  // stdin closed
        if (proc && w == WAIT_OBJECT_0 + proc_idx) { say("the process ended"); break; }
        if (!g_stdin_open) break;
        UINT32 avail = 0;
        while (SUCCEEDED(capture->GetNextPacketSize(&avail)) && avail > 0) {
            BYTE* data = nullptr; UINT32 frames = 0; DWORD pkflags = 0;
            hr = capture->GetBuffer(&data, &frames, &pkflags, nullptr, nullptr);
            if (hr == AUDCLNT_S_BUFFER_EMPTY) break;
            if (FAILED(hr)) { say("GetBuffer: 0x%08lx", (unsigned long)hr); code = WASAPI; g_stdin_open = false; break; }
            size_t bytes = (size_t)frames * fmt.nBlockAlign;
            if (frames) {
                if (pkflags & AUDCLNT_BUFFERFLAGS_SILENT) {
                    static std::vector<float> zeros;
                    zeros.assign(frames, 0.0f);
                    fwrite(zeros.data(), sizeof(float), frames, stdout);
                } else {
                    fwrite(data, 1, bytes, stdout);
                }
                if (ferror(stdout)) { g_stdin_open = false; }  // the reader is gone: done
            }
            capture->ReleaseBuffer(frames);
        }
        fflush(stdout);
    }
    client->Stop();
    if (proc) CloseHandle(proc);
    CloseHandle(evt);
    return code;
}

// ---- audio sessions (list, volume) --------------------------------------------------------------

template <class F>
static bool sessions_of(IMMDevice* dev, bool& any, F& cb) {
    ComPtr<IAudioSessionManager2> mgr;
    if (FAILED(dev->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr, &mgr))) return false;
    ComPtr<IAudioSessionEnumerator> en;
    if (FAILED(mgr->GetSessionEnumerator(&en))) return false;
    int count = 0; en->GetCount(&count);
    for (int i = 0; i < count; i++) {
        ComPtr<IAudioSessionControl> ctl;
        if (FAILED(en->GetSession(i, &ctl))) continue;
        ComPtr<IAudioSessionControl2> ctl2;
        if (FAILED(ctl.As(&ctl2))) continue;
        DWORD pid = 0; ctl2->GetProcessId(&pid);
        AudioSessionState state = AudioSessionStateInactive; ctl->GetState(&state);
        cb(pid, state, ctl2);
        any = true;
    }
    return true;
}

// Every session on every active render endpoint (the call app may play to a device that is not the default);
// `cb(pid, state, ctl2)`.
template <class F>
static bool each_session(F&& cb) {
    ComPtr<IMMDeviceEnumerator> en;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en)))) return false;
    ComPtr<IMMDeviceCollection> devices;
    if (FAILED(en->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &devices))) return false;
    UINT n = 0; devices->GetCount(&n);
    bool ok = false, any = false;
    for (UINT i = 0; i < n; i++) {
        ComPtr<IMMDevice> dev;
        if (SUCCEEDED(devices->Item(i, &dev)) && sessions_of(dev.Get(), any, cb))
            ok = true;
    }
    return ok;
}

static int run_list() {
    bool ok = each_session([](DWORD pid, AudioSessionState state, ComPtr<IAudioSessionControl2>& ctl2) {
        if (!pid) return;  // the system sounds session
        std::string name = exe_name(pid);
        printf("{\"pid\": %lu, \"name\": \"%s\", \"exe\": \"%s\", \"playing\": %s}\n",
               (unsigned long)pid, json_escape(name).c_str(), json_escape(name).c_str(),
               state == AudioSessionStateActive ? "true" : "false");
    });
    fflush(stdout);
    if (!ok) quit(WASAPI, "could not enumerate the audio sessions");
    return OK;
}

// Is `pid` the target, or a descendant of it (the call app's tree)?
static bool in_tree(DWORD pid, DWORD target) {
    if (pid == target) return true;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32W e = { sizeof(e) };
    std::vector<std::pair<DWORD, DWORD>> parents;  // (pid, ppid)
    if (Process32FirstW(snap, &e)) do {
        parents.push_back({ e.th32ProcessID, e.th32ParentProcessID });
    } while (Process32NextW(snap, &e));
    CloseHandle(snap);
    DWORD cur = pid;
    for (int depth = 0; depth < 64 && cur; depth++) {  // bounded: a cycle (reused pids) never loops forever
        if (cur == target) return true;
        DWORD parent = 0;
        for (auto& pr : parents) if (pr.first == cur) { parent = pr.second; break; }
        cur = parent;
    }
    return false;
}

static int run_volume(DWORD pid, bool tree, bool set, float level) {
    bool any = false;
    bool ok = each_session([&](DWORD spid, AudioSessionState, ComPtr<IAudioSessionControl2>& ctl2) {
        if (!spid) return;
        if (!(spid == pid || (tree && in_tree(spid, pid)))) return;
        ComPtr<ISimpleAudioVolume> vol;
        if (FAILED(ctl2.As(&vol))) return;
        if (set) {
            vol->SetMasterVolume(level, nullptr);
            any = true;
        } else {
            float v = 1.0f; vol->GetMasterVolume(&v);
            printf("{\"pid\": %lu, \"volume\": %.4f}\n", (unsigned long)spid, v);
            any = true;
        }
    });
    fflush(stdout);
    if (!ok) quit(WASAPI, "could not enumerate the audio sessions");
    if (!any) quit(NO_PROCESS, "no audio session for the process");
    return OK;
}

// ---- the mute spike helper (mute): hold the call app's sessions at 0, new ones too ------------------------------
// `volume --set 0` is a one-shot: a session that starts after it (a call beginning, a new browser tab) plays at full
// volume. `mute` stays: it sets every session of the target's tree to 0, registers IAudioSessionNotification and
// mutes new ones as they appear. It is mute-only: a line "{"event":"muted","pid":P,"level":L}" per session lets the
// Python side keep the original level (per app) and own the restore, so neither this helper dying nor a Windows
// restart (which persists the muted 0) can lose it. On stdin close it stops muting and exits (no restore here).

static std::mutex g_mute_lock;
static std::set<std::wstring> g_mute_seen;  // session instance ids already handled (dedupe enumeration vs. callback)
static DWORD g_mute_target = 0;
static bool g_mute_tree = false;
static bool g_mute_stopping = false;  // set under the lock before unregister: a late OnSessionCreated then no-ops

static void mute_session(ComPtr<IAudioSessionControl2>& ctl2) {
    DWORD spid = 0; ctl2->GetProcessId(&spid);
    if (!spid) return;
    if (!(spid == g_mute_target || (g_mute_tree && in_tree(spid, g_mute_target)))) return;
    std::wstring key;
    LPWSTR id = nullptr;
    if (SUCCEEDED(ctl2->GetSessionInstanceIdentifier(&id)) && id) { key = id; CoTaskMemFree(id); }
    ComPtr<ISimpleAudioVolume> vol;
    if (FAILED(ctl2.As(&vol))) return;
    float original = 1.0f; vol->GetMasterVolume(&original);
    {
        std::lock_guard<std::mutex> lk(g_mute_lock);
        if (g_mute_stopping) return;  // stopping: do not mute a session created as we tear down
        if (!key.empty()) {
            if (g_mute_seen.count(key)) return;
            g_mute_seen.insert(key);
        }
    }
    vol->SetMasterVolume(0.0f, nullptr);
    printf("{\"event\": \"muted\", \"pid\": %lu, \"level\": %.4f}\n", (unsigned long)spid, original);
    fflush(stdout);
}

struct SessionNotify : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IAudioSessionNotification> {
    STDMETHOD(OnSessionCreated)(IAudioSessionControl* ctl) override {
        ComPtr<IAudioSessionControl2> ctl2;
        if (ctl && SUCCEEDED(ctl->QueryInterface(IID_PPV_ARGS(&ctl2)))) mute_session(ctl2);
        return S_OK;
    }
};

static int run_mute(DWORD pid, bool tree) {
    g_mute_target = pid; g_mute_tree = tree;
    ComPtr<IMMDeviceEnumerator> en;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en))))
        quit(WASAPI, "no device enumerator");
    ComPtr<IMMDeviceCollection> devices;
    if (FAILED(en->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &devices))) quit(WASAPI, "no render endpoints");
    UINT n = 0; devices->GetCount(&n);
    std::vector<ComPtr<IAudioSessionManager2>> mgrs;  // kept alive: notifications stop if the manager is released
    std::vector<ComPtr<SessionNotify>> notifs;
    for (UINT i = 0; i < n; i++) {
        ComPtr<IMMDevice> dev;
        if (FAILED(devices->Item(i, &dev))) continue;
        ComPtr<IAudioSessionManager2> mgr;
        if (FAILED(dev->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr, &mgr))) continue;
        ComPtr<IAudioSessionEnumerator> sen;  // priming the enumerator is required for OnSessionCreated to fire
        if (SUCCEEDED(mgr->GetSessionEnumerator(&sen))) {
            int c = 0; sen->GetCount(&c);
            for (int j = 0; j < c; j++) {
                ComPtr<IAudioSessionControl> ctl;
                if (FAILED(sen->GetSession(j, &ctl))) continue;
                ComPtr<IAudioSessionControl2> ctl2;
                if (SUCCEEDED(ctl.As(&ctl2))) mute_session(ctl2);
            }
        }
        auto nc = Microsoft::WRL::Make<SessionNotify>();
        if (SUCCEEDED(mgr->RegisterSessionNotification(nc.Get()))) { mgrs.push_back(mgr); notifs.push_back(nc); }
    }
    say("muting %lu (tree)", (unsigned long)pid);
    CreateThread(nullptr, 0, watch_stdin, nullptr, 0, nullptr);
    WaitForSingleObject(g_stdin_event, INFINITE);  // the MTA delivers OnSessionCreated on its own threads meanwhile
    { std::lock_guard<std::mutex> lk(g_mute_lock); g_mute_stopping = true; }  // stop muting first, then unregister
    for (size_t i = 0; i < mgrs.size(); i++) mgrs[i]->UnregisterSessionNotification(notifs[i].Get());
    return OK;  // mute-only: the Python side restores from its saved per-app level
}

// ---- device-change notifications (watch) --------------------------------------------------------

struct NotifyClient : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IMMNotificationClient> {
    STDMETHOD(OnDefaultDeviceChanged)(EDataFlow flow, ERole role, LPCWSTR id) override {
        if (role == eConsole) {
            printf("{\"event\": \"default\", \"flow\": \"%s\", \"id\": \"%s\"}\n",
                   flow == eRender ? "render" : "capture", json_escape(wide_to_utf8(id)).c_str());
            fflush(stdout);
        }
        return S_OK;
    }
    STDMETHOD(OnDeviceStateChanged)(LPCWSTR id, DWORD state) override {
        printf("{\"event\": \"state\", \"id\": \"%s\", \"state\": %lu}\n",
               json_escape(wide_to_utf8(id)).c_str(), (unsigned long)state);
        fflush(stdout);
        return S_OK;
    }
    STDMETHOD(OnDeviceAdded)(LPCWSTR) override { return S_OK; }
    STDMETHOD(OnDeviceRemoved)(LPCWSTR) override { return S_OK; }
    STDMETHOD(OnPropertyValueChanged)(LPCWSTR, const PROPERTYKEY) override { return S_OK; }
};

static int run_watch() {
    ComPtr<IMMDeviceEnumerator> en;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en))))
        quit(WASAPI, "no device enumerator");
    auto nc = Microsoft::WRL::Make<NotifyClient>();
    if (!hr_ok(en->RegisterEndpointNotificationCallback(nc.Get()), "RegisterEndpointNotificationCallback")) quit(WASAPI);
    say("watching device changes");
    CreateThread(nullptr, 0, watch_stdin, nullptr, 0, nullptr);
    WaitForSingleObject(g_stdin_event, INFINITE);  // the COM MTA delivers notifications on its own threads; no polling
    en->UnregisterEndpointNotificationCallback(nc.Get());
    return OK;
}

// ---- entry --------------------------------------------------------------------------------------

static DWORD arg_dword(int argc, wchar_t** argv, const wchar_t* flag, DWORD dflt) {
    for (int i = 0; i < argc; i++) if (!wcscmp(argv[i], flag) && i + 1 < argc) return wcstoul(argv[i + 1], nullptr, 10);
    return dflt;
}

static bool has_flag(int argc, wchar_t** argv, const wchar_t* flag) {
    for (int i = 0; i < argc; i++) if (!wcscmp(argv[i], flag)) return true;
    return false;
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) quit(USAGE, "usage: peresmeshnik-loopback check|console|list|tap|volume|mute|watch");
    std::wstring cmd = argv[1];
    DWORD build = os_build();

    if (cmd == L"check") {
        printf("windows=%lu loopback=%s\n", (unsigned long)build, build >= MIN_BUILD ? "yes" : "no");
        fflush(stdout);
        return build >= MIN_BUILD ? OK : OLD_WINDOWS;
    }
    if (cmd == L"console") {  // does this process own a console window? the app spawns the helper with none (CREATE_NO_WINDOW)
        printf("console=%d\n", GetConsoleWindow() != nullptr ? 1 : 0);
        fflush(stdout);
        return OK;
    }
    if (build < MIN_BUILD) quit(OLD_WINDOWS, "Windows older than 10 2004 (build 19041): no process loopback");

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);  // MTA: process loopback and notifications run on it
    if (FAILED(hr)) quit(WASAPI, "CoInitializeEx failed");

    int code = USAGE;
    if (cmd == L"list") {
        code = run_list();
    } else if (cmd == L"watch") {
        code = run_watch();
    } else if (cmd == L"tap") {
        DWORD pid = arg_dword(argc, argv, L"--pid", 0);
        DWORD rate = arg_dword(argc, argv, L"--rate", 48000);
        if (!pid) quit(USAGE, "tap needs --pid N");
        code = run_tap(pid, rate);
    } else if (cmd == L"volume") {
        DWORD pid = arg_dword(argc, argv, L"--pid", 0);
        if (!pid) quit(USAGE, "volume needs --pid N");
        bool set = has_flag(argc, argv, L"--set");
        float level = 0.0f;
        for (int i = 0; i < argc; i++) if (!wcscmp(argv[i], L"--set") && i + 1 < argc) level = (float)wcstod(argv[i + 1], nullptr);
        code = run_volume(pid, has_flag(argc, argv, L"--tree"), set, level);
    } else if (cmd == L"mute") {
        DWORD pid = arg_dword(argc, argv, L"--pid", 0);
        if (!pid) quit(USAGE, "mute needs --pid N");
        code = run_mute(pid, has_flag(argc, argv, L"--tree"));
    } else {
        say("unknown command");
    }
    CoUninitialize();
    return code;
}
