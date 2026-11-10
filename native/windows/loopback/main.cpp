// peresmeshnik-loopback: the call app's sound through a WASAPI process loopback, as raw PCM on stdout
// (Windows 10 2004 / build 19041+). The Windows counterpart of native/macos/tap.
//
//   peresmeshnik-loopback check            one line "windows=<build> loopback=yes|no"; exit 0, 5 too old
//   peresmeshnik-loopback console          one line "console=0|1": does this process own a console window
//   peresmeshnik-loopback list             the audio sessions, one JSON object per line: pid, name, playing, exe
//   peresmeshnik-loopback tap --pid N [--tree]   process loopback of pid (and, with --tree, its process tree)
//   peresmeshnik-loopback volume --pid N [--set L] [--tree]   get or set the sessions' volume (one-shot); a JSON
//                                          line per session: pid, device (endpoint id), volume
//   peresmeshnik-loopback mute --pid N [--tree] [--level L]   hold pid's (tree's) sessions at L (0), new ones too, on
//                                          every device
//                                          (mute-only; the app restores): {"event":"muted","pid","level","device"}
//   peresmeshnik-loopback keep --exe NAME --level L [--device ID]... [--for S]   give an app its level back: each
//                                          session of a process named NAME (or of its tree) at 0 on one of the devices
//                                          (any without --device), now or as it appears, is set to L:
//                                          {"event":"restored","pid","device"} (also for one found not at 0). It does
//                                          not read stdin: it outlives the app (a call app's next call without
//                                          Peresmeshnik), and ends once every device is done, or after S seconds
//   peresmeshnik-loopback watch            device-change notifications, one JSON line each (default-device change)
//
// tap: a WASAPI loopback that captures only what process `pid` (with --tree, it and its children) plays, mixed to
// mono. stdout: one line "peresmeshnik-loopback 1 rate=<Hz> channels=1 format=f32le", then mono float32
// little-endian at that rate. It lives while stdin is open. The process loopback is taken AFTER the app's audio
// session volume (a session at 0 gives silence: measured on the runner, stage 7), so the app does not mute the call
// app but ducks it (mute --level) and brings the loopback back up. An app that has not played yet gives silence until
// it does. A WASAPI event, a wait on the process handle, a blocked read
// of stdin; the capture is also checked once a second (CHECK_MS), as peresmeshnik-audio does: Windows may invalidate
// it (a driver reset) without an event, and a dead capture must end, not wait forever. Windows Audio restarting is not
// told to a process loopback at all (GetNextPacketSize keeps succeeding, no event: runner, stage 7), so the service is
// watched too (an SCM notification): stopped -> wait for it to run again (up to 30 s) -> exit 7.
// Exit: 0 stdin or stdout closed, 2 usage, 3 no such process (or it ended: the call app was closed), 4 access denied,
// 5 Windows older than 10 2004 (build 19041), 6 a WASAPI/COM error, 7 the capture was invalidated: start again (a
// new header, maybe a new rate). Messages go to stderr.
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
#include <winsvc.h>
#pragma comment(lib, "advapi32.lib")

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <io.h>
#include <fcntl.h>
#include <map>
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
static const DWORD CHECK_MS = 1000;  // a capture is checked at least this often (Windows may end one without a word)

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

// ---- Windows Audio (Audiosrv) stopping: an SCM notification, delivered as an APC to the waiting (alertable) thread

static HANDLE g_audiosrv_stopped = CreateEventW(nullptr, TRUE, FALSE, nullptr);
static SERVICE_NOTIFYW g_audiosrv_note = {};

static void CALLBACK audiosrv_changed(PVOID) { SetEvent(g_audiosrv_stopped); }

struct Audiosrv {
    SC_HANDLE scm = nullptr, svc = nullptr;
    bool watch() {
        scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
        svc = scm ? OpenServiceW(scm, L"Audiosrv", SERVICE_QUERY_STATUS) : nullptr;
        if (!svc) return false;
        g_audiosrv_note.dwVersion = SERVICE_NOTIFY_STATUS_CHANGE;
        g_audiosrv_note.pfnNotifyCallback = audiosrv_changed;
        return NotifyServiceStatusChangeW(svc, SERVICE_NOTIFY_STOPPED | SERVICE_NOTIFY_STOP_PENDING, &g_audiosrv_note)
               == ERROR_SUCCESS;
    }
    void wait_running(DWORD ms) {  // the rare way back after a restart: a short poll, never during a call
        SERVICE_STATUS st = {};
        for (DWORD t = 0; t < ms; t += 250) {
            if (QueryServiceStatus(svc, &st) && st.dwCurrentState == SERVICE_RUNNING) return;
            Sleep(250);
        }
    }
    ~Audiosrv() {
        if (svc) CloseServiceHandle(svc);
        if (scm) CloseServiceHandle(scm);
    }
};

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
    HANDLE waits[4]; DWORD n_waits = 0;
    waits[n_waits++] = evt;                              // 0: an audio buffer is ready (WASAPI event)
    waits[n_waits++] = g_stdin_event;                    // 1: stdin closed (stop)
    DWORD proc_idx = 0;
    if (proc) { proc_idx = n_waits; waits[n_waits++] = proc; }  // the process ended
    Audiosrv audiosrv;
    DWORD srv_idx = 0;
    if (audiosrv.watch()) { srv_idx = n_waits; waits[n_waits++] = g_audiosrv_stopped; }  // Windows Audio stopped
    else say("cannot watch the Windows Audio service: a restart of it is not noticed");

    int code = OK;
    while (g_stdin_open) {
        // alertable: the SCM notification runs here; a timeout is the liveness check below
        DWORD w = WaitForMultipleObjectsEx(n_waits, waits, FALSE, CHECK_MS, TRUE);
        if (w == WAIT_IO_COMPLETION) continue;
        if (w == WAIT_OBJECT_0 + 1) break;  // stdin closed
        if (proc && w == WAIT_OBJECT_0 + proc_idx) { say("the process ended"); code = NO_PROCESS; break; }
        if (srv_idx && w == WAIT_OBJECT_0 + srv_idx) {
            say("Windows Audio stopped: the capture is gone; starting again once it runs");
            audiosrv.wait_running(30000);
            code = RESTART;
            break;
        }
        if (!g_stdin_open) break;
        UINT32 avail = 0;
        while (SUCCEEDED(hr = capture->GetNextPacketSize(&avail)) && avail > 0) {
            BYTE* data = nullptr; UINT32 frames = 0; DWORD pkflags = 0;
            hr = capture->GetBuffer(&data, &frames, &pkflags, nullptr, nullptr);
            if (hr == AUDCLNT_S_BUFFER_EMPTY) break;
            if (FAILED(hr)) {  // invalidated between GetNextPacketSize and here (a driver reset): start again, as below
                say("GetBuffer: 0x%08lx", (unsigned long)hr);
                code = hr == AUDCLNT_E_DEVICE_INVALIDATED ? RESTART : WASAPI;
                g_stdin_open = false;
                break;
            }
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
        if (FAILED(hr) && code == OK) {  // the capture is dead (an event never comes again): end, the app starts anew
            say("the capture stopped: 0x%08lx", (unsigned long)hr);
            code = hr == AUDCLNT_E_DEVICE_INVALIDATED ? RESTART : WASAPI;
            break;
        }
        fflush(stdout);
    }
    client->Stop();
    if (proc) CloseHandle(proc);
    CloseHandle(evt);
    return code;
}

// ---- audio sessions (list, volume) --------------------------------------------------------------

static std::wstring device_id(IMMDevice* dev) {
    LPWSTR id = nullptr;
    std::wstring s;
    if (SUCCEEDED(dev->GetId(&id)) && id) { s = id; CoTaskMemFree(id); }
    return s;
}

template <class F>
static bool sessions_of(IMMDevice* dev, bool& any, F& cb) {
    std::wstring device = device_id(dev);
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
        cb(pid, state, ctl2, device);
        any = true;
    }
    return true;
}

// Every session on every active render endpoint (the call app may play to a device that is not the default);
// `cb(pid, state, ctl2, device id)`.
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
    bool ok = each_session([](DWORD pid, AudioSessionState state, ComPtr<IAudioSessionControl2>&, const std::wstring&) {
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

// `pid` and its ancestors, nearest first: (pid, exe name) each.
static std::vector<std::pair<DWORD, std::wstring>> ancestry(DWORD pid) {
    std::vector<std::pair<DWORD, std::wstring>> chain;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return chain;
    PROCESSENTRY32W e = { sizeof(e) };
    std::map<DWORD, std::pair<DWORD, std::wstring>> procs;  // pid -> (ppid, exe)
    if (Process32FirstW(snap, &e)) do {
        procs[e.th32ProcessID] = { e.th32ParentProcessID, e.szExeFile };
    } while (Process32NextW(snap, &e));
    CloseHandle(snap);
    DWORD cur = pid;
    for (int depth = 0; depth < 64 && cur; depth++) {  // bounded: a cycle (reused pids) never loops forever
        auto it = procs.find(cur);
        if (it == procs.end()) break;
        chain.push_back({ cur, it->second.second });
        cur = it->second.first;
    }
    return chain;
}

// Is `pid` the target, or a descendant of it (the call app's tree)?
static bool in_tree(DWORD pid, DWORD target) {
    if (pid == target) return true;
    for (auto& a : ancestry(pid)) if (a.first == target) return true;
    return false;
}

// Is `pid` a process named `exe`, or a descendant of one (the call app's tree, by name: it may not run yet)?
static bool in_exe_tree(DWORD pid, const std::wstring& exe) {
    for (auto& a : ancestry(pid)) if (!_wcsicmp(a.second.c_str(), exe.c_str())) return true;
    return false;
}

static int run_volume(DWORD pid, bool tree, bool set, float level) {
    bool any = false;
    bool ok = each_session([&](DWORD spid, AudioSessionState, ComPtr<IAudioSessionControl2>& ctl2,
                               const std::wstring& device) {
        if (!spid) return;
        if (!(spid == pid || (tree && in_tree(spid, pid)))) return;
        ComPtr<ISimpleAudioVolume> vol;
        if (FAILED(ctl2.As(&vol))) return;
        if (set && FAILED(vol->SetMasterVolume(level, nullptr))) return;
        float v = 1.0f; vol->GetMasterVolume(&v);
        printf("{\"pid\": %lu, \"device\": \"%s\", \"volume\": %.4f}\n", (unsigned long)spid,
               json_escape(wide_to_utf8(device.c_str())).c_str(), v);
        any = true;
    });
    fflush(stdout);
    if (!ok) quit(WASAPI, "could not enumerate the audio sessions");
    if (!any) quit(NO_PROCESS, "no audio session for the process");
    return OK;
}

// ---- watching sessions (mute, keep) --------------------------------------------------------------------------
// Every session of every active render endpoint now, and each new one as it appears (IAudioSessionNotification),
// also on an endpoint that becomes active later (a headset plugged in: IMMNotificationClient wakes the main thread,
// which registers there; COM calls are not made inside the notification). `g_on_session(ctl2, device id)` runs on
// the main thread for the existing ones and on the MTA's threads for new ones; it ends when stdin closes.

static void (*g_on_session)(ComPtr<IAudioSessionControl2>&, const std::wstring&) = nullptr;
static std::mutex g_watch_lock;
static bool g_watch_stopping = false;  // set under the lock before unregister: a late OnSessionCreated then no-ops
static HANDLE g_devices_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);  // an endpoint came or went

static bool watch_stopping() { std::lock_guard<std::mutex> lk(g_watch_lock); return g_watch_stopping; }

struct SessionNotify : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IAudioSessionNotification> {
    std::wstring device;
    STDMETHOD(OnSessionCreated)(IAudioSessionControl* ctl) override {
        ComPtr<IAudioSessionControl2> ctl2;
        if (ctl && SUCCEEDED(ctl->QueryInterface(IID_PPV_ARGS(&ctl2))) && !watch_stopping()) g_on_session(ctl2, device);
        return S_OK;
    }
};

struct EndpointNotify : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IMMNotificationClient> {
    STDMETHOD(OnDeviceStateChanged)(LPCWSTR, DWORD) override { SetEvent(g_devices_event); return S_OK; }
    STDMETHOD(OnDeviceAdded)(LPCWSTR) override { SetEvent(g_devices_event); return S_OK; }
    STDMETHOD(OnDeviceRemoved)(LPCWSTR) override { SetEvent(g_devices_event); return S_OK; }
    STDMETHOD(OnDefaultDeviceChanged)(EDataFlow, ERole, LPCWSTR) override { return S_OK; }
    STDMETHOD(OnPropertyValueChanged)(LPCWSTR, const PROPERTYKEY) override { return S_OK; }
};

struct Watched { ComPtr<IAudioSessionManager2> mgr; ComPtr<SessionNotify> nc; };

// Register on every active render endpoint not watched yet (its sessions now go to g_on_session); drop the ones
// that are gone (a replugged one is registered anew: its old manager is dead).
static void watch_endpoints(IMMDeviceEnumerator* en, std::map<std::wstring, Watched>& watched) {
    ComPtr<IMMDeviceCollection> devices;
    if (FAILED(en->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &devices))) return;
    UINT n = 0; devices->GetCount(&n);
    std::set<std::wstring> active;
    for (UINT i = 0; i < n; i++) {
        ComPtr<IMMDevice> dev;
        if (FAILED(devices->Item(i, &dev))) continue;
        std::wstring key = device_id(dev.Get());
        if (key.empty()) continue;
        active.insert(key);
        if (watched.count(key)) continue;
        ComPtr<IAudioSessionManager2> mgr;
        if (FAILED(dev->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr, &mgr))) continue;
        ComPtr<IAudioSessionEnumerator> sen;  // priming the enumerator is required for OnSessionCreated to fire
        if (SUCCEEDED(mgr->GetSessionEnumerator(&sen))) {
            int c = 0; sen->GetCount(&c);
            for (int j = 0; j < c; j++) {
                ComPtr<IAudioSessionControl> ctl;
                if (FAILED(sen->GetSession(j, &ctl))) continue;
                ComPtr<IAudioSessionControl2> ctl2;
                if (SUCCEEDED(ctl.As(&ctl2))) g_on_session(ctl2, key);
            }
        }
        auto nc = Microsoft::WRL::Make<SessionNotify>();
        nc->device = key;
        if (SUCCEEDED(mgr->RegisterSessionNotification(nc.Get()))) watched[key] = { mgr, nc };
    }
    for (auto it = watched.begin(); it != watched.end();) {
        if (active.count(it->first)) { ++it; continue; }
        it->second.mgr->UnregisterSessionNotification(it->second.nc.Get());
        it = watched.erase(it);
    }
}

// Ends when stdin closes, or (keep) when `done` is set or `lifetime_ms` has passed.
static int watch_sessions(void (*on)(ComPtr<IAudioSessionControl2>&, const std::wstring&), HANDLE done = nullptr,
                          ULONGLONG lifetime_ms = 0) {
    g_on_session = on;
    ComPtr<IMMDeviceEnumerator> en;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en))))
        quit(WASAPI, "no device enumerator");
    auto endpoints = Microsoft::WRL::Make<EndpointNotify>();  // first: an endpoint arriving during the scan is not missed
    bool endpoints_ok = SUCCEEDED(en->RegisterEndpointNotificationCallback(endpoints.Get()));
    std::map<std::wstring, Watched> watched;  // kept alive: notifications stop if the manager is released
    watch_endpoints(en.Get(), watched);
    if (!done) CreateThread(nullptr, 0, watch_stdin, nullptr, 0, nullptr);
    HANDLE w[2] = { done ? done : g_stdin_event, g_devices_event };
    ULONGLONG end = lifetime_ms ? GetTickCount64() + lifetime_ms : 0;
    while (true) {
        DWORD left = INFINITE;
        if (end) {
            ULONGLONG now = GetTickCount64();
            if (now >= end) break;
            left = (DWORD)(end - now < 0x7fffffff ? end - now : 0x7fffffff);
        }
        if (WaitForMultipleObjects(2, w, FALSE, left) != WAIT_OBJECT_0 + 1) break;  // ended, done or out of time
        watch_endpoints(en.Get(), watched);  // an endpoint came or went
    }
    { std::lock_guard<std::mutex> lk(g_watch_lock); g_watch_stopping = true; }  // stop acting first, then unregister
    if (endpoints_ok) en->UnregisterEndpointNotificationCallback(endpoints.Get());
    for (auto& kv : watched) kv.second.mgr->UnregisterSessionNotification(kv.second.nc.Get());
    return OK;
}

// ---- the mute spike helper (mute): hold the call app's sessions at 0, new ones too ------------------------------
// `volume --set 0` is a one-shot: a session that starts after it (a call beginning, a new browser tab, a headset
// plugged in) plays at full volume. `mute` stays: it sets every session of the target's tree to 0 and mutes new ones
// as they appear. It is mute-only: a line "{"event":"muted","pid":P,"level":L,"device":ID}" per session lets the
// Python side keep the original level (per app) and the devices it is at 0 on, and own the restore, so neither this
// helper dying nor a Windows restart (which persists the muted 0) can lose it. On stdin close it stops muting and
// exits (no restore here).

static std::mutex g_mute_lock;
static std::set<std::wstring> g_mute_seen;  // session instance ids already handled (dedupe enumeration vs. callback)
static DWORD g_mute_target = 0;
static bool g_mute_tree = false;
static float g_mute_level = 0.0f;

static void mute_session(ComPtr<IAudioSessionControl2>& ctl2, const std::wstring& device) {
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
        if (!key.empty()) {
            if (g_mute_seen.count(key)) return;
            g_mute_seen.insert(key);
        }
    }
    HRESULT hr = vol->SetMasterVolume(g_mute_level, nullptr);
    if (FAILED(hr)) {  // not ducked: the app must not bring this sound up (it would clip)
        printf("{\"event\": \"failed\", \"pid\": %lu, \"device\": \"%s\", \"hr\": %lu}\n", (unsigned long)spid,
               json_escape(wide_to_utf8(device.c_str())).c_str(), (unsigned long)hr);
    } else {
        printf("{\"event\": \"muted\", \"pid\": %lu, \"level\": %.4f, \"device\": \"%s\"}\n", (unsigned long)spid,
               original, json_escape(wide_to_utf8(device.c_str())).c_str());
    }
    fflush(stdout);
}

static int run_mute(DWORD pid, bool tree, float level) {
    g_mute_target = pid; g_mute_tree = tree; g_mute_level = level;
    say("muting %lu (tree)", (unsigned long)pid);
    return watch_sessions(mute_session);  // mute-only: the Python side restores from its saved per-app level
}

// ---- giving the volume back (keep) ---------------------------------------------------------------------------
// The call app was muted and could not get its level back: it was closed before the translation was turned off (no
// session to set), or its device was unplugged. Windows keeps an app's volume per device and persists the muted 0,
// so its next session there starts silent. `keep` waits for it: a session of a process named --exe (or of one of
// its children) on one of the --device endpoints (any without) that is at 0 gets --level; one found not at 0 (the
// user raised it, or Windows did not keep the 0) is left as it is. Either way a "restored" line says that device is
// done; the Python side stops the helper when every device is.

static std::wstring g_keep_exe;
static float g_keep_level = 1.0f;
static std::set<std::wstring> g_keep_devices, g_keep_left;  // the devices asked for; those not done yet
static HANDLE g_keep_done = CreateEventW(nullptr, TRUE, FALSE, nullptr);

static void keep_session(ComPtr<IAudioSessionControl2>& ctl2, const std::wstring& device) {
    if (!g_keep_devices.empty() && !g_keep_devices.count(device)) return;
    DWORD spid = 0; ctl2->GetProcessId(&spid);
    if (!spid || !in_exe_tree(spid, g_keep_exe)) return;
    ComPtr<ISimpleAudioVolume> vol;
    if (FAILED(ctl2.As(&vol))) return;
    float v = 1.0f;
    if (FAILED(vol->GetMasterVolume(&v))) return;
    if (v < 0.01f && FAILED(vol->SetMasterVolume(g_keep_level, nullptr))) return;
    printf("{\"event\": \"restored\", \"pid\": %lu, \"device\": \"%s\", \"was\": %.4f}\n", (unsigned long)spid,
           json_escape(wide_to_utf8(device.c_str())).c_str(), v);
    fflush(stdout);  // fails without a reader (the app quit): the volume is set all the same
    std::lock_guard<std::mutex> lk(g_watch_lock);
    g_keep_left.erase(device);
    if (g_keep_devices.empty() || g_keep_left.empty()) SetEvent(g_keep_done);  // every device done (any: the first)
}

static int run_keep(const std::wstring& exe, float level, const std::set<std::wstring>& devices, DWORD seconds) {
    g_keep_exe = exe; g_keep_level = level; g_keep_devices = g_keep_left = devices;
    say("giving %ls its volume back (%.2f) on %zu device(s), 0 = any, for %lu s at most", exe.c_str(), level,
        devices.size(), (unsigned long)seconds);
    return watch_sessions(keep_session, g_keep_done, (ULONGLONG)seconds * 1000);
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
    if (argc < 2) quit(USAGE, "usage: peresmeshnik-loopback check|console|list|tap|volume|mute|keep|watch");
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
        float level = 0.0f;
        for (int i = 0; i < argc; i++) if (!wcscmp(argv[i], L"--level") && i + 1 < argc) level = (float)wcstod(argv[i + 1], nullptr);
        if (level < 0.0f || level >= 0.01f) quit(USAGE, "mute --level must be below 0.01 (a near silence)");
        code = run_mute(pid, has_flag(argc, argv, L"--tree"), level);
    } else if (cmd == L"keep") {
        std::wstring exe; float level = -1.0f; std::set<std::wstring> devices; DWORD seconds = 14 * 24 * 3600;
        for (int i = 2; i + 1 < argc; i++) {
            if (!wcscmp(argv[i], L"--exe")) exe = argv[++i];
            else if (!wcscmp(argv[i], L"--level")) level = (float)wcstod(argv[++i], nullptr);
            else if (!wcscmp(argv[i], L"--device")) devices.insert(argv[++i]);
            else if (!wcscmp(argv[i], L"--for")) seconds = wcstoul(argv[++i], nullptr, 10);
        }
        if (exe.empty() || level < 0.01f || level > 1.0f || !seconds) quit(USAGE, "keep needs --exe NAME --level L (0.01..1)");
        code = run_keep(exe, level, devices, seconds);
    } else {
        say("unknown command");
    }
    CoUninitialize();
    return code;
}
