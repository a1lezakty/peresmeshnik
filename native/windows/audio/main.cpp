// peresmeshnik-audio: the WASAPI side of Peresmeshnik's "me -> them" on Windows, as native/macos/audio is on macOS.
// One small process per stream; Python only starts, reads and stops them.
//
//   peresmeshnik-audio list                                   the endpoints, JSON
//   peresmeshnik-audio record [--from DEV] [--follow] [--rate 16000]   float32le mono on stdout
//   peresmeshnik-audio play --to DEV [--rate 24000] [--latency 40]     float32le mono from stdin; exits once it played,
//                                                             after EOF; plays only with --latency ms queued (after a
//                                                             stall too)
//   peresmeshnik-audio loop [--from DEV] --to DEV [--watch DEV]        the microphone into DEV (CABLE Input) until stdin
//                                                             closes; with --watch only while another process records
//                                                             that input (CABLE Output: the call app)
//
// DEV is an endpoint id, else a friendly name. --from '' is the default input. --follow (always for loop): while DEV
// is not there, the default input if it is a real one, else the first real one (not virtual: VB-Cable and the like are
// software devices); back to DEV when it returns. Windows converts rate and channels (AUTOCONVERTPCM): a format change
// of a device only reopens it. record and loop end when stdin closes (the app is gone: no microphone stays open).
// Events on stderr, a line each: "using <id>\t<name>", "lost <id>\t<name>", "none"; loop also "idle" (nobody records
// --watch: the microphone is closed, no microphone icon in the taskbar) and "call" (somebody does).
// Exit codes: 0 done, 2 usage, 3 no such device, 4 the device went away, 5 WASAPI / COM error, 6 no microphone
// permission (Windows privacy settings), 7 (play) the output stream was invalidated (a format change): start it again.
// Nothing polls: WASAPI events, endpoint and audio session notifications, a blocked read of stdin.
//
// Build: native/windows/audio/build.cmd [out.exe] (cl via vswhere, as native/windows/loopback).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <devicetopology.h>
#include <mmdeviceapi.h>
#include <wrl/implements.h>

#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cwctype>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using Microsoft::WRL::ClassicCom;
using Microsoft::WRL::ComPtr;
using Microsoft::WRL::FtmBase;
using Microsoft::WRL::Make;
using Microsoft::WRL::RuntimeClass;
using Microsoft::WRL::RuntimeClassFlags;

enum Code { OK = 0, USAGE = 2, NO_DEVICE = 3, GONE = 4, WASAPI = 5, PERMISSION = 6, RESTART = 7 };

// Here, not from functiondiscoverykeys_devpkey.h: without INITGUID those are only declared (a link error).
static const PROPERTYKEY FRIENDLY_NAME = {{0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};
static const PROPERTYKEY DEVICE_FORMAT = {{0xf19f064d, 0x082c, 0x4e27, {0xbc, 0x73, 0x68, 0x82, 0xa1, 0xbb, 0x8e, 0x4c}}, 0};

static void say(const std::string& s) {
    fprintf(stderr, "%s\n", s.c_str());
    fflush(stderr);
}

[[noreturn]] static void die(int code, const std::string& s) {
    say("peresmeshnik-audio: " + s);
    fflush(stdout);
    ExitProcess(code);  // the audio engine drops our streams with the process
}

static std::string utf8(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

static std::string hex(HRESULT hr) {
    char b[16];
    snprintf(b, sizeof(b), "0x%08lx", (unsigned long)hr);
    return b;
}

static std::string json(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += c;
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += c;
    }
    return o + "\"";
}

// ---- devices ------------------------------------------------------------------------------------

struct Device {
    std::wstring id;
    std::string name, transport;
    bool capture = false;
    DWORD rate = 0;
    WORD channels = 0;
    std::string label() const { return utf8(id) + "\t" + name; }
};

static ComPtr<IMMDeviceEnumerator> g_enum;

// "virt" for a software device (VB-Cable, Voicemeeter: root-enumerated), from the KS filter the endpoint sits on.
static std::string transport_of(IMMDevice* d) {
    ComPtr<IDeviceTopology> topo;
    ComPtr<IConnector> con;
    LPWSTR other = nullptr;
    if (FAILED(d->Activate(__uuidof(IDeviceTopology), CLSCTX_ALL, nullptr, (void**)topo.GetAddressOf())) ||
        FAILED(topo->GetConnector(0, &con)) || FAILED(con->GetDeviceIdConnectedTo(&other)) || !other)
        return "";
    std::wstring s(other);
    CoTaskMemFree(other);
    std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) { return (wchar_t)towlower(c); });
    if (s.find(L"\\\\?\\root#") != std::wstring::npos) return "virt";
    if (s.find(L"bth") != std::wstring::npos) return "bluetooth";
    if (s.find(L"\\\\?\\usb#") != std::wstring::npos) return "usb";
    return "";
}

static Device info(IMMDevice* d, bool capture) {
    Device r;
    r.capture = capture;
    LPWSTR id = nullptr;
    if (SUCCEEDED(d->GetId(&id)) && id) {
        r.id = id;
        CoTaskMemFree(id);
    }
    ComPtr<IPropertyStore> ps;
    if (SUCCEEDED(d->OpenPropertyStore(STGM_READ, &ps))) {
        PROPVARIANT v;
        PropVariantInit(&v);
        if (SUCCEEDED(ps->GetValue(FRIENDLY_NAME, &v)) && v.vt == VT_LPWSTR) r.name = utf8(v.pwszVal);
        PropVariantClear(&v);
        if (SUCCEEDED(ps->GetValue(DEVICE_FORMAT, &v)) && v.vt == VT_BLOB && v.blob.cbSize >= sizeof(WAVEFORMATEX)) {
            auto* f = (const WAVEFORMATEX*)v.blob.pBlobData;
            r.rate = f->nSamplesPerSec;
            r.channels = f->nChannels;
        }
        PropVariantClear(&v);
    }
    r.channels = std::max<WORD>(r.channels, 1);
    r.transport = transport_of(d);
    return r;
}

static std::vector<Device> devices(EDataFlow flow) {
    std::vector<Device> out;
    ComPtr<IMMDeviceCollection> c;
    UINT n = 0;
    if (FAILED(g_enum->EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE, &c)) || FAILED(c->GetCount(&n))) return out;
    for (UINT i = 0; i < n; i++) {
        ComPtr<IMMDevice> d;
        if (SUCCEEDED(c->Item(i, &d))) out.push_back(info(d.Get(), flow == eCapture));
    }
    return out;
}

static std::wstring default_id(EDataFlow flow) {
    ComPtr<IMMDevice> d;
    LPWSTR id = nullptr;
    std::wstring r;
    if (SUCCEEDED(g_enum->GetDefaultAudioEndpoint(flow, eConsole, &d)) && SUCCEEDED(d->GetId(&id)) && id) {
        r = id;
        CoTaskMemFree(id);
    }
    return r;
}

static std::optional<Device> find(const std::vector<Device>& all, const std::wstring& key) {
    for (auto& d : all) if (d.id == key) return d;
    std::string k = utf8(key);
    for (auto& d : all) if (d.name == k) return d;
    return std::nullopt;
}

static bool active(const std::wstring& id) {
    ComPtr<IMMDevice> d;
    DWORD st = 0;
    return SUCCEEDED(g_enum->GetDevice(id.c_str(), &d)) && SUCCEEDED(d->GetState(&st)) && st == DEVICE_STATE_ACTIVE;
}

// Events the main thread waits on; the COM callbacks only set them.
static HANDLE g_devices = CreateEventW(nullptr, FALSE, FALSE, nullptr);   // an endpoint came, went, or the default changed
static HANDLE g_sessions = CreateEventW(nullptr, FALSE, FALSE, nullptr);  // an audio session of --watch changed
static HANDLE g_stdin = CreateEventW(nullptr, TRUE, FALSE, nullptr);      // stdin closed

struct Endpoints : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IMMNotificationClient> {
    STDMETHOD(OnDefaultDeviceChanged)(EDataFlow, ERole, LPCWSTR) override { SetEvent(g_devices); return S_OK; }
    STDMETHOD(OnDeviceStateChanged)(LPCWSTR, DWORD) override { SetEvent(g_devices); return S_OK; }
    STDMETHOD(OnDeviceAdded)(LPCWSTR) override { SetEvent(g_devices); return S_OK; }
    STDMETHOD(OnDeviceRemoved)(LPCWSTR) override { SetEvent(g_devices); return S_OK; }
    STDMETHOD(OnPropertyValueChanged)(LPCWSTR, const PROPERTYKEY) override { return S_OK; }
};

static DWORD WINAPI wait_stdin(LPVOID) {
    char b[256];
    DWORD n = 0;
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    while (ReadFile(in, b, sizeof(b), &n, nullptr) && n) {}
    SetEvent(g_stdin);
    return 0;
}

// ---- streams ------------------------------------------------------------------------------------

struct Ring {
    std::vector<float> buf;
    size_t head = 0, count = 0;
    explicit Ring(size_t cap) : buf(cap) {}
    size_t space() const { return buf.size() - count; }
    size_t write(const float* p, size_t n) {
        n = std::min(n, space());
        for (size_t i = 0; i < n; i++) buf[(head + count + i) % buf.size()] = p[i];
        count += n;
        return n;
    }
    size_t read(float* p, size_t n) {
        n = std::min(n, count);
        for (size_t i = 0; i < n; i++) p[i] = buf[(head + i) % buf.size()];
        head = (head + n) % buf.size();
        count -= n;
        return n;
    }
    void trim(size_t keep) {  // drop the oldest: the microphone runs ahead of the output's clock
        if (count > keep) head = (head + count - keep) % buf.size(), count = keep;
    }
};

// A shared-mode, event-driven stream at mono float32 `rate`: Windows converts to and from the device's own format.
struct Stream {
    Device dev;
    ComPtr<IAudioClient> client;
    ComPtr<IAudioCaptureClient> cap;
    ComPtr<IAudioRenderClient> ren;
    HANDLE evt = nullptr;
    UINT32 size = 0;  // the buffer, frames

    HRESULT open(const Device& d, DWORD rate, DWORD ms) {
        dev = d;
        ComPtr<IMMDevice> m;
        HRESULT hr = g_enum->GetDevice(d.id.c_str(), &m);
        if (SUCCEEDED(hr)) hr = m->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)client.GetAddressOf());
        if (FAILED(hr)) return hr;
        WAVEFORMATEX f = {};
        f.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
        f.nChannels = 1;
        f.nSamplesPerSec = rate;
        f.wBitsPerSample = 32;
        f.nBlockAlign = 4;
        f.nAvgBytesPerSec = rate * 4;
        DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                      AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        if (FAILED(hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, (REFERENCE_TIME)ms * 10000, 0, &f, nullptr)))
            return hr;
        evt = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (FAILED(hr = client->SetEventHandle(evt)) || FAILED(hr = client->GetBufferSize(&size))) return hr;
        hr = d.capture ? client->GetService(IID_PPV_ARGS(&cap)) : client->GetService(IID_PPV_ARGS(&ren));
        return FAILED(hr) ? hr : client->Start();
    }

    void close() {
        if (client) client->Stop();
        cap.Reset();
        ren.Reset();
        client.Reset();
        if (evt) CloseHandle(evt);
        evt = nullptr;
    }

    // Every packet waiting, to `emit`; false when the stream is invalidated (gone, or its format changed).
    template <class F> bool read(F&& emit) {
        static std::vector<float> zeros;
        UINT32 next = 0;
        HRESULT hr;
        while (SUCCEEDED(hr = cap->GetNextPacketSize(&next)) && next) {
            BYTE* data = nullptr;
            UINT32 n = 0;
            DWORD fl = 0;
            if (FAILED(hr = cap->GetBuffer(&data, &n, &fl, nullptr, nullptr))) break;
            if (fl & AUDCLNT_BUFFERFLAGS_SILENT) {
                zeros.assign(n, 0.0f);
                emit(zeros.data(), (size_t)n);
            } else {
                emit((const float*)data, (size_t)n);
            }
            cap->ReleaseBuffer(n);
        }
        return SUCCEEDED(hr);
    }

    // Free room in the buffer, frames; -1 when invalidated.
    long room() {
        UINT32 pad = 0;
        return SUCCEEDED(client->GetCurrentPadding(&pad)) ? (long)(size - pad) : -1;
    }

    bool write(const float* p, UINT32 n) {
        BYTE* buf = nullptr;
        if (!n) return true;
        if (FAILED(ren->GetBuffer(n, &buf))) return false;
        memcpy(buf, p, (size_t)n * 4);
        return SUCCEEDED(ren->ReleaseBuffer(n, 0));
    }
};

// Out of `ring` into `out` on its event: starts with `prime` frames queued, again after a stall (nothing left in the
// ring nor in the device). False when the stream is invalidated.
static bool render(Stream& out, Ring& ring, size_t prime, bool& primed, bool flush_all, std::vector<float>& tmp) {
    long room = out.room();
    if (room < 0) return false;
    if (!primed && (ring.count >= prime || flush_all)) primed = true;
    size_t n = primed ? std::min<size_t>((size_t)room, ring.count) : 0;
    if (tmp.size() < n) tmp.resize(n);
    ring.read(tmp.data(), n);
    if (!out.write(tmp.data(), (UINT32)n)) return false;
    if (primed && !n && room == (long)out.size && !flush_all) primed = false;
    return true;
}

// The microphone: `saved` while it is there, else (follow) the default input or the first real one.
struct Input {
    std::wstring saved, exclude;
    bool follow = true;
    DWORD rate = 16000;
    Stream s;
    bool open = false;
    std::string said;

    void event(const std::string& e) {
        if (e != said) say(e);
        said = e;
    }

    std::optional<Device> resolve() {
        std::vector<Device> ins;
        for (auto& d : devices(eCapture)) if (d.id != exclude) ins.push_back(d);
        if (!saved.empty()) if (auto d = find(ins, saved)) return d;
        if (!follow) return std::nullopt;
        std::wstring def = default_id(eCapture);
        for (auto& d : ins) if (d.id == def && d.transport != "virt") return d;
        for (auto& d : ins) if (d.transport != "virt") return d;
        return std::nullopt;
    }

    // On a device change, or `broken`: the stream stopped working (it went away, or a format change).
    void update(bool broken = false) {
        auto want = resolve();
        if (open && !broken && want && want->id == s.dev.id) return;
        if (open) {
            Device was = s.dev;
            s.close();
            open = false;
            if (!active(was.id)) {
                say("lost " + was.label());
                if (!follow) die(GONE, was.name + " went away");
            }
        }
        if (!want) {
            if (!follow) die(NO_DEVICE, "no input device " + utf8(saved));
            event("none");
            return;
        }
        HRESULT hr = s.open(*want, rate, 100);
        if (FAILED(hr)) {
            s.close();
            if (hr == E_ACCESSDENIED) die(PERMISSION, "no permission to use the microphone");
            if (!follow) die(WASAPI, "cannot record " + want->name + ": " + hex(hr));
            say("peresmeshnik-audio: cannot record " + want->name + ": " + hex(hr));
            event("none");
            return;
        }
        open = true;
        event("using " + want->label());
    }

    void close() {
        if (open) s.close();
        open = false;
        said.clear();
    }
};

// ---- who records --watch (the call app on CABLE Output) -----------------------------------------

struct SessionEvents : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IAudioSessionEvents> {
    STDMETHOD(OnDisplayNameChanged)(LPCWSTR, LPCGUID) override { return S_OK; }
    STDMETHOD(OnIconPathChanged)(LPCWSTR, LPCGUID) override { return S_OK; }
    STDMETHOD(OnSimpleVolumeChanged)(float, BOOL, LPCGUID) override { return S_OK; }
    STDMETHOD(OnChannelVolumeChanged)(DWORD, float[], DWORD, LPCGUID) override { return S_OK; }
    STDMETHOD(OnGroupingParamChanged)(LPCGUID, LPCGUID) override { return S_OK; }
    STDMETHOD(OnStateChanged)(AudioSessionState) override { SetEvent(g_sessions); return S_OK; }
    STDMETHOD(OnSessionDisconnected)(AudioSessionDisconnectReason) override { SetEvent(g_sessions); return S_OK; }
};

struct SessionCreated : RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IAudioSessionNotification> {
    STDMETHOD(OnSessionCreated)(IAudioSessionControl*) override { SetEvent(g_sessions); return S_OK; }
};

// The capture sessions of one endpoint: a call app recording it has an active one. Each session's state changes are
// watched (registered here, on the main thread; the callbacks only set g_sessions).
struct Readers {
    struct Watched {
        ComPtr<IAudioSessionControl> ctl;
        ComPtr<SessionEvents> events;
    };
    ComPtr<IAudioSessionManager2> mgr;
    ComPtr<SessionCreated> created;
    std::map<std::wstring, Watched> watched;

    bool init(const Device& d) {
        ComPtr<IMMDevice> m;
        ComPtr<IAudioSessionEnumerator> e;
        created = Make<SessionCreated>();
        return SUCCEEDED(g_enum->GetDevice(d.id.c_str(), &m)) &&
               SUCCEEDED(m->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr, (void**)mgr.GetAddressOf())) &&
               SUCCEEDED(mgr->GetSessionEnumerator(&e)) &&  // once before OnSessionCreated can fire (documented)
               SUCCEEDED(mgr->RegisterSessionNotification(created.Get()));
    }

    bool any() {
        ComPtr<IAudioSessionEnumerator> e;
        int n = 0;
        if (FAILED(mgr->GetSessionEnumerator(&e)) || FAILED(e->GetCount(&n))) return false;
        std::map<std::wstring, Watched> now;
        bool reading = false;
        for (int i = 0; i < n; i++) {
            ComPtr<IAudioSessionControl> c;
            ComPtr<IAudioSessionControl2> c2;
            if (FAILED(e->GetSession(i, &c)) || FAILED(c.As(&c2))) continue;
            LPWSTR iid = nullptr;
            std::wstring key = std::to_wstring(i);
            if (SUCCEEDED(c2->GetSessionInstanceIdentifier(&iid)) && iid) {
                key = iid;
                CoTaskMemFree(iid);
            }
            AudioSessionState st = AudioSessionStateInactive;
            DWORD pid = 0;
            c->GetState(&st);
            c2->GetProcessId(&pid);
            if (st == AudioSessionStateExpired) continue;
            auto it = watched.find(key);
            if (it != watched.end()) {
                now[key] = it->second;
                watched.erase(it);
            } else {
                Watched w{c, Make<SessionEvents>()};
                if (SUCCEEDED(c->RegisterAudioSessionNotification(w.events.Get()))) now[key] = w;
            }
            if (st == AudioSessionStateActive && pid != GetCurrentProcessId()) reading = true;
        }
        for (auto& kv : watched) kv.second.ctl->UnregisterAudioSessionNotification(kv.second.events.Get());
        watched.swap(now);
        return reading;
    }
};

// ---- commands -----------------------------------------------------------------------------------

static int run_list() {
    std::wstring din = default_id(eCapture), dout = default_id(eRender);
    std::string s = "[";
    for (EDataFlow flow : {eCapture, eRender}) {
        for (auto& d : devices(flow)) {
            if (s.size() > 1) s += ", ";
            s += "{\"uid\": " + json(utf8(d.id)) + ", \"name\": " + json(d.name) +
                 ", \"inputs\": " + std::to_string(d.capture ? d.channels : 0) +
                 ", \"outputs\": " + std::to_string(d.capture ? 0 : d.channels) + ", \"rate\": " + std::to_string(d.rate) +
                 ", \"transport\": " + json(d.transport) + ", \"default_input\": " + (d.id == din ? "true" : "false") +
                 ", \"default_output\": " + (d.id == dout ? "true" : "false") + "}";
        }
    }
    s += "]\n";
    fwrite(s.data(), 1, s.size(), stdout);
    fflush(stdout);
    return OK;
}

static int run_record(const std::wstring& from, bool follow, DWORD rate) {
    Input in;
    in.saved = from;
    in.follow = follow || from.empty();
    in.rate = rate;
    in.update();
    CreateThread(nullptr, 0, wait_stdin, nullptr, 0, nullptr);
    for (;;) {
        HANDLE w[3] = {g_stdin, g_devices, in.s.evt};
        DWORD r = WaitForMultipleObjects(in.open ? 3 : 2, w, FALSE, INFINITE);
        if (r == WAIT_OBJECT_0) return OK;
        if (r == WAIT_OBJECT_0 + 1) {
            in.update();
            continue;
        }
        bool ok = in.s.read([](const float* p, size_t n) {
            if (fwrite(p, 4, n, stdout) < n) ExitProcess(OK);  // the reader is gone
        });
        if (fflush(stdout)) ExitProcess(OK);
        if (!ok) in.update(true);
    }
}

// play: stdin -> `ring` on its own thread; a full ring blocks it (the writer waits, as on a pipe).
static std::mutex g_lock;
static std::condition_variable g_room;
static Ring* g_ring = nullptr;
static bool g_eof = false;

static DWORD WINAPI feed(LPVOID) {
    std::vector<char> b(16384);
    size_t have = 0;
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    for (;;) {
        DWORD n = 0;
        if (!ReadFile(in, b.data() + have, (DWORD)(b.size() - have), &n, nullptr) || !n) break;
        have += n;
        size_t k = have / 4, done = 0;
        {
            std::unique_lock<std::mutex> lk(g_lock);
            while (done < k) {
                done += g_ring->write((const float*)b.data() + done, k - done);
                if (done < k) g_room.wait(lk);
            }
        }
        memmove(b.data(), b.data() + k * 4, have - k * 4);
        have -= k * 4;
    }
    std::lock_guard<std::mutex> lk(g_lock);
    g_eof = true;
    return 0;
}

static int run_play(const std::wstring& to, DWORD rate, DWORD latency) {
    auto target = find(devices(eRender), to);
    if (!target) die(NO_DEVICE, "no output device " + utf8(to));
    Stream out;
    HRESULT hr = out.open(*target, rate, std::max<DWORD>(latency, 30));
    if (FAILED(hr)) die(WASAPI, "cannot play into " + target->name + ": " + hex(hr));
    Ring ring(rate / 4);
    g_ring = &ring;
    CreateThread(nullptr, 0, feed, nullptr, 0, nullptr);
    std::vector<float> tmp;
    bool primed = false;
    const size_t prime = (size_t)rate * latency / 1000;
    for (;;) {
        HANDLE w[2] = {out.evt, g_devices};
        if (WaitForMultipleObjects(2, w, FALSE, INFINITE) == WAIT_OBJECT_0 + 1) {
            if (!active(target->id)) die(GONE, target->name + " went away");
            continue;
        }
        bool ok, end;
        {
            std::lock_guard<std::mutex> lk(g_lock);
            ok = render(out, ring, prime, primed, g_eof, tmp);
            end = g_eof && !ring.count;
        }
        g_room.notify_one();
        if (!ok) {
            if (!active(target->id)) die(GONE, target->name + " went away");
            die(RESTART, "the output stream of " + target->name + " was reset");  // the app opens a new one
        }
        if (end && out.room() == (long)out.size) break;  // all of it went to the device
    }
    Sleep(30);  // the engine's last period
    out.close();
    return OK;
}

static int run_loop(const std::wstring& from, const std::wstring& to, const std::wstring& watch) {
    auto target = find(devices(eRender), to);
    if (!target) die(NO_DEVICE, "no output device " + utf8(to));
    std::optional<Device> watched;
    Readers readers;
    if (!watch.empty()) {
        if (!(watched = find(devices(eCapture), watch))) die(NO_DEVICE, "no input device " + utf8(watch));
        if (!readers.init(*watched)) die(WASAPI, "cannot watch who records " + watched->name);
    }
    const DWORD rate = 48000;
    const size_t keep = rate * 60 / 1000, prime = rate * 20 / 1000;  // at most 60 ms queued: two clocks
    Ring ring(rate / 2);
    std::vector<float> tmp;
    Input in;
    in.saved = from;
    in.exclude = watched ? watched->id : L"";
    in.rate = rate;
    Stream out;
    bool running = false, primed = false;
    auto start = [&] {
        say("call");
        HRESULT hr = out.open(*target, rate, 40);
        if (FAILED(hr)) die(WASAPI, "cannot play into " + target->name + ": " + hex(hr));
        ring.trim(0);
        primed = false;
        in.update();
        running = true;
    };
    auto end = [&] {
        in.close();  // the microphone, then the output
        out.close();
        running = false;
        say("idle");
    };
    if (!watched || readers.any()) start();
    else say("idle");
    CreateThread(nullptr, 0, wait_stdin, nullptr, 0, nullptr);
    for (;;) {
        HANDLE w[5] = {g_stdin, g_devices, g_sessions, out.evt, in.s.evt};
        DWORD n = running ? (in.open ? 5 : 4) : 3;
        DWORD r = WaitForMultipleObjects(n, w, FALSE, INFINITE) - WAIT_OBJECT_0;
        if (r == 0) break;
        if (r == 1) {
            if (!active(target->id)) die(GONE, target->name + " went away");
            if (watched && !active(watched->id)) die(GONE, watched->name + " went away");
            if (running) in.update();
        } else if (r == 2) {
            bool now = readers.any();
            if (now && !running) start();
            else if (!now && running) end();
        } else if (r == 3) {
            if (!render(out, ring, prime, primed, false, tmp)) {  // a format change of the output: open it again
                out.close();
                if (!active(target->id)) die(GONE, target->name + " went away");
                HRESULT hr = out.open(*target, rate, 40);
                if (FAILED(hr)) die(WASAPI, "cannot play into " + target->name + ": " + hex(hr));
            }
        } else if (r == 4) {
            bool ok = in.s.read([&](const float* p, size_t k) {
                ring.write(p, k);
                if (ring.count > keep) ring.trim(keep / 2);
            });
            if (!ok) in.update(true);
        }
    }
    if (running) end();
    return OK;
}

// ---- entry --------------------------------------------------------------------------------------

static std::vector<std::wstring> g_args;

static std::optional<std::wstring> opt(const wchar_t* name) {
    for (size_t i = 0; i + 1 < g_args.size(); i++) if (g_args[i] == name) return g_args[i + 1];
    return std::nullopt;
}

static bool flag(const wchar_t* name) { return std::find(g_args.begin(), g_args.end(), name) != g_args.end(); }

static DWORD number(const wchar_t* name, DWORD def, DWORD lo, DWORD hi) {
    auto s = opt(name);
    if (!s) return def;
    DWORD v = wcstoul(s->c_str(), nullptr, 10);
    if (v < lo || v > hi) die(USAGE, "bad " + utf8(name) + " " + utf8(*s));
    return v;
}

int wmain(int argc, wchar_t** argv) {
    _setmode(_fileno(stdout), _O_BINARY);  // PCM and UTF-8 JSON as they are
    _setmode(_fileno(stdin), _O_BINARY);
    std::wstring cmd = argc > 1 ? argv[1] : L"";
    g_args.assign(argv + std::min(argc, 2), argv + argc);
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) die(WASAPI, "CoInitializeEx failed");
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&g_enum))))
        die(WASAPI, "no audio device enumerator (is the Windows Audio service running?)");
    auto endpoints = Make<Endpoints>();
    if (cmd != L"list" && FAILED(g_enum->RegisterEndpointNotificationCallback(endpoints.Get())))
        die(WASAPI, "no device notifications");
    auto to = opt(L"--to");
    if ((cmd == L"play" || cmd == L"loop") && (!to || to->empty())) die(USAGE, "--to DEV is needed");
    int code;
    if (cmd == L"list") code = run_list();
    else if (cmd == L"record")
        code = run_record(opt(L"--from").value_or(L""), flag(L"--follow"), number(L"--rate", 16000, 8000, 192000));
    else if (cmd == L"play") code = run_play(*to, number(L"--rate", 24000, 8000, 192000), number(L"--latency", 0, 0, 999));
    else if (cmd == L"loop") code = run_loop(opt(L"--from").value_or(L""), *to, opt(L"--watch").value_or(L""));
    else die(USAGE, "usage: peresmeshnik-audio list | record [--from DEV] [--follow] [--rate R] | "
                    "play --to DEV [--rate R] [--latency MS] | loop [--from DEV] --to DEV [--watch DEV]");
    fflush(stdout);
    ExitProcess(code);  // not through the static destructors: COM objects stay registered until the end
}
