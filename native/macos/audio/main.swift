// peresmeshnik-audio: the Core Audio side of Peresmeshnik's "me -> them" on macOS. One small process per stream,
// as parec / pacat / module-loopback are on Linux; Python only starts, reads and stops them.
//
//   peresmeshnik-audio list                               the devices, JSON
//   peresmeshnik-audio permission                         the microphone permission: authorized | denied | ...
//   peresmeshnik-audio record [--from DEV] [--follow] [--rate 16000]   float32le mono on stdout
//   peresmeshnik-audio play --to DEV [--rate 24000] [--latency 40]   float32le mono from stdin; exits once it played,
//                                                         after EOF; plays only with --latency ms queued (after a stall too)
//   peresmeshnik-audio loop [--from DEV] --to DEV [--always]   the microphone into DEV (BlackHole 2ch) until stdin
//                                                         closes; macOS 14+: only while another process records DEV
//   peresmeshnik-audio aggregate --uid UID --of DEV       (tests) a device that comes and goes: lives until stdin closes
//
// DEV is a device UID, else a name. --from '' is the default input. --follow (always for loop): while DEV is not there,
// the default input, else the first real one (not virtual, not an aggregate); back to DEV when it returns. A sample
// rate change of the device reopens it. record and loop end when stdin closes (the app is gone: no microphone stays
// open). Events on stderr, a line each: "using <uid>\t<name>", "lost <uid>\t<name>", "none"; loop also "idle" (nobody
// records DEV: the microphone is closed, no orange dot, AirPods keep their music profile) and "call" (somebody does).
// Exit codes: 0 done, 2 usage, 3 no such device, 4 the device went away, 5 Core Audio error, 6 no microphone permission,
// 7 (play) the output's sample rate changed: start it again (it reads the new rate). loop follows a rate change itself.
// SIGTERM / SIGINT / SIGHUP end it cleanly: every IOProc is stopped before the exit. A process that just dies (SIGKILL)
// with its IOProc running can leave its last IO buffer in coreaudiod's mix, repeated: seen on the CI runner, a 10.7 ms
// buffer looping in BlackHole with no process left. So the app stops helpers with SIGTERM, never SIGKILL.
//
// Build: native/macos/audio/build.sh -> native/macos/audio/build/peresmeshnik-audio (one swiftc, with an embedded
// Info.plist that carries NSMicrophoneUsageDescription)

import AVFoundation
import CoreAudio
import Foundation
import os

signal(SIGPIPE, SIG_IGN)
setvbuf(stderr, nil, _IOLBF, 0)

func say(_ s: String) {
    fputs(s + "\n", stderr)
}

/// The IOProcs running now: stopped before any exit; so are the aggregate devices we made.
let ioLock = NSLock()
var ioProcs: [AudioObjectID: [AudioDeviceIOProcID]] = [:]
var aggregates: [AudioObjectID] = []

func started(_ dev: AudioObjectID, _ proc: AudioDeviceIOProcID) {
    ioLock.lock()
    ioProcs[dev, default: []].append(proc)
    ioLock.unlock()
}

func stopIO(_ dev: AudioObjectID, _ proc: AudioDeviceIOProcID) {
    ioLock.lock()
    defer { ioLock.unlock() }
    let key = unsafeBitCast(proc, to: UnsafeRawPointer.self)  // a C function pointer: not Equatable
    guard let i = ioProcs[dev]?.firstIndex(where: { unsafeBitCast($0, to: UnsafeRawPointer.self) == key }) else { return }
    ioProcs[dev]!.remove(at: i)
    AudioDeviceStop(dev, proc)
    AudioDeviceDestroyIOProcID(dev, proc)
}

func quit(_ code: Int32) -> Never {
    ioLock.lock()
    for (dev, procs) in ioProcs {
        for p in procs {
            AudioDeviceStop(dev, p)
            AudioDeviceDestroyIOProcID(dev, p)
        }
    }
    ioProcs = [:]
    for a in aggregates { AudioHardwareDestroyAggregateDevice(a) }
    // _exit, not exit: exit() flushes stdout, which a pump thread may hold in a blocked write (once in ~400 stops a
    // helper hung there). stderr is line-buffered, stdout flushed after each write. Under the lock: nothing starts after.
    _exit(code)
}

func die(_ code: Int32, _ s: String) -> Never {
    say("peresmeshnik-audio: " + s)
    quit(code)
}

let signalSources: [DispatchSourceSignal] = [SIGTERM, SIGINT, SIGHUP].map { sig in
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .global())
    src.setEventHandler { quit(0) }
    src.resume()
    return src
}

let system = AudioObjectID(kAudioObjectSystemObject)
let global = kAudioObjectPropertyScopeGlobal

func addr(_ sel: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = global) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func get<T>(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ zero: T, _ scope: AudioObjectPropertyScope = global) -> T? {
    var a = addr(sel, scope), v = zero, size = UInt32(MemoryLayout<T>.size)
    let st = withUnsafeMutablePointer(to: &v) { AudioObjectGetPropertyData(obj, &a, 0, nil, &size, $0) }
    return st == noErr ? v : nil
}

func string(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector) -> String {
    var a = addr(sel), size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var v: Unmanaged<CFString>?
    guard AudioObjectGetPropertyData(obj, &a, 0, nil, &size, &v) == noErr, let s = v else { return "" }
    return s.takeRetainedValue() as String
}

func objects(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = global) -> [AudioObjectID] {
    var a = addr(sel, scope), size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(obj, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(obj, &a, 0, nil, &size, &ids) == noErr else { return [] }
    return Array(ids.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
}

func deviceIDs() -> [AudioObjectID] { objects(system, kAudioHardwarePropertyDevices) }

func channels(_ d: AudioObjectID, input: Bool) -> Int {
    var a = addr(kAudioDevicePropertyStreamConfiguration, input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(d, &a, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let p = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
    defer { p.deallocate() }
    guard AudioObjectGetPropertyData(d, &a, 0, nil, &size, p) == noErr else { return 0 }
    return UnsafeMutableAudioBufferListPointer(p.assumingMemoryBound(to: AudioBufferList.self)).reduce(0) { $0 + Int($1.mNumberChannels) }
}

func fourcc(_ v: UInt32) -> String {
    v == 0 ? "" : String(decoding: [24, 16, 8, 0].map { UInt8((v >> $0) & 0xff) }, as: UTF8.self)
}

struct Device {
    let id: AudioObjectID, uid: String, name: String, inputs: Int, outputs: Int, rate: Double, transport: String
    var real: Bool {  // not BlackHole & co, not an aggregate; Krisp's is the microphone processed (audio PROCESSED)
        transport == "virt" ? name.contains("Krisp") : transport != "grup"
    }
    var label: String { "\(uid)\t\(name)" }
}

func device(_ id: AudioObjectID) -> Device {
    Device(id: id, uid: string(id, kAudioDevicePropertyDeviceUID), name: string(id, kAudioObjectPropertyName),
           inputs: channels(id, input: true), outputs: channels(id, input: false),
           rate: get(id, kAudioDevicePropertyNominalSampleRate, Float64(0)) ?? 0,
           transport: fourcc(get(id, kAudioDevicePropertyTransportType, UInt32(0)) ?? 0))
}

func devices() -> [Device] { deviceIDs().map(device) }

func defaultDevice(input: Bool) -> AudioObjectID {
    get(system, input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice, AudioObjectID(0)) ?? 0
}

func find(_ key: String, input: Bool, in all: [Device] = devices()) -> Device? {
    let side = all.filter { input ? $0.inputs > 0 : $0.outputs > 0 }
    return side.first { $0.uid == key } ?? side.first { $0.name == key }
}

/// A property listener block; `remove()` takes it off (else they pile up: one per call, per device change).
struct Listener {
    let obj: AudioObjectID, a: AudioObjectPropertyAddress, q: DispatchQueue, block: AudioObjectPropertyListenerBlock

    func remove() {
        var a = a
        AudioObjectRemovePropertyListenerBlock(obj, &a, q, block)
    }
}

@discardableResult
func listen(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ q: DispatchQueue,
            _ scope: AudioObjectPropertyScope = global, _ f: @escaping () -> Void) -> Listener {
    var a = addr(sel, scope)
    let block: AudioObjectPropertyListenerBlock = { _, _ in f() }
    AudioObjectAddPropertyListenerBlock(obj, &a, q, block)
    return Listener(obj: obj, a: a, q: q, block: block)
}

/// Our IO cycle on this device (per process). The loop takes 1024 frames (21 ms at 48 kHz, half the wake-ups of the
/// usual 512: 0.0058 against 0.0068 CPU s per s on the runner); record and play keep the device's own, for latency.
func setCycle(_ dev: AudioObjectID, _ frames: UInt32) {
    var a = addr(kAudioDevicePropertyBufferFrameSize), n = frames
    AudioObjectSetPropertyData(dev, &a, 0, nil, UInt32(MemoryLayout<UInt32>.size), &n)
}

/// Exit when stdin closes: the app ended (or died), nothing may keep the microphone open.
func exitOnStdinEOF() {
    Thread {
        var b = [UInt8](repeating: 0, count: 256)
        while read(0, &b, b.count) > 0 {}
        quit(0)
    }.start()
}

// MARK: - audio plumbing

/// A ring of mono samples between a real-time IOProc and a worker thread.
final class Ring {
    let cap: Int
    private let buf: UnsafeMutablePointer<Float>
    private var r = 0, w = 0
    private let lock = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)

    init(_ cap: Int) {
        self.cap = cap
        buf = .allocate(capacity: cap)
        lock.initialize(to: os_unfair_lock())
    }

    var count: Int {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        return w - r
    }

    @discardableResult func write(_ p: UnsafePointer<Float>, _ n: Int) -> Int {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        let k = min(n, cap - (w - r))
        for i in 0..<k { buf[(w + i) % cap] = p[i] }
        w += k
        return k
    }

    @discardableResult func read(_ p: UnsafeMutablePointer<Float>, _ n: Int) -> Int {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        let k = min(n, w - r)
        for i in 0..<k { p[i] = buf[(r + i) % cap] }
        r += k
        return k
    }

    /// Keep only the newest `n` (clock drift between two devices).
    func trim(to n: Int) {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        if w - r > n { r = w - n }
    }
}

/// Streaming sample-rate conversion of mono float32 (AVAudioConverter keeps its filter state between calls).
final class Resample {
    let from: Double, to: Double
    private let conv: AVAudioConverter?
    private let inFmt: AVAudioFormat, outFmt: AVAudioFormat

    init(_ from: Double, _ to: Double) {
        self.from = from
        self.to = to
        inFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: from, channels: 1, interleaved: false)!
        outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: to, channels: 1, interleaved: false)!
        conv = from == to ? nil : AVAudioConverter(from: inFmt, to: outFmt)
    }

    func run(_ p: UnsafePointer<Float>, _ n: Int, _ emit: (UnsafePointer<Float>, Int) -> Void) {
        guard n > 0 else { return }
        guard let conv else { return emit(p, n) }
        let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(n))!
        inBuf.frameLength = AVAudioFrameCount(n)
        inBuf.floatChannelData![0].update(from: p, count: n)
        let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: AVAudioFrameCount(Double(n) * to / from) + 256)!
        var given = false
        while true {
            var err: NSError?
            outBuf.frameLength = 0
            let st = conv.convert(to: outBuf, error: &err) { _, status in
                if given {
                    status.pointee = .noDataNow
                    return nil
                }
                given = true
                status.pointee = .haveData
                return inBuf
            }
            if st == .error { die(5, "resampling: \(err?.localizedDescription ?? "?")") }
            if outBuf.frameLength > 0 { emit(outBuf.floatChannelData![0], Int(outBuf.frameLength)) }
            if st != .haveData { return }
        }
    }
}

let maxFrames = 16384

/// The input of one device into a ring (mono); every IO cycle wakes `wake`.
final class Capture {
    let dev: Device
    let ring: Ring
    private var proc: AudioDeviceIOProcID?

    init?(_ dev: Device, wake: DispatchSemaphore, cycle: UInt32 = 0) {
        self.dev = dev
        ring = Ring(Int(max(dev.rate, 8000) * 2))
        let mono = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
        let ring = ring
        if cycle > 0 { setCycle(dev.id, cycle) }
        let st = AudioDeviceCreateIOProcIDWithBlock(&proc, dev.id, nil) { _, input, _, output, _ in
            if let b = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input)).first,
               let data = b.mData, b.mNumberChannels > 0 {
                let ch = Int(b.mNumberChannels), frames = min(Int(b.mDataByteSize) / 4 / ch, maxFrames)
                let x = data.assumingMemoryBound(to: Float.self)
                // mono: a stereo microphone's two sides mixed; of more channels the first (an audio interface's
                // microphone input; the mean of 8 would put it 18 dB down)
                for i in 0..<frames { mono[i] = ch == 2 ? (x[2 * i] + x[2 * i + 1]) / 2 : x[i * ch] }
                ring.write(mono, frames)
                wake.signal()
            }
            for b in UnsafeMutableAudioBufferListPointer(output) {  // a device with outputs too (BlackHole): add nothing
                if let d = b.mData { memset(d, 0, Int(b.mDataByteSize)) }
            }
        }
        guard st == noErr, let proc else {
            say("cannot open \(dev.name) (\(st))")
            return nil
        }
        started(dev.id, proc)  // before it runs: a quit() meanwhile stops it too
        let go = AudioDeviceStart(dev.id, proc)
        guard go == noErr else {
            stopIO(dev.id, proc)  // and destroys its IOProcID
            say("cannot start \(dev.name) (\(go))")
            return nil
        }
    }

    func stop() {
        guard let proc else { return }
        stopIO(dev.id, proc)
        self.proc = nil
    }
}

final class Flag {
    var on = false
}

/// A ring (mono) to every channel of a device's output. Silence until `prime` samples are there (again after it ran
/// dry), so the output does not crackle at the input's pace; `drain.on`: whatever is left plays. Every IO cycle signals
/// `pulled`.
final class Render {
    let dev: Device
    let drain = Flag()
    private var proc: AudioDeviceIOProcID?

    init?(_ dev: Device, from ring: Ring, prime: Int = 0, pulled: DispatchSemaphore? = nil, cycle: UInt32 = 0) {
        self.dev = dev
        let mono = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
        let drain = drain
        var primed = prime == 0
        if cycle > 0 { setCycle(dev.id, cycle) }
        let st = AudioDeviceCreateIOProcIDWithBlock(&proc, dev.id, nil) { _, _, _, output, _ in
            let bufs = UnsafeMutableAudioBufferListPointer(output)
            let frames = min(bufs.map { Int($0.mDataByteSize) / 4 / max(Int($0.mNumberChannels), 1) }.max() ?? 0, maxFrames)
            if !primed && (ring.count >= prime || drain.on) { primed = true }
            let got = primed ? ring.read(mono, frames) : 0
            if got < frames {
                primed = prime == 0
                for i in got..<frames { mono[i] = 0 }
            }
            for b in bufs {
                guard let d = b.mData else { continue }
                let ch = Int(b.mNumberChannels), y = d.assumingMemoryBound(to: Float.self)
                for i in 0..<min(Int(b.mDataByteSize) / 4 / max(ch, 1), frames) {
                    for c in 0..<ch { y[i * ch + c] = mono[i] }
                }
            }
            pulled?.signal()
        }
        guard st == noErr, let proc else {
            say("cannot open \(dev.name) (\(st))")
            return nil
        }
        started(dev.id, proc)  // before it runs: a quit() meanwhile stops it too
        let go = AudioDeviceStart(dev.id, proc)
        guard go == noErr else {
            stopIO(dev.id, proc)  // and destroys its IOProcID
            say("cannot start \(dev.name) (\(go))")
            return nil
        }
    }

    func stop() {
        guard let proc else { return }
        stopIO(dev.id, proc)
        self.proc = nil
    }
}

/// The microphone: `saved` while it is there, else (follow) the default input or the first real one; follows device
/// changes on its own queue and says so on stderr. `current` is read by the worker thread.
final class Input {
    let saved: String, follow: Bool, exclude: String, cycle: UInt32
    let wake = DispatchSemaphore(value: 0)
    private let q = DispatchQueue(label: "devices")
    private let lock = NSLock()
    private var cap: Capture?
    private var said = ""
    private var listeners: [Listener] = [], rateListener: Listener?
    private(set) var stopped = false

    init(saved: String, follow: Bool, exclude: String = "", cycle: UInt32 = 0) {
        self.saved = saved
        self.follow = follow || saved.isEmpty
        self.exclude = exclude
        self.cycle = cycle
        listeners = [listen(system, kAudioHardwarePropertyDevices, q) { [weak self] in self?.update() },
                     listen(system, kAudioHardwarePropertyDefaultInputDevice, q) { [weak self] in self?.update() }]
        q.sync { update() }
    }

    var current: Capture? {
        lock.lock(); defer { lock.unlock() }
        return cap
    }

    func stop() {
        q.sync {
            cap?.stop()
            setCap(nil)
            stopped = true
            for l in listeners { l.remove() }
            rateListener?.remove()
            listeners = []
            rateListener = nil
        }
        wake.signal()
    }

    func resolve(_ all: [Device]) -> Device? {
        let ins = all.filter { $0.inputs > 0 && $0.uid != exclude }
        if !saved.isEmpty, let d = find(saved, input: true, in: ins) { return d }
        guard follow else { return nil }
        let def = defaultDevice(input: true)
        return ins.first { $0.id == def && $0.real } ?? ins.first { $0.real }
    }

    private func update() {  // on q
        if stopped { return }
        let all = devices()
        let want = resolve(all)
        if let c = cap, c.dev.id == want?.id, c.dev.rate == want?.rate { return }
        if let c = cap {
            c.stop()
            setCap(nil)
            if !all.contains(where: { $0.id == c.dev.id }) {
                say("lost \(c.dev.label)")
                if !follow { die(4, "\(c.dev.name) went away") }
            }
        }
        guard let d = want else {
            if !follow { die(3, "no input device \(saved)") }
            event("none")
            return
        }
        guard let c = Capture(d, wake: wake, cycle: cycle) else {
            event("none")
            return
        }
        rateListener?.remove()
        rateListener = listen(d.id, kAudioDevicePropertyNominalSampleRate, q) { [weak self] in self?.update() }
        setCap(c)
        event("using \(d.label)")
        wake.signal()
    }

    private func setCap(_ c: Capture?) {
        lock.lock()
        cap = c
        lock.unlock()
    }

    private func event(_ s: String) {
        if s != said { say(s) }
        said = s
    }
}

/// macOS 14+: whether a process other than this one records from `dev` now (a call app with BlackHole 2ch as its
/// microphone), from Core Audio's process objects and their notifications; `changed` on every flip, on its queue.
/// `supported` is false when the system lists no process objects (then the caller keeps the loop on).
@available(macOS 14.0, *)
final class Readers {
    private let dev: AudioObjectID, me = getpid(), changed: (Bool) -> Void
    private let q = DispatchQueue(label: "readers")
    private var watched: [AudioObjectID: [Listener]] = [:], reading = false
    let supported: Bool

    init(_ dev: AudioObjectID, _ changed: @escaping (Bool) -> Void) {
        self.dev = dev
        self.changed = changed
        supported = !objects(system, kAudioHardwarePropertyProcessObjectList).isEmpty
        guard supported else { return }
        listen(system, kAudioHardwarePropertyProcessObjectList, q) { [weak self] in self?.update() }
        q.sync { update() }  // the first look before the caller says it is ready
    }

    private func update() {  // on q
        var any = false
        let now = objects(system, kAudioHardwarePropertyProcessObjectList)
        for (p, ls) in watched where !now.contains(p) {  // that process is gone
            for l in ls { l.remove() }
            watched[p] = nil
        }
        for p in now {
            if watched[p] == nil {
                watched[p] = [listen(p, kAudioProcessPropertyIsRunningInput, q) { [weak self] in self?.update() },
                              listen(p, kAudioProcessPropertyDevices, q, kAudioObjectPropertyScopeInput) { [weak self] in self?.update() }]
            }
            if get(p, kAudioProcessPropertyPID, pid_t(0)) == me { continue }
            if get(p, kAudioProcessPropertyIsRunningInput, UInt32(0)) == 1,
               objects(p, kAudioProcessPropertyDevices, kAudioObjectPropertyScopeInput).contains(dev) {
                any = true
            }
        }
        if any != reading {
            reading = any
            changed(any)
        }
    }
}

/// The worker: whatever the current capture gives, converted to `rate`, to `emit`.
func pump(_ input: Input, rate: Double, _ emit: @escaping (UnsafePointer<Float>, Int) -> Void) {
    Thread {
        let tmp = UnsafeMutablePointer<Float>.allocate(capacity: 1 << 18)
        var src: Capture?, rs: Resample?
        while true {
            input.wake.wait()
            if input.stopped { return }
            guard let c = input.current else { continue }
            if c !== src {
                src = c
                rs = Resample(c.dev.rate, rate)
            }
            let n = c.ring.read(tmp, 1 << 18)
            rs!.run(tmp, n, emit)
        }
    }.start()
}

// MARK: - commands

var args = Array(CommandLine.arguments.dropFirst())
let command = args.isEmpty ? "" : args.removeFirst()

func opt(_ name: String, _ def: String? = nil) -> String? {
    guard let i = args.firstIndex(of: name) else { return def }
    return i + 1 < args.count ? args[i + 1] : def
}

func flag(_ name: String) -> Bool { args.contains(name) }

func rate(_ def: Double) -> Double {
    guard let s = opt("--rate") else { return def }
    guard let r = Double(s), r >= 8000, r <= 192_000 else { die(2, "bad --rate \(s)") }
    return r
}

func permission() -> String {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: return "authorized"
    case .denied: return "denied"
    case .restricted: return "restricted"
    case .notDetermined: return "undetermined"
    @unknown default: return "unknown"
    }
}

/// Before reading a microphone: ask once (the system dialog names the app that started us), refuse when denied:
/// macOS would hand us silence instead of an error.
func needMicrophone() {
    switch permission() {
    case "denied", "restricted":
        die(6, "no permission to use the microphone: System Settings → Privacy & Security → Microphone → allow Peresmeshnik")
    case "undetermined":
        let s = DispatchSemaphore(value: 0)
        var ok = true
        AVCaptureDevice.requestAccess(for: .audio) { ok = $0; s.signal() }
        if s.wait(timeout: .now() + 120) == .success, !ok {
            die(6, "no permission to use the microphone: System Settings → Privacy & Security → Microphone → allow Peresmeshnik")
        }
    default:
        return
    }
}

func output(_ key: String?) -> Device {
    guard let key, !key.isEmpty else { die(2, "--to DEV is needed") }
    guard let d = find(key, input: false) else { die(3, "no output device \(key)") }
    return d
}

/// Exit 4 when `dev` is removed.
func watchGone(_ dev: Device) {
    let q = DispatchQueue(label: "gone")
    listen(system, kAudioHardwarePropertyDevices, q) {
        if !deviceIDs().contains(dev.id) { die(4, "\(dev.name) went away") }
    }
}

switch command {
case "list":
    let all = devices(), defIn = defaultDevice(input: true), defOut = defaultDevice(input: false)
    let rows: [[String: Any]] = all.map {
        ["id": Int($0.id), "uid": $0.uid, "name": $0.name, "inputs": $0.inputs, "outputs": $0.outputs, "rate": $0.rate,
         "transport": $0.transport, "default_input": $0.id == defIn, "default_output": $0.id == defOut]
    }
    let data = try! JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    print()

case "permission":
    print(permission())

case "record":
    let to = rate(16000)
    needMicrophone()
    let input = Input(saved: opt("--from", "") ?? "", follow: flag("--follow"))
    exitOnStdinEOF()
    pump(input, rate: to) { p, n in
        if fwrite(p, 4, n, stdout) < n { quit(0) }  // the reader is gone
        fflush(stdout)
    }
    dispatchMain()

case "loop":
    var target = output(opt("--to"))
    needMicrophone()
    watchGone(target)
    let from = opt("--from", "") ?? ""
    var running: (Render, Input)?
    let runLock = NSLock()  // run() from the readers' queue, a rate change from its own
    func start() {
        say("call")
        target = device(target.id)  // its rate now: a call app or Audio MIDI Setup may have changed it meanwhile
        let ring = Ring(Int(target.rate / 2))
        let keep = Int(target.rate * 0.06)  // at most 60 ms queued: the mic and the target run on two clocks
        guard let render = Render(target, from: ring, prime: Int(target.rate * 0.02), cycle: 1024) else { return }
        let input = Input(saved: from, follow: true, exclude: target.uid, cycle: 1024)
        pump(input, rate: target.rate) { p, n in
            ring.write(p, n)
            if ring.count > keep { ring.trim(to: keep / 2) }
        }
        running = (render, input)
    }
    func end() {
        guard let r = running else { return }
        r.1.stop()  // the microphone, then the output
        r.0.stop()
        running = nil
        say("idle")
    }
    func run(_ on: Bool) {
        runLock.lock()
        defer { runLock.unlock() }
        if on, running == nil {
            start()
        } else if !on {
            end()
        }
    }
    listen(target.id, kAudioDevicePropertyNominalSampleRate, DispatchQueue(label: "rate")) {
        runLock.lock()
        defer { runLock.unlock() }
        if let r = running, device(target.id).rate != r.0.dev.rate {  // at the new rate, else a wrong speed and pitch
            end()
            start()
        }
    }
    var readers: AnyObject?
    if #available(macOS 14.0, *), !flag("--always") {
        let r = Readers(target.id) { run($0) }  // run() on its queue, the first time before this returns
        if !r.supported {
            run(true)
        } else if running == nil {
            say("idle")
        }
        readers = r
    } else {
        run(true)
    }
    exitOnStdinEOF()
    withExtendedLifetime(readers) { dispatchMain() }

case "play":
    let target = output(opt("--to"))
    let from = rate(24000)
    guard let ms = Double(opt("--latency", "0")!), ms >= 0, ms < 1000 else { die(2, "bad --latency") }
    watchGone(target)
    listen(target.id, kAudioDevicePropertyNominalSampleRate, DispatchQueue(label: "rate")) {
        if device(target.id).rate != target.rate { die(7, "the rate of \(target.name) changed") }  // the app starts us again
    }
    if device(target.id).rate != target.rate { die(7, "the rate of \(target.name) changed") }  // before the listener was there
    let ring = Ring(Int(target.rate / 4))  // a full ring blocks stdin: the writer waits as on a pipe
    let pulled = DispatchSemaphore(value: 0)
    func nextCycle() {  // signalled on every IO cycle, waited on only now and then: the next one, not a backlog
        while pulled.wait(timeout: .now()) == .success {}
        pulled.wait()
    }
    guard let render = Render(target, from: ring, prime: Int(target.rate * ms / 1000), pulled: pulled) else {
        die(5, "cannot play into \(target.name)")
    }
    let rs = Resample(from, target.rate)
    var bytes = [UInt8](repeating: 0, count: 16384)
    var have = 0
    while true {
        let n = bytes.withUnsafeMutableBytes { read(0, $0.baseAddress! + have, $0.count - have) }
        if n <= 0 { break }
        have += n
        let whole = have / 4 * 4
        bytes.withUnsafeBytes { raw in
            rs.run(raw.baseAddress!.assumingMemoryBound(to: Float.self), whole / 4) { p, k in
                var done = 0
                while done < k {
                    done += ring.write(p + done, k - done)
                    if done < k { nextCycle() }
                }
            }
        }
        if whole < have { bytes.replaceSubrange(0..<(have - whole), with: bytes[whole..<have]) }
        have -= whole
    }
    render.drain.on = true
    while ring.count > 0 { nextCycle() }
    usleep(60_000)  // the last IO buffer
    quit(0)

case "aggregate":
    guard let uid = opt("--uid"), let of = opt("--of"), let sub = find(of, input: true) ?? find(of, input: false) else {
        die(2, "aggregate --uid UID --of DEV")
    }
    let desc: [String: Any] = [
        kAudioAggregateDeviceUIDKey: uid, kAudioAggregateDeviceNameKey: opt("--name", uid)!,
        kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: sub.uid]],
        kAudioAggregateDeviceMainSubDeviceKey: sub.uid, kAudioAggregateDeviceIsPrivateKey: 0,
    ]
    var id = AudioObjectID(0)
    let st = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &id)
    guard st == noErr else { die(5, "cannot create the aggregate device (\(st))") }
    ioLock.lock()
    aggregates.append(id)  // quit() removes it, on SIGTERM too
    ioLock.unlock()
    print("created \(id)")
    fflush(stdout)
    var b = [UInt8](repeating: 0, count: 64)
    while read(0, &b, b.count) > 0 {}
    quit(0)

default:
    die(2, "usage: peresmeshnik-audio list | permission | record [--from DEV] [--follow] [--rate R] | "
        + "play --to DEV [--rate R] | loop [--from DEV] --to DEV | aggregate --uid UID --of DEV")
}
