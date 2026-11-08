// peresmeshnik-tap: the call app's sound through a Core Audio process tap, as raw PCM on stdout (macOS 14.4+).
//
//   peresmeshnik-tap check     one line "macos=15.3.1 tap=yes permission=granted|denied|unknown"; exit 0, 4 denied, 5 too old
//   peresmeshnik-tap request   ask for "System Audio Recording" (the system dialog); exit 0 granted, 4 denied
//   peresmeshnik-tap list      the processes Core Audio knows, one JSON object per line: pid, bundle, name, playing
//   peresmeshnik-tap tap (--bundle ID | --pid N) [--no-ask]
//
// tap: the app's processes (--bundle: that bundle id, or one that starts with it and a dot: its helpers, e.g.
// com.google.Chrome.helper) go into one mono tap that mutes them while it runs (mutedWhenTapped): only we hear them.
// stdout: one line "peresmeshnik-tap 1 rate=<Hz> channels=1 format=f32le", then mono float32 little-endian at that
// rate. It lives while stdin is open. An app that has not used sound yet is waited for (the header comes once the tap
// runs); the app's processes that start later (a helper, the next call) join the tap. Nothing polls: Core Audio's
// notifications, a process-exit source and a blocked read of stdin. Permission not asked yet: the system dialog first
// (--no-ask: start anyway, the system asks by itself when the tap starts).
// Exit: 0 stdin or stdout closed, 2 usage, 3 no such process (--pid: not running, or it ended), 4 no permission,
// 5 macOS older than 14.4, 6 a Core Audio error. Messages go to stderr.
//
// Build (no Xcode project): swiftc -O -target arm64-apple-macos12 native/macos/tap/*.swift -o <dir>/peresmeshnik-tap

import AppKit
import CoreAudio
import Foundation

enum Code: Int32 { case ok = 0, usage = 2, noProcess = 3, noPermission = 4, oldMacOS = 5, coreAudio = 6 }

func say(_ text: String) { FileHandle.standardError.write(Data("peresmeshnik-tap: \(text)\n".utf8)) }

func quit(_ code: Code, _ text: String? = nil) -> Never {
    if let text { say(text) }
    exit(code.rawValue)
}

func fourcc(_ status: OSStatus) -> String {
    let n = UInt32(bitPattern: status)
    let bytes = [24, 16, 8, 0].map { UInt8((n >> UInt32($0)) & 0xff) }
    return bytes.allSatisfy({ $0 >= 32 && $0 < 127 }) ? "'\(String(decoding: bytes, as: UTF8.self))' (\(status))" : "\(status)"
}

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: 0)  // 0: the main element
}

let system = AudioObjectID(kAudioObjectSystemObject)

func read<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ initial: T) -> T? {
    var a = address(selector)
    var value = initial
    var size = UInt32(MemoryLayout<T>.size)
    return AudioObjectGetPropertyData(object, &a, 0, nil, &size, &value) == noErr ? value : nil
}

func readString(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String {
    var a = address(selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = withUnsafeMutablePointer(to: &value) { AudioObjectGetPropertyData(object, &a, 0, nil, &size, $0) }
    guard status == noErr, let value else { return "" }
    return value.takeRetainedValue() as String
}

func readIDs(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> [AudioObjectID] {
    var a = address(selector)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(object, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(object, &a, 0, nil, &size, &ids) == noErr else { return [] }
    return Array(ids.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
}

// "System Audio Recording" has no public API to read or ask: TCC's own calls, as Apple's sample-like apps do (AudioCap)
let tcc = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW)
let service = "kTCCServiceAudioCapture" as CFString

func permission() -> String {
    guard let tcc, let f = dlsym(tcc, "TCCAccessPreflight") else { return "unknown" }
    typealias Preflight = @convention(c) (CFString, CFDictionary?) -> Int32
    switch unsafeBitCast(f, to: Preflight.self)(service, nil) {
    case 0: return "granted"
    case 1: return "denied"
    default: return "unknown"  // not asked yet
    }
}

func ask(_ then: @escaping (Bool) -> Void) {
    guard let tcc, let f = dlsym(tcc, "TCCAccessRequest") else { return then(true) }  // no such call: the tap asks by itself
    typealias Request = @convention(c) (CFString, CFDictionary?, @escaping @convention(block) (Bool) -> Void) -> Void
    unsafeBitCast(f, to: Request.self)(service, nil) { granted in then(granted) }
}

let version = ProcessInfo.processInfo.operatingSystemVersion
let versionText = "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
let tapSupported = ProcessInfo.processInfo.isOperatingSystemAtLeast(OperatingSystemVersion(majorVersion: 14, minorVersion: 4, patchVersion: 0))
let deniedText = "no permission to record the sound of other apps (System Settings > Privacy & Security > "
    + "Screen & System Audio Recording > System Audio Recording Only)"

@available(macOS 14.2, *)
func listCommand() -> Never {
    for object in readIDs(system, kAudioHardwarePropertyProcessObjectList) {
        let pid = read(object, kAudioProcessPropertyPID, pid_t(-1)) ?? -1
        let line: [String: Any] = [
            "pid": Int(pid),
            "bundle": readString(object, kAudioProcessPropertyBundleID),
            "name": NSRunningApplication(processIdentifier: pid)?.localizedName ?? "",
            "playing": (read(object, kAudioProcessPropertyIsRunningOutput, UInt32(0)) ?? 0) != 0,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) {
            print(String(decoding: data, as: UTF8.self))
        }
    }
    exit(0)
}

@available(macOS 14.2, *)
final class Tapper {
    let bundle: String
    let pid: pid_t?
    let queue = DispatchQueue(label: "peresmeshnik-tap")  // every change of the tap, in order
    // Core Audio's notifications come on a queue of their own and only hand on: a listener on `queue` would wait for
    // the tap being made there (making the tap changes the process list), and making it waits for the listener
    let notes = DispatchQueue(label: "peresmeshnik-tap.notes")
    let io = DispatchQueue(label: "peresmeshnik-tap.io", qos: .userInteractive)
    let uuid = UUID()
    var tap = AudioObjectID(kAudioObjectUnknown)
    var device = AudioObjectID(kAudioObjectUnknown)
    var proc: AudioDeviceIOProcID?
    var tapped: [AudioObjectID] = []
    var exitSource: DispatchSourceProcess?
    var mono = [Float](repeating: 0, count: 8192)
    var writing = true  // the IO queue only

    init(bundle: String, pid: pid_t?) {
        self.bundle = bundle
        self.pid = pid
    }

    func matching() -> [AudioObjectID] {
        readIDs(system, kAudioHardwarePropertyProcessObjectList).filter { object in
            if let pid { return read(object, kAudioProcessPropertyPID, pid_t(-1)) == pid }
            let b = readString(object, kAudioProcessPropertyBundleID)
            return !b.isEmpty && (b == bundle || b.hasPrefix(bundle + "."))
        }
    }

    func pids(_ objects: [AudioObjectID]) -> String {
        objects.map { String(read($0, kAudioProcessPropertyPID, pid_t(-1)) ?? -1) }.joined(separator: " ")
    }

    func description(_ processes: [AudioObjectID]) -> CATapDescription {
        let d = CATapDescription(monoMixdownOfProcesses: processes)
        d.uuid = uuid
        d.name = "Peresmeshnik call tap"
        d.isPrivate = true
        d.muteBehavior = .mutedWhenTapped
        return d
    }

    func start() {  // on the queue
        var a = address(kAudioHardwarePropertyProcessObjectList)
        let status = AudioObjectAddPropertyListenerBlock(system, &a, notes) { [self] _, _ in queue.async { self.changed() } }
        if status != noErr { stop(.coreAudio, "listening to the process list: \(fourcc(status))") }
        if let pid {
            let source = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: queue)
            source.setEventHandler { [self] in stop(.noProcess, "process \(pid) ended") }
            source.resume()
            exitSource = source
        }
        let target = pid.map { "process \($0)" } ?? bundle
        if matching().isEmpty { say("waiting for \(target) to use sound") }
        changed()
    }

    func changed() {
        let now = matching()
        guard !now.isEmpty, Set(now) != Set(tapped) else { return }
        if tap == kAudioObjectUnknown {
            run(now)
        } else {  // a new process of the app (or one gone): the same tap, the new list
            var a = address(kAudioTapPropertyDescription)
            var d = description(now)
            let status = AudioObjectSetPropertyData(tap, &a, 0, nil, UInt32(MemoryLayout<CATapDescription>.size), &d)
            if status != noErr {
                say("could not change the tapped processes: \(fourcc(status))")
                return
            }
            say("tapping \(pids(now))")
        }
        tapped = now
    }

    func run(_ processes: [AudioObjectID]) {
        func must(_ status: OSStatus, _ what: String) {
            if status != noErr { stop(.coreAudio, "\(what): \(fourcc(status))") }
        }
        say("making a tap of \(pids(processes))")
        must(AudioHardwareCreateProcessTap(description(processes), &tap), "creating the tap")
        let format = read(tap, kAudioTapPropertyFormat, AudioStreamBasicDescription())
        let spec: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Peresmeshnik call tap",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: uuid.uuidString, kAudioSubTapDriftCompensationKey: true]],
        ]
        must(AudioHardwareCreateAggregateDevice(spec as CFDictionary, &device), "creating the tap's device")
        say("tap \(tap), device \(device)")
        var rate = read(device, kAudioDevicePropertyNominalSampleRate, Float64(0)) ?? 0
        if rate <= 0 { rate = format?.mSampleRate ?? 0 }
        if rate <= 0 { stop(.coreAudio, "the tap has no sample rate") }
        must(AudioDeviceCreateIOProcIDWithBlock(&proc, device, io) { [self] _, input, _, _, _ in pump(input) }, "the tap's IO")
        let header = "peresmeshnik-tap 1 rate=\(Int(rate)) channels=1 format=f32le\n"
        header.withCString { _ = Darwin.write(1, $0, strlen($0)) }
        say("starting")
        must(AudioDeviceStart(device, proc), "starting the tap")
        let given = format.map { "\($0.mSampleRate) Hz, \($0.mChannelsPerFrame) ch" } ?? "?"
        say("tapping \(pids(processes)) at \(Int(rate)) Hz (tap format: \(given))")
    }

    func pump(_ input: UnsafePointer<AudioBufferList>) {  // the IO queue: mix down to mono, write; only blocks on the pipe
        guard writing else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        guard let first = buffers.first, first.mNumberChannels > 0 else { return }
        let frames = Int(first.mDataByteSize) / (4 * Int(first.mNumberChannels))
        if mono.count < frames { mono = [Float](repeating: 0, count: frames) }
        var channels = 0
        let ok = mono.withUnsafeMutableBufferPointer { out -> Bool in
            for i in 0..<frames { out[i] = 0 }
            for b in buffers {
                guard let data = b.mData, b.mNumberChannels > 0 else { continue }
                let n = Int(b.mNumberChannels)
                let p = data.assumingMemoryBound(to: Float.self)
                for f in 0..<min(frames, Int(b.mDataByteSize) / (4 * n)) {
                    for c in 0..<n { out[f] += p[f * n + c] }
                }
                channels += n
            }
            if channels > 1 {
                let k = 1 / Float(channels)
                for i in 0..<frames { out[i] *= k }
            }
            var at = UnsafeRawPointer(out.baseAddress!)
            var left = frames * 4
            while left > 0 {
                let n = Darwin.write(1, at, left)
                if n < 0 && errno == EINTR { continue }
                if n <= 0 { return false }
                left -= n
                at = at.advanced(by: n)
            }
            return true
        }
        if !ok {
            writing = false
            queue.async { [self] in stop(.ok, "stdout closed") }
        }
    }

    func stop(_ code: Code, _ text: String) -> Never {
        if let proc {
            AudioDeviceStop(device, proc)
            AudioDeviceDestroyIOProcID(device, proc)
        }
        if device != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(device) }
        if tap != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tap) }
        quit(code, text)
    }
}

@available(macOS 14.2, *)
func tapCommand(_ args: [String]) -> Never {
    func value(_ name: String) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
    let bundle = value("--bundle") ?? ""
    let pid = value("--pid").flatMap { pid_t($0) }
    if bundle.isEmpty == (pid == nil) { quit(.usage, "tap: give --bundle ID or --pid N") }
    if let pid, kill(pid, 0) != 0 && errno == ESRCH { quit(.noProcess, "no process \(pid)") }
    let tapper = Tapper(bundle: bundle, pid: pid)
    Thread {  // the parent closes stdin (or dies): the tap goes, the app is heard as usual again
        var b = [UInt8](repeating: 0, count: 256)
        while Darwin.read(0, &b, b.count) > 0 {}
        tapper.queue.async { tapper.stop(.ok, "stdin closed") }
    }.start()
    switch permission() {
    case "denied":
        quit(.noPermission, deniedText)
    case "unknown" where !args.contains("--no-ask"):
        say("asking for permission to record the sound of other apps")
        ask { granted in
            if !granted { quit(.noPermission, deniedText) }
            tapper.queue.async { tapper.start() }
        }
    default:
        tapper.queue.async { tapper.start() }
    }
    dispatchMain()
}

signal(SIGPIPE, SIG_IGN)
do {  // Core Audio's notifications on its own thread: the main thread parks in dispatchMain(), no run loop runs there
    var loop: CFRunLoop? = nil
    var a = address(kAudioHardwarePropertyRunLoop)
    AudioObjectSetPropertyData(system, &a, 0, nil, UInt32(MemoryLayout<CFRunLoop?>.size), &loop)
}
let args = Array(CommandLine.arguments.dropFirst())
let old = "macOS \(versionText) is older than 14.4: no process taps (set the call app's speaker to BlackHole 16ch)"
switch args.first ?? "" {
case "check":
    let p = permission()
    print("macos=\(versionText) tap=\(tapSupported ? "yes" : "no") permission=\(p)")
    fflush(stdout)
    exit(!tapSupported ? Code.oldMacOS.rawValue : p == "denied" ? Code.noPermission.rawValue : 0)
case "request":
    if !tapSupported { quit(.oldMacOS, old) }
    if permission() == "granted" { exit(0) }
    ask { granted in exit(granted ? 0 : Code.noPermission.rawValue) }
    dispatchMain()
case "list", "tap":
    if !tapSupported { quit(.oldMacOS, old) }
    if #available(macOS 14.2, *) {
        if args[0] == "list" { listCommand() }
        tapCommand(Array(args.dropFirst()))
    }
    quit(.oldMacOS, old)
default:
    quit(.usage, "usage: peresmeshnik-tap check | request | list | tap (--bundle ID | --pid N) [--no-ask]")
}
