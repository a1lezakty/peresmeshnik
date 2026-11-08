// Headphones that come and go, for the tests of Core Audio's notifications: a public aggregate device named argv[1]
// over argv[2] (default "BlackHole 2ch", so what plays there can be recorded), made at start and destroyed when stdin
// closes. Prints "made <id>" once it exists. Exit 2: no such sub-device, 6: Core Audio refused.
import CoreAudio
import Foundation

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: 0)
}

func string(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String {
    var a = address(selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = withUnsafeMutablePointer(to: &value) { AudioObjectGetPropertyData(object, &a, 0, nil, &size, $0) }
    guard status == noErr, let value else { return "" }
    return value.takeRetainedValue() as String
}

func uid(of name: String) -> String? {
    var a = address(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    let system = AudioObjectID(kAudioObjectSystemObject)
    guard AudioObjectGetPropertyDataSize(system, &a, 0, nil, &size) == noErr else { return nil }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(system, &a, 0, nil, &size, &ids) == noErr else { return nil }
    return ids.first { string($0, kAudioObjectPropertyName) == name }.map { string($0, kAudioDevicePropertyDeviceUID) }
}

let args = CommandLine.arguments
guard args.count > 1, let sub = uid(of: args.count > 2 ? args[2] : "BlackHole 2ch") else {
    FileHandle.standardError.write(Data("fakedevice: usage: fakedevice NAME [SUB-DEVICE]; the sub-device must exist\n".utf8))
    exit(2)
}
let spec: [String: Any] = [
    kAudioAggregateDeviceNameKey: args[1],
    kAudioAggregateDeviceUIDKey: "peresmeshnik-test-" + UUID().uuidString,
    kAudioAggregateDeviceIsPrivateKey: false,  // the test process must see it
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: sub]],
    kAudioAggregateDeviceMainSubDeviceKey: sub,
]
var device = AudioObjectID(kAudioObjectUnknown)
let status = AudioHardwareCreateAggregateDevice(spec as CFDictionary, &device)
if status != noErr {
    FileHandle.standardError.write(Data("fakedevice: cannot make it: \(status)\n".utf8))
    exit(6)
}
print("made \(device)")
fflush(stdout)
var buffer = [UInt8](repeating: 0, count: 64)
while read(0, &buffer, buffer.count) > 0 {}
AudioHardwareDestroyAggregateDevice(device)
print("gone")
