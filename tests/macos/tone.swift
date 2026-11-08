// A stand-in call app for the tap tests: plays a sine at argv[1] Hz on the default output until it is killed.
// The test builds it into an .app (and a helper .app inside), so Core Audio knows it by a bundle id.
import AVFoundation

let hz = Double(CommandLine.arguments.dropFirst().first ?? "440") ?? 440
let engine = AVAudioEngine()
let rate = engine.outputNode.outputFormat(forBus: 0).sampleRate
var phase = 0.0
let source = AVAudioSourceNode { _, _, frames, list -> OSStatus in
    for f in 0..<Int(frames) {
        let v = Float(0.3 * sin(phase))
        phase += 2 * Double.pi * hz / rate
        for b in UnsafeMutableAudioBufferListPointer(list) { b.mData!.assumingMemoryBound(to: Float.self)[f] = v }
    }
    return noErr
}
engine.attach(source)
engine.connect(source, to: engine.mainMixerNode, format: AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1))
do { try engine.start() } catch { FileHandle.standardError.write(Data("tone: \(error)\n".utf8)); exit(1) }
dispatchMain()
