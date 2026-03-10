import AVFoundation
import Foundation

public enum AudioIO {

    public static func loadAudio(from url: URL) throws -> DemucsAudio {
        do {
            let file = try AVAudioFile(forReading: url)
            let sourceFormat = file.processingFormat
            let channels = Int(sourceFormat.channelCount)
            guard channels > 0 else {
                throw DemucsError.audioIO("Input file has zero channels")
            }

            let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sourceFormat.sampleRate,
                channels: sourceFormat.channelCount,
                interleaved: false
            )
            guard let outputFormat else {
                throw DemucsError.audioIO("Failed to create float32 output format")
            }

            let frameCapacity = AVAudioFrameCount(file.length)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: frameCapacity) else {
                throw DemucsError.audioIO("Failed to allocate audio buffer")
            }

            try file.read(into: buffer)
            guard let channelData = buffer.floatChannelData else {
                throw DemucsError.audioIO("No float channel data available")
            }

            let frames = Int(buffer.frameLength)
            var channelMajor = [Float](repeating: 0, count: channels * frames)
            for c in 0..<channels {
                let src = channelData[c]
                let base = c * frames
                for t in 0..<frames {
                    channelMajor[base + t] = src[t]
                }
            }

            return try DemucsAudio(
                channelMajor: channelMajor,
                channels: channels,
                sampleRate: Int(sourceFormat.sampleRate)
            )
        } catch let err as DemucsError {
            throw err
        } catch {
            throw DemucsError.audioIO(error.localizedDescription)
        }
    }

    public static func writeWAV(
        _ audio: DemucsAudio,
        to url: URL,
        bitsPerSample: UInt16 = 16
    ) throws {
        guard bitsPerSample == 16 else {
            throw DemucsError.audioIO("Only 16-bit PCM WAV is currently supported")
        }

        let channels = audio.channels
        let frames = audio.frameCount
        let samples = audio.channelMajorSamples

        let bytesPerSample = Int(bitsPerSample / 8)
        let blockAlign = channels * bytesPerSample
        let byteRate = audio.sampleRate * blockAlign
        let dataSize = frames * blockAlign
        let riffChunkSize = 36 + dataSize

        var data = Data(capacity: 44 + dataSize)

        func appendUInt32(_ value: UInt32) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        func appendUInt16(_ value: UInt16) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        func appendInt16(_ value: Int16) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: "RIFF".utf8)
        appendUInt32(UInt32(riffChunkSize))
        data.append(contentsOf: "WAVE".utf8)

        data.append(contentsOf: "fmt ".utf8)
        appendUInt32(16)
        appendUInt16(1) // PCM
        appendUInt16(UInt16(channels))
        appendUInt32(UInt32(audio.sampleRate))
        appendUInt32(UInt32(byteRate))
        appendUInt16(UInt16(blockAlign))
        appendUInt16(bitsPerSample)

        data.append(contentsOf: "data".utf8)
        appendUInt32(UInt32(dataSize))

        for t in 0..<frames {
            for c in 0..<channels {
                let sample = samples[c * frames + t]
                let clamped = AudioDSP.clampAudio(sample)
                let pcm = Int16(clamped * 32767.0)
                appendInt16(pcm)
            }
        }

        do {
            try data.write(to: url)
        } catch {
            throw DemucsError.audioIO(error.localizedDescription)
        }
    }
}
