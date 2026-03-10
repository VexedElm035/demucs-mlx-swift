import Foundation

public final class DemucsSeparator {
    public let modelName: String
    public let descriptor: DemucsModelDescriptor

    public var parameters: DemucsSeparationParameters

    private let model: StemSeparationModel

    public init(
        modelName: String = "htdemucs",
        parameters: DemucsSeparationParameters = DemucsSeparationParameters(),
        modelDirectory: URL? = nil
    ) throws {
        self.modelName = modelName
        self.descriptor = try DemucsModelRegistry.descriptor(for: modelName)
        self.parameters = try parameters.validated()
        self.model = try DemucsModelFactory.makeModel(for: descriptor, modelDirectory: modelDirectory)
    }

    public var sampleRate: Int {
        descriptor.sampleRate
    }

    public var audioChannels: Int {
        descriptor.audioChannels
    }

    public var sources: [String] {
        descriptor.sourceNames
    }

    public func updateParameters(_ parameters: DemucsSeparationParameters) throws {
        self.parameters = try parameters.validated()
    }

    public func separate(fileAt url: URL) throws -> DemucsSeparationResult {
        let audio = try AudioIO.loadAudio(from: url)
        return try separate(audio: audio)
    }

    public func separate(audio: DemucsAudio) throws -> DemucsSeparationResult {
        let validated = try parameters.validated()

        let input = audio.channelMajorSamples
        let remixed = AudioDSP.remixChannels(
            channelMajor: input,
            inputChannels: audio.channels,
            targetChannels: descriptor.audioChannels,
            frames: audio.frameCount
        )

        let resampled = AudioDSP.resampleChannelMajor(
            remixed,
            channels: descriptor.audioChannels,
            inputSampleRate: audio.sampleRate,
            targetSampleRate: descriptor.sampleRate,
            frames: audio.frameCount
        )

        let normalizedAudio = try DemucsAudio(
            channelMajor: resampled.samples,
            channels: descriptor.audioChannels,
            sampleRate: descriptor.sampleRate
        )

        let engine = SeparationEngine(model: model, parameters: validated)
        let stemsFlat = try engine.separate(
            mix: resampled.samples,
            channels: descriptor.audioChannels,
            frames: resampled.frames,
            sampleRate: descriptor.sampleRate
        )

        var stems: [String: DemucsAudio] = [:]
        for (sourceIndex, sourceName) in descriptor.sourceNames.enumerated() {
            var sourceSamples = [Float](repeating: 0, count: descriptor.audioChannels * resampled.frames)

            for c in 0..<descriptor.audioChannels {
                let sourceBase = (sourceIndex * descriptor.audioChannels + c) * resampled.frames
                let dstBase = c * resampled.frames
                for t in 0..<resampled.frames {
                    sourceSamples[dstBase + t] = stemsFlat[sourceBase + t]
                }
            }

            stems[sourceName] = try DemucsAudio(
                channelMajor: sourceSamples,
                channels: descriptor.audioChannels,
                sampleRate: descriptor.sampleRate
            )
        }

        return DemucsSeparationResult(input: normalizedAudio, stems: stems)
    }
}
