import ArgumentParser
import DemucsMLX
import Foundation

public struct DemucsCLI: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "demucs-mlx-swift",
        abstract: "Demucs-style audio stem separation with Swift + MLX",
        discussion: "Separates input audio files into drums, bass, other, and vocals stems."
    )

    @Argument(help: "Input audio files")
    public var tracks: [String] = []

    @Option(name: [.short, .long], help: "Model name")
    public var name: String = "htdemucs"

    @Option(name: [.short, .long], help: "Output directory")
    public var out: String = "separated"

    @Option(name: .customLong("model-dir"), help: "Directory containing model files (.safetensors + _config.json)")
    public var modelDir: String?

    @Option(name: .long, help: "Segment length in seconds")
    public var segment: Double?

    @Option(name: .long, help: "Overlap ratio [0, 1)")
    public var overlap: Float = 0.25

    @Option(name: .long, help: "Number of shift augmentations")
    public var shifts: Int = 1

    @Option(name: .long, help: "Optional random seed for deterministic shifts")
    public var seed: Int?

    @Option(name: [.short, .long], help: "Chunk batch size")
    public var batchSize: Int = 8

    @Flag(name: .customLong("no-split"), help: "Disable chunked overlap-add inference")
    public var noSplit: Bool = false

    @Flag(name: .customLong("list-models"), help: "List available models")
    public var listModels: Bool = false

    public init() {}

    public mutating func run() throws {
        if listModels {
            for model in listAvailableDemucsModels() {
                print(model)
            }
            return
        }

        guard !tracks.isEmpty else {
            throw ValidationError("Please provide at least one input track or use --list-models")
        }

        let params = DemucsSeparationParameters(
            shifts: shifts,
            overlap: overlap,
            split: !noSplit,
            segmentSeconds: segment,
            batchSize: batchSize,
            seed: seed
        )

        let modelDirectoryURL = modelDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let separator = try DemucsSeparator(modelName: name, parameters: params, modelDirectory: modelDirectoryURL)
        let outputRoot = URL(fileURLWithPath: out, isDirectory: true)
        try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)

        for track in tracks {
            let inputURL = URL(fileURLWithPath: track)
            print("Separating: \(inputURL.path)")

            let result = try separator.separate(fileAt: inputURL)
            let trackDir = outputRoot.appendingPathComponent(inputURL.deletingPathExtension().lastPathComponent, isDirectory: true)
            try FileManager.default.createDirectory(at: trackDir, withIntermediateDirectories: true)

            for source in separator.sources {
                guard let stemAudio = result.stems[source] else { continue }
                let stemURL = trackDir.appendingPathComponent("\(source).wav", isDirectory: false)
                try AudioIO.writeWAV(stemAudio, to: stemURL)
                print("  wrote \(stemURL.path)")
            }
        }
    }
}
