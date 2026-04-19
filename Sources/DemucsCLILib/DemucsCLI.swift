import ArgumentParser
import DemucsMLX
import CryptoKit
import Foundation
import MLX

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

    @Option(name: [.short, .long], help: "Tier preset: 1=performance (hdemucs_mmi), 2=balanceado (htdemucs), 3=calidad (mdx_extra_q), 4=balanced_ft (htdemucs_ft), 5=6stems (htdemucs_6s)")
    public var model: Tier?

    @Option(name: [.short, .long], help: "Output directory")
    public var out: String = "separated"

    @Option(name: .customLong("model-dir"), help: "Directory containing model files (.safetensors + _config.json)")
    public var modelDir: String?

    @Option(name: .long, help: "Segment length in seconds")
    public var segment: Double?

    @Option(name: .long, help: "Overlap ratio [0, 1)")
    public var overlap: Float?

    @Option(name: .long, help: "Number of shift augmentations")
    public var shifts: Int?

    @Option(name: .long, help: "Optional random seed for deterministic shifts")
    public var seed: Int?

    @Option(name: [.short, .long], help: "Chunk batch size")
    public var batchSize: Int = 1

    @Flag(name: .customLong("no-split"), help: "Disable chunked overlap-add inference")
    public var noSplit: Bool = false

    @Option(name: .customLong("two-stems"), help: "Only output the given stem and its complement (e.g. vocals produces vocals.wav and no_vocals.wav)")
    public var twoStems: String?

    @Flag(name: .customLong("list-models"), help: "List available models")
    public var listModels: Bool = false

    @Flag(name: .customLong("async"), help: "Use the closure-based async API with progress reporting")
    public var useAsync: Bool = false

    @Option(name: .customLong("cancel-after"), help: "Cancel separation after N seconds (for testing cancellation)")
    public var cancelAfter: Double?

    @Flag(name: .customLong("debug"), help: "Print performance metrics after separation")
    public var debug: Bool = false

    @Flag(name: .customLong("cpu"), help: "Force execution on CPU")
    public var forceCPU: Bool = false

    @Flag(name: .customLong("manifest"), help: "Write stems into a unique run folder and generate manifest.json")
    public var writeManifest: Bool = false

    @Flag(name: .customLong("no-parallel-write"), help: "Write stem files sequentially instead of in parallel (can reduce peak memory / footprint during output)")
    public var noParallelWrite: Bool = false

    // MARK: MLX memory controls

    @Option(
        name: .customLong("mlx-cache-limit"),
        help:
            "Limit MLX memory cache. Accepts bytes (e.g. 2097152), unit suffixes (e.g. 2mb, 512k, 1gb), or a small integer meaning MB (e.g. 2 -> 2MB). Default: unlimited."
    )
    public var mlxCacheLimit: String?

    @Flag(name: .customLong("mlx-clear-cache"), help: "Call Memory.clearCache() after each track (can reduce retained memory between runs)")
    public var mlxClearCache: Bool = false

    // MARK: Output format options

    @Flag(name: .customLong("mp3"), help: "Output as AAC in .m4a (Apple's lossy equivalent of MP3)")
    public var mp3: Bool = false

    @Flag(name: .customLong("wav"), help: "Output as WAV (default)")
    public var wav: Bool = false

    @Flag(name: .customLong("flac"), help: "Output as FLAC lossless")
    public var flac: Bool = false

    @Flag(name: .customLong("alac"), help: "Output as Apple Lossless (ALAC) in .m4a")
    public var alac: Bool = false

    @Flag(name: .customLong("int24"), help: "Output 24-bit integer WAV")
    public var int24: Bool = false

    @Flag(name: .customLong("float32"), help: "Output 32-bit float WAV")
    public var float32: Bool = false

    @Option(name: .customLong("mp3-bitrate"), help: "AAC bitrate in kbps when using --mp3 (default: 320)")
    public var mp3Bitrate: Int = 320

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

        // Resolve tier preset (if provided) into model name + default parameters.
        let tierPreset = model?.preset
        let resolvedModelName = tierPreset?.modelName ?? name
        let effectiveSegment = segment ?? tierPreset?.segmentSeconds
        let effectiveOverlap = overlap ?? tierPreset?.overlap ?? 0.25
        let effectiveShifts = shifts ?? tierPreset?.shifts ?? 1

        // Determine output format and file extension from flags.
        let (outputFormat, fileExtension) = try resolveOutputFormat()

        // Allow setting MLX cache limit via CLI flag (preferred) or environment variable (bytes / units).
        if let cacheLimitInput = mlxCacheLimit ?? ProcessInfo.processInfo.environment["DEMUCS_MLX_CACHE_LIMIT"] {
            let cacheLimitBytes = try Self.parseCacheLimitBytes(cacheLimitInput)
            Memory.cacheLimit = max(0, cacheLimitBytes)
        }

        let params = DemucsSeparationParameters(
            shifts: effectiveShifts,
            overlap: effectiveOverlap,
            split: !noSplit,
            segmentSeconds: effectiveSegment,
            batchSize: batchSize,
            seed: seed
        )

        if let tierPreset {
            print("Loading tier \(model!.rawValue) (\(tierPreset.label)) (model: \(resolvedModelName))...")
        } else {
            print("Loading model '\(resolvedModelName)'...")
        }
        let modelDirectoryURL = modelDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let separator = try DemucsSeparator(modelName: resolvedModelName, parameters: params, modelDirectory: modelDirectoryURL)
        print("Model loaded. Sources: \(separator.sources.joined(separator: ", "))")

        let outputRoot = URL(fileURLWithPath: out, isDirectory: true)
        try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)

        // Validate --two-stems value against the model's source names
        if let stem = twoStems {
            guard separator.sources.contains(stem) else {
                throw ValidationError(
                    "Stem \"\(stem)\" is not in the selected model. "
                    + "Must be one of: \(separator.sources.joined(separator: ", "))"
                )
            }
        }

        for track in tracks {
            let inputURL = URL(fileURLWithPath: track)
            print("\nSeparating: \(inputURL.path)")

            let totalStart = CFAbsoluteTimeGetCurrent()

            let result: DemucsSeparationResult
            let separationStart = CFAbsoluteTimeGetCurrent()
            if useAsync || cancelAfter != nil {
                if forceCPU {
                    // The current async API doesn't scope the MLX default device across the background queue.
                    // Fall back to synchronous separation so --cpu is deterministic.
                    print("  Note: --cpu disables --async; running synchronously on CPU")
                    result = try Device.withDefaultDevice(.cpu) {
                        try separator.separate(fileAt: inputURL)
                    }
                } else {
                    result = try Self.separateAsync(separator: separator, inputURL: inputURL, cancelAfterSeconds: cancelAfter)
                }
            }
            else {
                if forceCPU {
                    result = try Device.withDefaultDevice(.cpu) {
                        try separator.separate(fileAt: inputURL)
                    }
                } else {
                    result = try separator.separate(fileAt: inputURL)
                }
            }

            let separationElapsed = CFAbsoluteTimeGetCurrent() - separationStart
            print(String(format: "Separation complete (%.2fs)", separationElapsed))

            let trackBaseName = inputURL.deletingPathExtension().lastPathComponent
            let outputDir: URL
            let separationID: String
            if writeManifest {
                separationID = Self.uuidHex()
                let timestamp = Self.utcTimestampForFolder()
                let runFolder = "\(trackBaseName)_\(timestamp)_\(separationID.prefix(8))"
                outputDir = outputRoot.appendingPathComponent(runFolder, isDirectory: true)
            } else {
                separationID = ""
                outputDir = outputRoot.appendingPathComponent(trackBaseName, isDirectory: true)
            }
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            if let stem = twoStems {
                // Two-stem mode: write the selected stem and its complement
                guard let selectedAudio = result.stems[stem]
                else { continue }

                // Compute the complement: original mix minus the selected stem
                let mixSamples = result.input.channelMajorSamples
                let stemSamples = selectedAudio.channelMajorSamples
                var complementSamples = [Float](repeating: 0, count: mixSamples.count)
                for i in 0 ..< mixSamples.count {
                    complementSamples[i] = mixSamples[i] - stemSamples[i]
                }

                let complementAudio = try DemucsAudio(
                    channelMajor: complementSamples,
                    channels: selectedAudio.channels,
                    sampleRate: selectedAudio.sampleRate
                )

                // Write stems
                let stemFile = writeManifest ? "\(separationID)_\(stem).\(fileExtension)" : "\(stem).\(fileExtension)"
                let complementFile = writeManifest ? "\(separationID)_no_\(stem).\(fileExtension)" : "no_\(stem).\(fileExtension)"
                let stemURL = outputDir.appendingPathComponent(stemFile, isDirectory: false)
                let complementURL = outputDir.appendingPathComponent(complementFile, isDirectory: false)

                let writeError = Self.writeStems([
                    (selectedAudio, stemURL, outputFormat),
                    (complementAudio, complementURL, outputFormat),
                ], parallel: !noParallelWrite)
                if let error = writeError { throw error }
                print("  wrote \(stemURL.path)")
                print("  wrote \(complementURL.path)")

                if writeManifest {
                    let totalElapsed = CFAbsoluteTimeGetCurrent() - totalStart
                    try Self.writeManifest(
                        outputDir: outputDir,
                        inputURL: inputURL,
                        separationID: separationID,
                        tier: tierPreset?.label,
                        modelName: resolvedModelName,
                        deviceLabel: forceCPU ? "cpu" : Self.currentDeviceLabel(),
                        sources: [stem, "no_\(stem)"],
                        settings: ManifestSettings(
                            segment: effectiveSegment,
                            overlap: effectiveOverlap,
                            shifts: effectiveShifts,
                            forceCPU: forceCPU,
                            outputFormat: Self.outputFormatLabel(outputFormat: outputFormat, fileExtension: fileExtension),
                            mp3Bitrate: mp3Bitrate
                        ),
                        inputAudio: result.input,
                        stems: [stem: selectedAudio, "no_\(stem)": complementAudio],
                        stemFileExtension: fileExtension,
                        totalTimeSeconds: totalElapsed,
                        separationTimeSeconds: separationElapsed,
                        debug: debug
                    )
                } else if debug {
                    Self.printDebugSummary(
                        tier: tierPreset?.label,
                        modelName: resolvedModelName,
                        deviceLabel: forceCPU ? "cpu" : Self.currentDeviceLabel(),
                        stemCount: 2,
                        audioDurationSeconds: Double(result.input.frameCount) / Double(result.input.sampleRate),
                        separationTimeSeconds: separationElapsed,
                        totalTimeSeconds: CFAbsoluteTimeGetCurrent() - totalStart,
                        segmentSeconds: effectiveSegment,
                        overlap: effectiveOverlap,
                        shifts: effectiveShifts,
                        outputFormatLabel: Self.outputFormatLabel(outputFormat: outputFormat, fileExtension: fileExtension),
                        mp3BitrateKbps: mp3Bitrate
                    )
                }
            }
            else {
                // Normal mode: write all stems in parallel
                let writeJobs: [(DemucsAudio, URL, AudioOutputFormat)] = separator.sources.compactMap { source in
                    guard let stemAudio = result.stems[source] else { return nil }
                    let file = writeManifest ? "\(separationID)_\(source).\(fileExtension)" : "\(source).\(fileExtension)"
                    let stemURL = outputDir.appendingPathComponent(file, isDirectory: false)
                    return (stemAudio, stemURL, outputFormat)
                }

                let writeError = Self.writeStems(writeJobs, parallel: !noParallelWrite)
                if let error = writeError { throw error }
                for (_, url, _) in writeJobs {
                    print("  wrote \(url.path)")
                }

                let totalElapsed = CFAbsoluteTimeGetCurrent() - totalStart
                if writeManifest {
                    var stemsByName: [String: DemucsAudio] = [:]
                    for source in separator.sources {
                        if let audio = result.stems[source] {
                            stemsByName[source] = audio
                        }
                    }

                    try Self.writeManifest(
                        outputDir: outputDir,
                        inputURL: inputURL,
                        separationID: separationID,
                        tier: tierPreset?.label,
                        modelName: resolvedModelName,
                        deviceLabel: forceCPU ? "cpu" : Self.currentDeviceLabel(),
                        sources: separator.sources,
                        settings: ManifestSettings(
                            segment: effectiveSegment,
                            overlap: effectiveOverlap,
                            shifts: effectiveShifts,
                            forceCPU: forceCPU,
                            outputFormat: Self.outputFormatLabel(outputFormat: outputFormat, fileExtension: fileExtension),
                            mp3Bitrate: mp3Bitrate
                        ),
                        inputAudio: result.input,
                        stems: stemsByName,
                        stemFileExtension: fileExtension,
                        totalTimeSeconds: totalElapsed,
                        separationTimeSeconds: separationElapsed,
                        debug: debug
                    )
                } else if debug {
                    Self.printDebugSummary(
                        tier: tierPreset?.label,
                        modelName: resolvedModelName,
                        deviceLabel: forceCPU ? "cpu" : Self.currentDeviceLabel(),
                        stemCount: separator.sources.count,
                        audioDurationSeconds: Double(result.input.frameCount) / Double(result.input.sampleRate),
                        separationTimeSeconds: separationElapsed,
                        totalTimeSeconds: totalElapsed,
                        segmentSeconds: effectiveSegment,
                        overlap: effectiveOverlap,
                        shifts: effectiveShifts,
                        outputFormatLabel: Self.outputFormatLabel(outputFormat: outputFormat, fileExtension: fileExtension),
                        mp3BitrateKbps: mp3Bitrate
                    )
                }
            }

            // Optional: clear MLX cache after finishing this track.
            // This can reduce retained memory / peak footprint for one-off runs.
            if mlxClearCache || ProcessInfo.processInfo.environment["DEMUCS_MLX_CLEAR_CACHE"] == "1" {
                Memory.clearCache()
            }
        }
    }

    // MARK: - MLX cache limit parsing

    /// Parse a cache limit string into bytes.
    ///
    /// Supported inputs:
    /// - Raw bytes: "2097152"
    /// - Unit suffixes (case-insensitive): "512k", "512kb", "2m", "2mb", "1g", "1gb"
    /// - Small integers without units are treated as MB (to make "2" mean 2MB). To preserve
    ///   compatibility, large integers are treated as bytes.
    private static func parseCacheLimitBytes(_ raw: String) throws -> Int {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ValidationError("--mlx-cache-limit must not be empty")
        }

        let lower = trimmed.lowercased()

        // Split into numeric prefix + optional unit suffix.
        var numberPart = ""
        var unitPart = ""
        for ch in lower {
            if (ch >= "0" && ch <= "9") || ch == "." {
                if unitPart.isEmpty {
                    numberPart.append(ch)
                } else {
                    // Reject mixed forms like "1m2".
                    throw ValidationError("Invalid --mlx-cache-limit value: \(raw)")
                }
            } else if ch == " " {
                continue
            } else {
                unitPart.append(ch)
            }
        }

        guard let numeric = Double(numberPart), numeric.isFinite else {
            throw ValidationError("Invalid --mlx-cache-limit numeric value: \(raw)")
        }

        let multiplier: Double
        if unitPart.isEmpty {
            // Heuristic: treat small integers as MB; otherwise treat as bytes.
            // This keeps "2097152" behaving as bytes but allows "2" to mean 2MB.
            if numeric >= 0, numeric == floor(numeric), numeric <= 4096 {
                multiplier = 1024 * 1024
            } else {
                multiplier = 1
            }
        } else {
            switch unitPart {
            case "b":
                multiplier = 1
            case "k", "kb":
                multiplier = 1024
            case "m", "mb":
                multiplier = 1024 * 1024
            case "g", "gb":
                multiplier = 1024 * 1024 * 1024
            default:
                throw ValidationError("Invalid --mlx-cache-limit unit: \(unitPart). Use b/kb/mb/gb")
            }
        }

        let bytesDouble = numeric * multiplier
        if bytesDouble.isNaN || bytesDouble.isInfinite || bytesDouble < 0 {
            throw ValidationError("Invalid --mlx-cache-limit value: \(raw)")
        }

        // Clamp to Int range.
        if bytesDouble > Double(Int.max) {
            return Int.max
        }
        return Int(bytesDouble.rounded(.down))
    }

    // MARK: - Tier Presets

    public enum Tier: Int, CaseIterable, ExpressibleByArgument, Sendable {
        case performance = 1
        case balanceado = 2
        case calidad = 3
        case balancedFt = 4
        case sixStems = 5

        public var preset: TierPreset {
            switch self {
            case .performance:
                return TierPreset(label: "performance", modelName: "hdemucs_mmi", segmentSeconds: 2.0, overlap: 0.05, shifts: 0)
            case .balanceado:
                return TierPreset(label: "balanceado", modelName: "htdemucs", segmentSeconds: nil, overlap: 0.1, shifts: 1)
            case .calidad:
                return TierPreset(label: "calidad", modelName: "mdx_extra_q", segmentSeconds: 8.0, overlap: 0.25, shifts: 1)
            case .balancedFt:
                return TierPreset(label: "balanced_ft", modelName: "htdemucs_ft", segmentSeconds: nil, overlap: 0.1, shifts: 1)
            case .sixStems:
                return TierPreset(label: "6stems", modelName: "htdemucs_6s", segmentSeconds: nil, overlap: 0.1, shifts: 1)
            }
        }
    }

    public struct TierPreset: Sendable {
        public let label: String
        public let modelName: String
        public let segmentSeconds: Double?
        public let overlap: Float
        public let shifts: Int
    }

    // MARK: - Manifest

    private struct ManifestSettings: Codable {
        let segment: Double?
        let overlap: Float
        let shifts: Int
        let forceCPU: Bool
        let outputFormat: String
        let mp3Bitrate: Int

        enum CodingKeys: String, CodingKey {
            case segment
            case overlap
            case shifts
            case forceCPU = "force_cpu"
            case outputFormat = "output_format"
            case mp3Bitrate = "mp3_bitrate"
        }
    }

    private struct ManifestInput: Codable {
        let filename: String
        let sha256: String
        let sizeBytes: Int64
        let durationSec: Double
        let sampleRate: Int
        let audioChannels: Int

        enum CodingKeys: String, CodingKey {
            case filename
            case sha256
            case sizeBytes = "size_bytes"
            case durationSec = "duration_sec"
            case sampleRate = "sample_rate"
            case audioChannels = "audio_channels"
        }
    }

    private struct ManifestStem: Codable {
        let path: String
        let sha256: String
        let sizeBytes: Int64
        let durationSec: Double
        let sampleRate: Int
        let channels: Int

        enum CodingKeys: String, CodingKey {
            case path
            case sha256
            case sizeBytes = "size_bytes"
            case durationSec = "duration_sec"
            case sampleRate = "sample_rate"
            case channels
        }
    }

    private struct RunManifest: Codable {
        let version: Int
        let separationId: String
        let createdAt: String
        let tier: String?
        let model: String
        let device: String
        let stemCount: Int
        let sources: [String]
        let settings: ManifestSettings
        let input: ManifestInput
        let stems: [String: ManifestStem]

        enum CodingKeys: String, CodingKey {
            case version
            case separationId = "separation_id"
            case createdAt = "created_at"
            case tier
            case model
            case device
            case stemCount = "stem_count"
            case sources
            case settings
            case input
            case stems
        }
    }

    private static func writeManifest(
        outputDir: URL,
        inputURL: URL,
        separationID: String,
        tier: String?,
        modelName: String,
        deviceLabel: String,
        sources: [String],
        settings: ManifestSettings,
        inputAudio: DemucsAudio,
        stems: [String: DemucsAudio],
        stemFileExtension: String,
        totalTimeSeconds: Double,
        separationTimeSeconds: Double,
        debug: Bool
    ) throws {
        // Build stems section by inspecting the written files.
        var stemsManifest: [String: ManifestStem] = [:]
        stemsManifest.reserveCapacity(sources.count)

        for source in sources {
            guard let audio = stems[source] else { continue }
            let fileName = "\(separationID)_\(source).\(stemFileExtension)"
            let fileURL = outputDir.appendingPathComponent(fileName, isDirectory: false)
            let sha = try sha256File(fileURL)
            let size = try fileSizeBytes(fileURL)
            let duration = Double(audio.frameCount) / Double(audio.sampleRate)
            stemsManifest[source] = ManifestStem(
                path: fileName,
                sha256: sha,
                sizeBytes: size,
                durationSec: roundTo(duration, places: 6),
                sampleRate: audio.sampleRate,
                channels: audio.channels
            )
        }

        let inputSha = try sha256File(inputURL)
        let inputSize = try fileSizeBytes(inputURL)
        let inputDuration = Double(inputAudio.frameCount) / Double(inputAudio.sampleRate)

        let manifest = RunManifest(
            version: 1,
            separationId: separationID,
            createdAt: iso8601UTCNow(),
            tier: tier,
            model: modelName,
            device: deviceLabel,
            stemCount: stemsManifest.count,
            sources: sources,
            settings: settings,
            input: ManifestInput(
                filename: inputURL.lastPathComponent,
                sha256: inputSha,
                sizeBytes: inputSize,
                durationSec: roundTo(inputDuration, places: 6),
                sampleRate: inputAudio.sampleRate,
                audioChannels: inputAudio.channels
            ),
            stems: stemsManifest
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(manifest)

        let tmpURL = outputDir.appendingPathComponent("manifest.tmp", isDirectory: false)
        let manifestURL = outputDir.appendingPathComponent("manifest.json", isDirectory: false)
        try data.write(to: tmpURL, options: [.atomic])

        let fm = FileManager.default
        if fm.fileExists(atPath: manifestURL.path) {
            // Replace existing manifest.
            _ = try fm.replaceItemAt(manifestURL, withItemAt: tmpURL)
        } else {
            // First write.
            try fm.moveItem(at: tmpURL, to: manifestURL)
        }

        print("Manifest generated: \(manifestURL.path)")

        if debug {
            printDebugSummary(
                tier: tier,
                modelName: modelName,
                deviceLabel: deviceLabel,
                stemCount: stemsManifest.count,
                audioDurationSeconds: inputDuration,
                separationTimeSeconds: separationTimeSeconds,
                totalTimeSeconds: totalTimeSeconds,
                segmentSeconds: settings.segment,
                overlap: settings.overlap,
                shifts: settings.shifts,
                outputFormatLabel: settings.outputFormat,
                mp3BitrateKbps: settings.mp3Bitrate
            )
        }
    }

    private static func printDebugSummary(
        tier: String?,
        modelName: String,
        deviceLabel: String,
        stemCount: Int,
        audioDurationSeconds: Double,
        separationTimeSeconds: Double,
        totalTimeSeconds: Double,
        segmentSeconds: Double?,
        overlap: Float,
        shifts: Int,
        outputFormatLabel: String,
        mp3BitrateKbps: Int
    ) {
        let rtf = audioDurationSeconds > 0 ? separationTimeSeconds / audioDurationSeconds : Double.infinity
        print("\n=== Debug de Separacion ===")
        if let tier { print("Tier: \(tier)") }
        print("Modelo: \(modelName)")
        print("Cantidad de stems: \(stemCount)")
        print("Dispositivo: \(deviceLabel)")
        print(String(format: "Duracion del audio: %.2f s", audioDurationSeconds))
        print(String(format: "Tiempo de separacion (carga+inferencia): %.2f s", separationTimeSeconds))
        print(String(format: "Tiempo total (carga + inferencia + guardado): %.2f s", totalTimeSeconds))
        print(String(format: "Factor tiempo real (RTF): %.3fx", rtf))
        if let segmentSeconds {
            print("Segment usado en inferencia: \(segmentSeconds)")
        } else {
            print("Segment usado en inferencia: auto/modelo")
        }
        print("Overlap usado en inferencia: \(overlap)")
        print("Shifts usados en inferencia: \(shifts)")
        print("Formato de salida: \(outputFormatLabel)")
        if outputFormatLabel.contains("m4a") {
            print("AAC bitrate: \(mp3BitrateKbps) kbps")
        }
    }

    // MARK: - Utilities

    private static func outputFormatLabel(outputFormat: AudioOutputFormat, fileExtension: String) -> String {
        switch outputFormat {
        case .wav:
            return "wav"
        case .flac:
            return "flac"
        case .alac:
            return "alac"
        case .aac:
            // Keep naming consistent with actual extension.
            return fileExtension.lowercased()
        }
    }

    private static func currentDeviceLabel() -> String {
        switch Device.defaultDevice().deviceType {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case nil: return "unknown"
        }
    }

    private static func uuidHex() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    private static func utcTimestampForFolder() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        return formatter.string(from: Date())
    }

    private static func iso8601UTCNow() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }

    private static func roundTo(_ value: Double, places: Int) -> Double {
        guard places >= 0 else { return value }
        let p = pow(10.0, Double(places))
        return (value * p).rounded() / p
    }

    private static func fileSizeBytes(_ url: URL) throws -> Int64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static func sha256File(_ url: URL, chunkSize: Int = 1024 * 1024) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: chunkSize) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Stem Writing

    private static func writeStems(
        _ jobs: [(audio: DemucsAudio, url: URL, format: AudioOutputFormat)],
        parallel: Bool
    ) -> Error? {
        if !parallel {
            for job in jobs {
                do {
                    try AudioIO.writeAudio(job.audio, to: job.url, format: job.format)
                } catch {
                    return error
                }
            }
            return nil
        }
        return writeStemsParallel(jobs)
    }

    /// Write multiple stem files in parallel. Returns the first error encountered, or nil on success.
    private static func writeStemsParallel(
        _ jobs: [(audio: DemucsAudio, url: URL, format: AudioOutputFormat)]
    ) -> Error? {
        if jobs.isEmpty { return nil }
        if jobs.count == 1 {
            do {
                try AudioIO.writeAudio(jobs[0].audio, to: jobs[0].url, format: jobs[0].format)
                return nil
            } catch {
                return error
            }
        }

        final class ErrorBox: @unchecked Sendable {
            private let lock = NSLock()
            private var value: Error?

            func setIfNil(_ error: Error) {
                lock.lock()
                if value == nil { value = error }
                lock.unlock()
            }

            func get() -> Error? {
                lock.lock()
                defer { lock.unlock() }
                return value
            }
        }

        let errorBox = ErrorBox()
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "com.demucs.stem-writer", attributes: .concurrent)

        for job in jobs {
            group.enter()
            queue.async {
                do {
                    try AudioIO.writeAudio(job.audio, to: job.url, format: job.format)
                } catch {
                    errorBox.setIfNil(error)
                }
                group.leave()
            }
        }

        group.wait()
        return errorBox.get()
    }

    // MARK: - Async Separation

    /// Thread-safe box for mutable state shared across closures.
    private final class AsyncState: @unchecked Sendable {
        private let lock = NSLock()
        var result: Result<DemucsSeparationResult, Error>?
        var lastProgressLineLength: Int = 0

        func setResult(_ value: Result<DemucsSeparationResult, Error>) {
            self.lock.lock()
            self.result = value
            self.lock.unlock()
        }

        func getResult() -> Result<DemucsSeparationResult, Error>? {
            self.lock.lock()
            defer { self.lock.unlock() }
            return self.result
        }
    }

    /// Use the closure-based async API with progress reporting.
    /// Blocks the calling thread until the separation completes.
    private static func separateAsync(separator: DemucsSeparator, inputURL: URL, cancelAfterSeconds: Double? = nil) throws -> DemucsSeparationResult {
        let semaphore = DispatchSemaphore(value: 0)
        let state = AsyncState()
        let cancelToken = DemucsCancelToken()

        // Schedule cancellation after a delay if requested
        if let delay = cancelAfterSeconds {
            print("  Will cancel after \(delay)s...")
            DispatchQueue.global().asyncAfter(deadline: .now() + delay, execute: {
                print("\n  Cancelling separation...")
                cancelToken.cancel()
            })
        }

        separator.separate(
            fileAt: inputURL,
            cancelToken: cancelToken,
            progress: { progress in
                // Called on main queue - print progress
                let percent = Int(progress.fraction * 100)
                let bar = progressBar(fraction: progress.fraction, width: 30)
                let etaStr: String
                if let eta = progress.estimatedTimeRemaining, eta > 0 && progress.fraction < 1.0 {
                    let mins = Int(eta) / 60
                    let secs = Int(eta) % 60
                    etaStr = mins > 0 ? " ETA \(mins)m\(String(format: "%02d", secs))s" : " ETA \(secs)s"
                } else {
                    etaStr = ""
                }
                let line = "\r  [\(bar)] \(percent)% - \(progress.stage)\(etaStr)"
                let padded = line.padding(toLength: max(line.count, state.lastProgressLineLength), withPad: " ", startingAt: 0)
                print(padded, terminator: "")
                fflush(stdout)
                state.lastProgressLineLength = line.count

                if ProcessInfo.processInfo.environment["DEMUCS_BENCH"] != nil {
                    fputs("PROGRESS_TS \(CFAbsoluteTimeGetCurrent()) \(progress.fraction) \(progress.stage)\n", stderr)
                }
            },
            completion: { result in
                // Called on main queue
                state.setResult(result)
                semaphore.signal()
            }
        )

        // Run the main run loop so that main-queue callbacks can fire
        while semaphore.wait(timeout: .now() + 0.05) == .timedOut {
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.05))
        }

        // Clear the progress line
        print("")

        guard let result = state.getResult()
        else {
            throw DemucsError.cancelled
        }

        return try result.get()
    }

    /// Render a simple ASCII progress bar.
    private static func progressBar(fraction: Float, width: Int) -> String {
        let filled = Int(fraction * Float(width))
        let empty = width - filled
        return String(repeating: "=", count: filled) + String(repeating: " ", count: empty)
    }

    // MARK: - Format resolution

    private func resolveOutputFormat() throws -> (AudioOutputFormat, String) {
        // Count how many exclusive format flags were set.
        let formatFlags = [mp3, flac, alac, wav].filter { $0 }
        if formatFlags.count > 1 {
            throw ValidationError("Only one of --mp3, --flac, --alac, --wav may be specified")
        }

        // Bit depth flags are only relevant for WAV output.
        let bitDepthFlags = [int24, float32].filter { $0 }
        if bitDepthFlags.count > 1 {
            throw ValidationError("Only one of --int24, --float32 may be specified")
        }

        if mp3 {
            if !bitDepthFlags.isEmpty {
                throw ValidationError("Bit depth flags (--int24, --float32) are not applicable to AAC output")
            }
            let kbps = max(32, mp3Bitrate)
            return (.aac(bitRate: kbps * 1_000), "m4a")
        }

        if flac {
            if !bitDepthFlags.isEmpty {
                throw ValidationError("Bit depth flags (--int24, --float32) are not applicable to FLAC output")
            }
            return (.flac, "flac")
        }

        if alac {
            if !bitDepthFlags.isEmpty {
                throw ValidationError("Bit depth flags (--int24, --float32) are not applicable to ALAC output")
            }
            return (.alac, "m4a")
        }

        if wav {
            // Explicit WAV selection, fall through to WAV bit depth handling below.
        }

        // WAV output (default).
        let bitDepth: WAVBitDepth
        if int24 {
            bitDepth = .int24
        }
        else if float32 {
            bitDepth = .float32
        }
        else {
            bitDepth = .int16
        }
        return (.wav(bitDepth: bitDepth), "wav")
    }
}
