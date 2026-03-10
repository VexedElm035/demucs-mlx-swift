import Foundation

/// Ensemble model that runs multiple sub-models and averages their outputs
/// with per-source weights.
///
/// Matches the Python MLX BagOfModelsMLX behavior.
final class BagOfModels: StemSeparationModel {
    let descriptor: DemucsModelDescriptor
    private let models: [StemSeparationModel]
    /// Per-model, per-source weights: weights[modelIdx][sourceIdx]
    private let weights: [[Float]]
    /// Sum of weights per source for normalization
    private let totals: [Float]

    init(
        descriptor: DemucsModelDescriptor,
        models: [StemSeparationModel],
        weights: [[Float]]?
    ) {
        self.descriptor = descriptor
        self.models = models

        let sourceCount = descriptor.sourceNames.count
        let defaultWeight = [Float](repeating: 1.0, count: sourceCount)

        if let weights, !weights.isEmpty {
            self.weights = weights.map { w in
                w.count == sourceCount ? w : defaultWeight
            }
        } else {
            self.weights = [[Float]](repeating: defaultWeight, count: models.count)
        }

        // Compute per-source totals for normalization
        var sums = [Float](repeating: 0.0, count: sourceCount)
        for modelWeights in self.weights {
            for s in 0..<sourceCount {
                sums[s] += modelWeights[s]
            }
        }
        self.totals = sums
    }

    func predict(
        batchData: [Float],
        batchSize: Int,
        channels: Int,
        frames: Int
    ) throws -> [Float] {
        let sourceCount = descriptor.sourceNames.count
        let outputSize = batchSize * sourceCount * channels * frames

        var accumulated = [Float](repeating: 0.0, count: outputSize)

        for (modelIdx, model) in models.enumerated() {
            let output = try model.predict(
                batchData: batchData,
                batchSize: batchSize,
                channels: channels,
                frames: frames
            )

            let w = weights[modelIdx]

            // Weighted accumulation per-source
            for b in 0..<batchSize {
                for s in 0..<sourceCount {
                    let weight = w[s]
                    if weight == 0 { continue }
                    for c in 0..<channels {
                        for t in 0..<frames {
                            let idx = ((b * sourceCount + s) * channels + c) * frames + t
                            accumulated[idx] += output[idx] * weight
                        }
                    }
                }
            }
        }

        // Normalize by total weight per source
        for b in 0..<batchSize {
            for s in 0..<sourceCount {
                let total = totals[s]
                if total <= 0 { continue }
                let invTotal = 1.0 / total
                for c in 0..<channels {
                    for t in 0..<frames {
                        let idx = ((b * sourceCount + s) * channels + c) * frames + t
                        accumulated[idx] *= invTotal
                    }
                }
            }
        }

        return accumulated
    }
}
