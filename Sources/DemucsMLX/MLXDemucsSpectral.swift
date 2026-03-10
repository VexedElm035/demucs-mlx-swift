import Accelerate
import Foundation
import MLX

struct DemucsComplexSpectrogram {
    let real: MLXArray
    let imag: MLXArray
}

final class DemucsSpectralPair {
    let nFFT: Int
    let hopLength: Int
    let freqBins: Int
    let center: Bool
    let window: [Float]

    private let forwardSetup: OpaquePointer
    private let inverseSetup: OpaquePointer

    init(nFFT: Int, hopLength: Int, center: Bool = true) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.freqBins = nFFT / 2 + 1
        self.center = center

        self.window = (0..<nFFT).map { n in
            0.5 * (1.0 - cos(2.0 * Float.pi * Float(n) / Float(nFFT)))
        }

        guard let fwd = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD) else {
            fatalError("Failed to create forward DFT setup")
        }
        forwardSetup = fwd

        guard let inv = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .INVERSE) else {
            fatalError("Failed to create inverse DFT setup")
        }
        inverseSetup = inv
    }

    deinit {
        vDSP_DFT_DestroySetup(forwardSetup)
        vDSP_DFT_DestroySetup(inverseSetup)
    }

    private func reflectPad(_ signal: [Float], left: Int, right: Int) -> [Float] {
        if signal.isEmpty {
            return [Float](repeating: 0, count: left + right)
        }

        var src = signal
        let maxPad = max(left, right)
        if src.count <= maxPad {
            let extra = maxPad - src.count + 1
            src.append(contentsOf: [Float](repeating: 0, count: extra))
        }

        var out = [Float]()
        out.reserveCapacity(src.count + left + right)

        if left > 0 {
            for i in 0..<left {
                let idx = max(0, min(src.count - 1, left - i))
                out.append(src[idx])
            }
        }

        out.append(contentsOf: src)

        if right > 0 {
            for i in 0..<right {
                let idx = max(0, min(src.count - 1, src.count - 2 - i))
                out.append(src[idx])
            }
        }

        return out
    }

    func stft(_ x: MLXArray) -> DemucsComplexSpectrogram {
        precondition(x.ndim == 3)

        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)

        let input = x.asArray(Float.self)

        var channelSignals: [[Float]] = []
        channelSignals.reserveCapacity(b * c)

        let centerPad = center ? (nFFT / 2) : 0
        for bc in 0..<(b * c) {
            let base = bc * t
            let sig = Array(input[base..<(base + t)])
            channelSignals.append(reflectPad(sig, left: centerPad, right: centerPad))
        }

        let frameCount = max(0, 1 + (channelSignals[0].count - nFFT) / hopLength)
        var outReal = [Float](repeating: 0, count: b * c * freqBins * frameCount)
        var outImag = [Float](repeating: 0, count: b * c * freqBins * frameCount)

        var frame = [Float](repeating: 0, count: nFFT)
        var inImag = [Float](repeating: 0, count: nFFT)
        var fftReal = [Float](repeating: 0, count: nFFT)
        var fftImag = [Float](repeating: 0, count: nFFT)

        for bc in 0..<(b * c) {
            let sig = channelSignals[bc]
            for fi in 0..<frameCount {
                let start = fi * hopLength
                for i in 0..<nFFT {
                    frame[i] = sig[start + i] * window[i]
                }
                inImag.withUnsafeMutableBufferPointer { ptr in
                    vDSP_vclr(ptr.baseAddress!, 1, vDSP_Length(nFFT))
                }

                vDSP_DFT_Execute(forwardSetup, frame, inImag, &fftReal, &fftImag)

                let base = ((bc * freqBins) * frameCount) + fi
                for f in 0..<freqBins {
                    let idx = base + f * frameCount
                    outReal[idx] = fftReal[f]
                    outImag[idx] = fftImag[f]
                }
            }
        }

        let shape = [b, c, freqBins, frameCount]
        return DemucsComplexSpectrogram(
            real: MLXArray(outReal).reshaped(shape),
            imag: MLXArray(outImag).reshaped(shape)
        )
    }

    func istft(_ z: DemucsComplexSpectrogram, length: Int) -> MLXArray {
        precondition(z.real.shape == z.imag.shape)
        precondition(z.real.ndim == 4 || z.real.ndim == 5)

        let ndim = z.real.ndim

        let outer: Int
        let finalShapePrefix: [Int]
        let frames: Int

        if ndim == 4 {
            let b = z.real.dim(0)
            let c = z.real.dim(1)
            frames = z.real.dim(3)
            outer = b * c
            finalShapePrefix = [b, c]
        } else {
            let b = z.real.dim(0)
            let s = z.real.dim(1)
            let c = z.real.dim(2)
            frames = z.real.dim(4)
            outer = b * s * c
            finalShapePrefix = [b, s, c]
        }

        let real = z.real.reshaped([outer, freqBins, frames]).asArray(Float.self)
        let imag = z.imag.reshaped([outer, freqBins, frames]).asArray(Float.self)

        let rawLength = nFFT + max(0, frames - 1) * hopLength
        let eps: Float = 1e-8

        var fullReal = [Float](repeating: 0, count: nFFT)
        var fullImag = [Float](repeating: 0, count: nFFT)
        var invReal = [Float](repeating: 0, count: nFFT)
        var invImag = [Float](repeating: 0, count: nFFT)

        var outAll = [Float](repeating: 0, count: outer * length)

        for o in 0..<outer {
            var signal = [Float](repeating: 0, count: rawLength)
            var denom = [Float](repeating: 0, count: rawLength)

            for fi in 0..<frames {
                for f in 0..<freqBins {
                    let idx = ((o * freqBins + f) * frames) + fi
                    fullReal[f] = real[idx]
                    fullImag[f] = imag[idx]
                }

                if nFFT >= 2 {
                    for k in 1..<(nFFT / 2) {
                        fullReal[nFFT - k] = fullReal[k]
                        fullImag[nFFT - k] = -fullImag[k]
                    }
                }

                vDSP_DFT_Execute(inverseSetup, fullReal, fullImag, &invReal, &invImag)

                var invScale = Float(1.0 / Float(nFFT))
                vDSP_vsmul(invReal, 1, &invScale, &invReal, 1, vDSP_Length(nFFT))

                let start = fi * hopLength
                for i in 0..<nFFT {
                    let sample = invReal[i] * window[i]
                    let wi = window[i] * window[i]
                    let dst = start + i
                    signal[dst] += sample
                    denom[dst] += wi
                }
            }

            for i in 0..<rawLength {
                signal[i] /= max(denom[i], eps)
            }

            var centered = signal
            if center {
                let pad = nFFT / 2
                if centered.count > (2 * pad) {
                    centered = Array(centered[pad..<(centered.count - pad)])
                } else {
                    centered = []
                }
            }

            if centered.count < length {
                centered.append(contentsOf: [Float](repeating: 0, count: length - centered.count))
            } else if centered.count > length {
                centered = Array(centered.prefix(length))
            }

            let base = o * length
            outAll.replaceSubrange(base..<(base + length), with: centered)
        }

        return MLXArray(outAll).reshaped(finalShapePrefix + [length])
    }
}
