import Foundation
import MLX
import MLXNN

protocol DemucsUnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

final class DemucsIdentity: Module, DemucsUnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

@inline(__always)
private func applyDemucsUnary(_ layer: Module, _ x: MLXArray) -> MLXArray {
    guard let unary = layer as? any DemucsUnaryLayer else {
        fatalError("Layer \(type(of: layer)) does not conform to DemucsUnaryLayer")
    }
    return unary.callAsFunction(x)
}

@inline(__always)
func demucsGELU(_ x: MLXArray) -> MLXArray {
    gelu(x)
}

@inline(__always)
func demucsGLU(_ x: MLXArray, axis: Int = 1) -> MLXArray {
    let parts = split(x, parts: 2, axis: axis)
    return parts[0] * sigmoid(parts[1])
}

@inline(__always)
func demucsCenterTrim(_ x: MLXArray, referenceLength: Int) -> MLXArray {
    let length = x.dim(-1)
    let delta = length - referenceLength
    if delta <= 0 {
        return x
    }
    let start = delta / 2
    let end = length - (delta - start)
    switch x.ndim {
    case 2:
        return x[0..., start..<end]
    case 3:
        return x[0..., 0..., start..<end]
    case 4:
        return x[0..., 0..., 0..., start..<end]
    case 5:
        return x[0..., 0..., 0..., 0..., start..<end]
    default:
        return x
    }
}

final class DemucsLayerScale: Module, DemucsUnaryLayer {
    @ParameterInfo(key: "scale") var scale: MLXArray
    private let channelLast: Bool

    init(channels: Int, initial: Float = 0.0, channelLast: Bool = false) {
        self.channelLast = channelLast
        self._scale.wrappedValue = MLXArray.zeros([channels]) + MLXArray(initial)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if channelLast {
            return x * scale
        }
        return x * scale.reshaped([scale.dim(0), 1])
    }
}

final class Conv1dNCL: Module, DemucsUnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(
        _ inputChannels: Int,
        _ outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nlc = x.transposed(0, 2, 1)
        let y = conv(nlc)
        return y.transposed(0, 2, 1)
    }
}

final class ConvTranspose1dNCL: Module, DemucsUnaryLayer {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(
        _ inputChannels: Int,
        _ outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        outputPadding: Int = 0,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            outputPadding: outputPadding,
            dilation: dilation,
            groups: 1,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nlc = x.transposed(0, 2, 1)
        let y = conv(nlc)
        return y.transposed(0, 2, 1)
    }
}

final class Conv2dNCHW: Module, DemucsUnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(
        _ inputChannels: Int,
        _ outputChannels: Int,
        kernelSize: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let y = conv(nhwc)
        return y.transposed(0, 3, 1, 2)
    }
}

final class ConvTranspose2dNCHW: Module, DemucsUnaryLayer {
    @ModuleInfo(key: "conv") var conv: ConvTransposed2d

    init(
        _ inputChannels: Int,
        _ outputChannels: Int,
        kernelSize: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        outputPadding: IntOrPair = 0,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = ConvTransposed2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            outputPadding: outputPadding,
            dilation: dilation,
            groups: 1,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let y = conv(nhwc)
        return y.transposed(0, 3, 1, 2)
    }
}

final class GroupNormNCL: Module, DemucsUnaryLayer {
    let groupCount: Int
    let channels: Int
    let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(groupCount: Int, channels: Int, eps: Float = 1e-5) {
        self.groupCount = groupCount
        self.channels = channels
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let g = groupCount
        precondition(c == channels && c % g == 0)

        let reshaped = x.reshaped([b, g, c / g, -1])
        let m = mean(reshaped, axes: [2, 3], keepDims: true)
        let v = variance(reshaped, axes: [2, 3], keepDims: true)
        var out = (reshaped - m) * rsqrt(v + MLXArray(eps))
        out = out.reshaped(x.shape)

        let wb = weight.reshaped([1, c, 1])
        let bb = bias.reshaped([1, c, 1])
        return out * wb + bb
    }
}

final class GroupNormNCHW: Module, DemucsUnaryLayer {
    let groupCount: Int
    let channels: Int
    let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(groupCount: Int, channels: Int, eps: Float = 1e-5) {
        self.groupCount = groupCount
        self.channels = channels
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let g = groupCount
        precondition(c == channels && c % g == 0)

        let reshaped = x.reshaped([b, g, c / g, -1])
        let m = mean(reshaped, axes: [2, 3], keepDims: true)
        let v = variance(reshaped, axes: [2, 3], keepDims: true)
        var out = (reshaped - m) * rsqrt(v + MLXArray(eps))
        out = out.reshaped(x.shape)

        let wb = weight.reshaped([1, c, 1, 1])
        let bb = bias.reshaped([1, c, 1, 1])
        return out * wb + bb
    }
}

enum DConvSlotMode {
    case conv
    case norm
    case normGELU
    case normGLU
    case identity
    case scale
}

final class DConvSlot: Module, DemucsUnaryLayer {
    let mode: DConvSlotMode
    let eps: Float = 1e-5

    @ModuleInfo(key: "conv") var conv: Conv1d?
    @ParameterInfo(key: "weight") var weight: MLXArray?
    @ParameterInfo(key: "bias") var bias: MLXArray?
    @ParameterInfo(key: "scale") var scale: MLXArray?

    init(
        mode: DConvSlotMode,
        inChannels: Int = 1,
        outChannels: Int = 1,
        kernelSize: Int = 1,
        dilation: Int = 1,
        padding: Int = 0,
        normChannels: Int = 1,
        scaleChannels: Int = 1,
        initialScale: Float = 0
    ) {
        self.mode = mode
        switch mode {
        case .conv:
            self._conv.wrappedValue = Conv1d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: kernelSize,
                stride: 1,
                padding: padding,
                dilation: dilation,
                groups: 1,
                bias: true
            )
            self._weight.wrappedValue = nil
            self._bias.wrappedValue = nil
            self._scale.wrappedValue = nil
        case .norm, .normGELU, .normGLU:
            self._conv.wrappedValue = nil
            self._weight.wrappedValue = MLXArray.ones([normChannels])
            self._bias.wrappedValue = MLXArray.zeros([normChannels])
            self._scale.wrappedValue = nil
        case .scale:
            self._conv.wrappedValue = nil
            self._weight.wrappedValue = nil
            self._bias.wrappedValue = nil
            self._scale.wrappedValue = MLXArray.zeros([scaleChannels]) + MLXArray(initialScale)
        case .identity:
            self._conv.wrappedValue = nil
            self._weight.wrappedValue = nil
            self._bias.wrappedValue = nil
            self._scale.wrappedValue = nil
        }
        super.init()
    }

    private func applyNorm(_ x: MLXArray) -> MLXArray {
        guard let weight, let bias else { fatalError("Missing norm parameters in DConvSlot") }
        let b = x.dim(0)
        let c = x.dim(1)
        let reshaped = x.reshaped([b, 1, c, -1])
        let m = mean(reshaped, axes: [2, 3], keepDims: true)
        let v = variance(reshaped, axes: [2, 3], keepDims: true)
        var out = (reshaped - m) * rsqrt(v + MLXArray(eps))
        out = out.reshaped(x.shape)
        return out * weight.reshaped([1, c, 1]) + bias.reshaped([1, c, 1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch mode {
        case .conv:
            guard let conv else { fatalError("Missing conv in DConvSlot.conv mode") }
            let nlc = x.transposed(0, 2, 1)
            let y = conv(nlc)
            return y.transposed(0, 2, 1)
        case .norm:
            return applyNorm(x)
        case .normGELU:
            return demucsGELU(applyNorm(x))
        case .normGLU:
            return demucsGLU(applyNorm(x), axis: 1)
        case .scale:
            guard let scale else { fatalError("Missing scale in DConvSlot.scale mode") }
            return x * scale.reshaped([scale.dim(0), 1])
        case .identity:
            return x
        }
    }
}

final class DConvBlock: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(
        channels: Int,
        hidden: Int,
        dilation: Int,
        kernel: Int,
        initialScale: Float,
        lstm: Bool = false,
        attn: Bool = false
    ) {
        let padding = dilation * (kernel / 2)

        // Build module list matching Python nn.Sequential structure:
        // [Conv, NormGELU, Identity, ?BLSTM, ?LocalState, Conv1x1, NormGLU, Identity, Scale]
        var mods: [Module] = [
            DConvSlot(
                mode: .conv,
                inChannels: channels,
                outChannels: hidden,
                kernelSize: kernel,
                dilation: dilation,
                padding: padding
            ),
            DConvSlot(mode: .normGELU, normChannels: hidden),
            DConvSlot(mode: .identity),
        ]

        // Insert attn first, then lstm at index 3 (lstm ends up before attn)
        if attn {
            mods.insert(DemucsLocalState(channels: hidden), at: 3)
        }
        if lstm {
            mods.insert(DemucsBLSTM(dim: hidden), at: 3)
        }

        mods.append(contentsOf: [
            DConvSlot(mode: .conv, inChannels: hidden, outChannels: 2 * channels, kernelSize: 1),
            DConvSlot(mode: .normGLU, normChannels: 2 * channels),
            DConvSlot(mode: .identity),
            DConvSlot(mode: .scale, scaleChannels: channels, initialScale: initialScale),
        ])

        self._layers.wrappedValue = mods
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Just run all modules in sequence (fused norm+act handles GELU/GLU)
        var y = x
        for layer in layers {
            if let unary = layer as? any DemucsUnaryLayer {
                y = unary.callAsFunction(y)
            }
        }
        return y
    }
}

final class DConv: Module, DemucsUnaryLayer {
    @ModuleInfo(key: "layers") var layers: [DConvBlock]

    init(
        channels: Int,
        compress: Float,
        depth: Int,
        initialScale: Float,
        kernel: Int = 3,
        lstm: Bool = false,
        attn: Bool = false
    ) {
        let hidden = max(1, Int(Float(channels) / compress))
        let count = abs(depth)
        let dilate = depth > 0
        self._layers.wrappedValue = (0..<count).map { idx in
            let dilation = dilate ? (1 << idx) : 1
            return DConvBlock(
                channels: channels,
                hidden: hidden,
                dilation: dilation,
                kernel: kernel,
                initialScale: initialScale,
                lstm: lstm,
                attn: attn
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in layers {
            y = y + layer(y)
        }
        return y
    }
}

final class HEncLayer: Module {
    let freq: Bool
    let strideValue: Int
    let empty: Bool
    let padValue: Int
    let fusedNorm1: Bool
    let fusedNorm2: Bool

    @ModuleInfo(key: "conv") var conv: Module
    @ModuleInfo(key: "norm1") var norm1: Module?
    @ModuleInfo(key: "norm2") var norm2: Module?
    @ModuleInfo(key: "rewrite") var rewrite: Module?

    @ModuleInfo(key: "dconv") var dconv: DConv?

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        normGroups: Int,
        empty: Bool,
        freq: Bool,
        dconvEnabled: Bool,
        normEnabled: Bool,
        context: Int,
        dconvDepth: Int,
        dconvComp: Float,
        dconvInit: Float,
        pad: Bool,
        rewrite: Bool,
        dconvLstm: Bool = false,
        dconvAttn: Bool = false
    ) {
        self.freq = freq
        self.strideValue = stride
        self.empty = empty
        self.padValue = pad ? (kernelSize / 4) : 0
        self.fusedNorm1 = normEnabled
        self.fusedNorm2 = normEnabled && rewrite

        let kFreq: IntOrPair = freq ? IntOrPair((kernelSize, 1)) : IntOrPair(kernelSize)
        let sFreq: IntOrPair = freq ? IntOrPair((stride, 1)) : IntOrPair(stride)
        let pFreq: IntOrPair = freq ? IntOrPair((self.padValue, 0)) : IntOrPair(self.padValue)

        if freq {
            self._conv.wrappedValue = Conv2dNCHW(
                inputChannels,
                outputChannels,
                kernelSize: kFreq,
                stride: sFreq,
                padding: pFreq,
                dilation: 1,
                groups: 1,
                bias: true
            )
        } else {
            self._conv.wrappedValue = Conv1dNCL(
                inputChannels,
                outputChannels,
                kernelSize: kernelSize,
                stride: stride,
                padding: self.padValue,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }

        // When empty, only conv is created (matching Python behavior)
        if empty {
            self._norm1.wrappedValue = nil
            self._norm2.wrappedValue = nil
            self._rewrite.wrappedValue = nil
            self._dconv.wrappedValue = nil
            super.init()
            return
        }

        if normEnabled {
            if freq {
                self._norm1.wrappedValue = GroupNormNCHW(groupCount: normGroups, channels: outputChannels)
            } else {
                self._norm1.wrappedValue = GroupNormNCL(groupCount: normGroups, channels: outputChannels)
            }
        } else {
            self._norm1.wrappedValue = DemucsIdentity()
        }

        let rewriteKernel = 1 + 2 * context
        if rewrite {
            if freq {
                self._rewrite.wrappedValue = Conv2dNCHW(
                    outputChannels,
                    2 * outputChannels,
                    kernelSize: IntOrPair(rewriteKernel),
                    stride: 1,
                    padding: IntOrPair(context),
                    dilation: 1,
                    groups: 1,
                    bias: true
                )
            } else {
                self._rewrite.wrappedValue = Conv1dNCL(
                    outputChannels,
                    2 * outputChannels,
                    kernelSize: rewriteKernel,
                    stride: 1,
                    padding: context,
                    dilation: 1,
                    groups: 1,
                    bias: true
                )
            }

            if normEnabled {
                if freq {
                    self._norm2.wrappedValue = GroupNormNCHW(groupCount: normGroups, channels: 2 * outputChannels)
                } else {
                    self._norm2.wrappedValue = GroupNormNCL(groupCount: normGroups, channels: 2 * outputChannels)
                }
            } else {
                self._norm2.wrappedValue = DemucsIdentity()
            }
        } else {
            self._rewrite.wrappedValue = nil
            self._norm2.wrappedValue = nil
        }

        if dconvEnabled && dconvDepth > 0 {
            self._dconv.wrappedValue = DConv(
                channels: outputChannels,
                compress: dconvComp,
                depth: dconvDepth,
                initialScale: dconvInit,
                lstm: dconvLstm,
                attn: dconvAttn
            )
        } else {
            self._dconv.wrappedValue = nil
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray, inject: MLXArray? = nil) -> MLXArray {
        var y = x
        if !freq && y.ndim == 4 {
            let b = y.dim(0)
            let c = y.dim(1)
            let t = y.dim(3)
            y = y.reshaped([b, c * y.dim(2), t])
        }

        if !freq {
            let length = y.dim(-1)
            let rem = length % strideValue
            if rem != 0 {
                var widths = [IntOrPair](repeating: 0, count: y.ndim)
                widths[widths.count - 1] = IntOrPair((0, strideValue - rem))
                y = padded(y, widths: widths, mode: .constant)
            }
        }

        y = applyDemucsUnary(conv, y)

        if empty {
            return y
        }

        if let inject {
            if inject.ndim == 3 && y.ndim == 4 {
                y = y + inject.expandedDimensions(axis: 2)
            } else {
                y = y + inject
            }
        }

        // Apply norm1 (with fused GELU if norm enabled, else separate GELU)
        if let n1 = norm1 {
            if fusedNorm1 {
                y = demucsGELU(applyDemucsUnary(n1, y))
            } else {
                y = demucsGELU(applyDemucsUnary(n1, y))
            }
        } else {
            y = demucsGELU(y)
        }

        if let dconvMod = dconv, dconvMod.layers.count > 0 {
            if freq {
                let b = y.dim(0)
                let c = y.dim(1)
                let f = y.dim(2)
                let t = y.dim(3)
                let folded = y.transposed(0, 2, 1, 3).reshaped([b * f, c, t])
                let unfolded = dconvMod(folded).reshaped([b, f, c, t]).transposed(0, 2, 1, 3)
                y = unfolded
            } else {
                y = dconvMod(y)
            }
        }

        if let rw = rewrite {
            let rewritten = applyDemucsUnary(rw, y)
            if let n2 = norm2 {
                if fusedNorm2 {
                    y = demucsGLU(applyDemucsUnary(n2, rewritten), axis: 1)
                } else {
                    y = demucsGLU(applyDemucsUnary(n2, rewritten), axis: 1)
                }
            } else {
                y = demucsGLU(rewritten, axis: 1)
            }
        }

        return y
    }
}

final class HDecLayer: Module {
    let last: Bool
    let freq: Bool
    let empty: Bool
    let padValue: Int
    let channelsIn: Int

    @ModuleInfo(key: "conv_tr") var convTr: Module
    @ModuleInfo(key: "norm2") var norm2: Module
    @ModuleInfo(key: "norm1") var norm1: Module?
    @ModuleInfo(key: "rewrite") var rewrite: Module?

    @ModuleInfo(key: "dconv") var dconv: DConv?

    init(
        inputChannels: Int,
        outputChannels: Int,
        last: Bool,
        kernelSize: Int,
        stride: Int,
        normGroups: Int,
        empty: Bool,
        freq: Bool,
        dconvEnabled: Bool,
        normEnabled: Bool,
        context: Int,
        dconvDepth: Int,
        dconvComp: Float,
        dconvInit: Float,
        pad: Bool,
        contextFreq: Bool,
        rewrite _: Bool,
        dconvLstm: Bool = false,
        dconvAttn: Bool = false
    ) {
        self.last = last
        self.freq = freq
        self.empty = empty
        self.padValue = pad ? (kernelSize / 4) : 0
        self.channelsIn = inputChannels

        if freq {
            self._convTr.wrappedValue = ConvTranspose2dNCHW(
                inputChannels,
                outputChannels,
                kernelSize: IntOrPair((kernelSize, 1)),
                stride: IntOrPair((stride, 1)),
                padding: 0,
                dilation: 1,
                outputPadding: 0,
                bias: true
            )
        } else {
            self._convTr.wrappedValue = ConvTranspose1dNCL(
                inputChannels,
                outputChannels,
                kernelSize: kernelSize,
                stride: stride,
                padding: 0,
                dilation: 1,
                outputPadding: 0,
                bias: true
            )
        }

        if normEnabled {
            if freq {
                self._norm2.wrappedValue = GroupNormNCHW(groupCount: normGroups, channels: outputChannels)
            } else {
                self._norm2.wrappedValue = GroupNormNCL(groupCount: normGroups, channels: outputChannels)
            }
        } else {
            self._norm2.wrappedValue = DemucsIdentity()
        }

        // When empty, only conv_tr and norm2 are created (matches Python behavior)
        if empty {
            self._norm1.wrappedValue = nil
            self._rewrite.wrappedValue = nil
            self._dconv.wrappedValue = nil
            super.init()
            return
        }

        if normEnabled {
            if freq {
                self._norm1.wrappedValue = GroupNormNCHW(groupCount: normGroups, channels: 2 * inputChannels)
            } else {
                self._norm1.wrappedValue = GroupNormNCL(groupCount: normGroups, channels: 2 * inputChannels)
            }
        } else {
            self._norm1.wrappedValue = DemucsIdentity()
        }

        let rewriteKernel = 1 + 2 * context
        let rewritePadding: IntOrPair
        let rewriteKernelPair: IntOrPair
        if freq {
            if contextFreq {
                rewriteKernelPair = IntOrPair(rewriteKernel)
                rewritePadding = IntOrPair(context)
            } else {
                rewriteKernelPair = IntOrPair((1, rewriteKernel))
                rewritePadding = IntOrPair((0, context))
            }
        } else {
            rewriteKernelPair = IntOrPair(rewriteKernel)
            rewritePadding = IntOrPair(context)
        }

        if freq {
            self._rewrite.wrappedValue = Conv2dNCHW(
                inputChannels,
                2 * inputChannels,
                kernelSize: rewriteKernelPair,
                stride: 1,
                padding: rewritePadding,
                dilation: 1,
                groups: 1,
                bias: true
            )
        } else {
            self._rewrite.wrappedValue = Conv1dNCL(
                inputChannels,
                2 * inputChannels,
                kernelSize: rewriteKernel,
                stride: 1,
                padding: context,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }

        self._dconv.wrappedValue = DConv(
            channels: inputChannels,
            compress: dconvComp,
            depth: dconvEnabled ? dconvDepth : 0,
            initialScale: dconvInit,
            lstm: dconvLstm,
            attn: dconvAttn
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray, length: Int) -> (MLXArray, MLXArray) {
        var y = x
        if freq && y.ndim == 3 {
            let b = y.dim(0)
            let t = y.dim(2)
            y = y.reshaped([b, channelsIn, -1, t])
        }

        var pre = y
        if !empty {
            pre = y + skip
            if let rewriteMod = rewrite, let norm1Mod = norm1 {
                let rewritten = applyDemucsUnary(rewriteMod, pre)
                pre = demucsGLU(applyDemucsUnary(norm1Mod, rewritten), axis: 1)
            }

            if let dconvMod = dconv, dconvMod.layers.count > 0 {
                if freq {
                    let b = pre.dim(0)
                    let c = pre.dim(1)
                    let f = pre.dim(2)
                    let t = pre.dim(3)
                    let folded = pre.transposed(0, 2, 1, 3).reshaped([b * f, c, t])
                    pre = dconvMod(folded).reshaped([b, f, c, t]).transposed(0, 2, 1, 3)
                } else {
                    pre = dconvMod(pre)
                }
            }
        }

        var z = applyDemucsUnary(convTr, pre)
        z = applyDemucsUnary(norm2, z)

        if freq {
            if padValue > 0 {
                let start = padValue
                let end = max(start, z.dim(-2) - padValue)
                z = z[0..., 0..., start..<end, 0...]
            }
        } else {
            let start = padValue
            let end = min(z.dim(-1), padValue + length)
            z = z[0..., 0..., start..<end]
        }

        if !last {
            z = demucsGELU(z)
        }

        return (z, pre)
    }
}
