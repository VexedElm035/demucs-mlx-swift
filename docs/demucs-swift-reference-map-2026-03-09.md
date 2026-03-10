# Demucs Swift Reference Map (2026-03-09)

## Goal
This document compares the available reference repositories and maps each one to concrete implementation needs for `demucs-mlx-swift`.

Focus: implementing Demucs in Swift with MLX parity against the Python MLX Demucs port and established MLX audio processing patterns.

## Quick Priority
1. **Primary model and algorithm parity**: `demucs-mlx`, `demucs`
2. **Primary Swift audio/DSP implementation patterns**: `mlx-audio-swift-master`, `qwen3-asr-swift`
3. **Primary framework internals and conversion patterns**: `mlx`, `mlx-examples`, `mlx-swift`
4. **Secondary/supporting references**: `mlx-audio`, `mlx-audio-master-codex`, `mlx-swift-examples`, `mlx-skills`

## Project-by-Project Comparison

| Project | Main value for us | Most relevant references | Demucs-Swift relevance |
|---|---|---|---|
| `demucs` | Canonical Demucs behavior and architecture | `demucs/apply.py`, `demucs/spec.py`, `demucs/demucs.py`, `demucs/hdemucs.py`, `demucs/htdemucs.py`, `demucs/wav.py`, `demucs/audio.py` | **Critical** |
| `demucs-mlx` | Closest execution/parity target in MLX | `demucs_mlx/apply_mlx.py`, `demucs_mlx/spec_mlx.py`, `demucs_mlx/wiener_mlx.py`, `demucs_mlx/mlx_layers.py`, `demucs_mlx/metal_kernels.py`, `demucs_mlx/mlx_htdemucs.py`, `demucs_mlx/mlx_hdemucs.py`, `demucs_mlx/mlx_convert.py`, `demucs_mlx/model_converter.py`, `demucs_mlx/separate.py` | **Critical** |
| `mlx-audio-swift-master` | Best Swift-native STFT/ISTFT and streaming DSP patterns | `Sources/MLXAudioCore/DSP.swift`, `Sources/MLXAudioCore/AudioUtils.swift`, `Sources/MLXAudioCore/PCMStreamConverter.swift`, `Sources/MLXAudioSTS/Models/DeepFilterNet/DeepFilterNetModel.swift`, `Sources/MLXAudioSTT/Streaming/IncrementalMelSpectrogram.swift`, `Sources/MLXAudioCodecs/Vocos/Vocos.swift` | **High** |
| `mlx-audio-master-codex` | Pure-MLX DeepFilterNet and streaming model runtime patterns | `mlx_audio/sts/models/deepfilternet/model.py`, `mlx_audio/sts/models/deepfilternet/streaming.py`, `mlx_audio/sts/models/deepfilternet/network.py`, `mlx_audio/sts/deepfilternet.py`, `notes/dfn-gml/deepfilternet-mlx-research.md` | Medium |
| `mlx-audio` | General Python audio DSP API patterns; baseline STFT/ISTFT code | `mlx_audio/dsp.py`, `mlx_audio/audio_io.py`, `mlx_audio/sts/models/deepfilternet/model.py` | Medium |
| `qwen3-asr-swift` | Strong Swift MLX audio preprocessing + streaming/chunking + conv wrappers | `Sources/Qwen3ASR/AudioPreprocessing.swift`, `Sources/Qwen3ASR/StreamingASR.swift`, `Sources/SpeechEnhancement/AudioProcessing.swift`, `Sources/SpeechEnhancement/SpeechEnhancement.swift`, `Sources/PersonaPlex/Conv.swift` | **High** |
| `mlx` | Core MLX behavior, FFT/compile/kernels semantics | `docs/src/python/fft.rst`, `docs/src/usage/compile.rst`, `docs/src/dev/custom_metal_kernels.rst`, `python/mlx/nn/layers/convolution.py` | **High** |
| `mlx-examples` | Model conversion and audio preprocessing examples | `whisper/mlx_whisper/audio.py`, `whisper/convert.py`, `encodec/README.md`, `musicgen/musicgen.py`, `musicgen/utils.py` | Medium |
| `mlx-skills` | Guidance/meta patterns (not implementation code) | `mlx_skills/skills/fast-mlx/SKILL.md`, `mlx_skills/skills/fast-mlx/references/fast-mlx-guide.md` | Low |
| `mlx-swift` | Swift API behavior and MLXFast/custom kernel interfaces | `README.md`, `Source/MLXFFT/FFT.swift`, `Source/MLXFast/MLXFastKernel.swift`, `skills/mlx-swift/references/neural-networks.md`, `skills/mlx-swift/references/custom-kernels.md` | **High** |
| `mlx-swift-examples` | App and tooling patterns, less direct audio-separation logic | `README.md`, `Tools/Tutorial/Tutorial.swift`, `Tools/llm-tool/LLMTool.swift`, `Applications/MLXChatExample/Services/MLXService.swift` | Low |

## Best References by Implementation Area

| Implementation area | Best source(s) | Why |
|---|---|---|
| Separation scheduling (split/overlap/shift/bag) | `demucs/apply.py`, `demucs-mlx/demucs_mlx/apply_mlx.py` | Defines canonical Demucs behavior and MLX-adapted parity details. |
| Spectrogram path (STFT/ISTFT wrappers, tensor shapes) | `demucs/spec.py`, `demucs-mlx/demucs_mlx/spec_mlx.py`, `mlx-audio-swift-master/Sources/MLXAudioCore/DSP.swift` | Gives parity math plus Swift-native DSP implementation style. |
| Wiener filtering + hybrid blend | `demucs/wav.py`, `demucs-mlx/demucs_mlx/wiener_mlx.py` | Required for Hybrid/HT Demucs quality parity. |
| Demucs/HDemucs/HTDemucs architecture details | `demucs/demucs.py`, `demucs/hdemucs.py`, `demucs/htdemucs.py`, `demucs-mlx/demucs_mlx/mlx_hdemucs.py`, `demucs-mlx/demucs_mlx/mlx_htdemucs.py` | Canonical architecture and the closest MLX translation. |
| NCL/NCHW bridging and conv wrappers | `demucs-mlx/demucs_mlx/mlx_layers.py`, `qwen3-asr-swift/Sources/PersonaPlex/Conv.swift`, `mlx/python/mlx/nn/layers/convolution.py` | Critical for avoiding silent shape/layout mismatches in Swift. |
| Audio I/O, resampling, channel remixing | `demucs/audio.py`, `demucs-mlx/demucs_mlx/separate.py`, `mlx-audio-swift-master/Sources/MLXAudioCore/AudioUtils.swift` | Directly maps to ingest/output correctness in Swift pipeline. |
| Streaming chunk-state patterns | `mlx-audio-swift-master/.../DeepFilterNetModel.swift`, `qwen3-asr-swift/Sources/Qwen3ASR/StreamingASR.swift`, `mlx-audio-master-codex/.../streaming.py` | Useful for future low-latency/incremental separation mode. |
| Model conversion/loading pipeline | `demucs-mlx/demucs_mlx/model_converter.py`, `demucs-mlx/demucs_mlx/mlx_convert.py`, `mlx-examples/whisper/convert.py` | Practical conversion strategies for checkpoints and key remapping. |
| Metal/custom kernel optimization | `demucs-mlx/demucs_mlx/metal_kernels.py`, `mlx/docs/src/dev/custom_metal_kernels.rst`, `mlx-swift/Source/MLXFast/MLXFastKernel.swift` | Needed for fusing hotspots once baseline parity is stable. |

## Recommended Reading/Implementation Order

1. `demucs-mlx/demucs_mlx/apply_mlx.py` and `demucs/apply.py`
2. `demucs-mlx/demucs_mlx/spec_mlx.py` and `demucs/spec.py`
3. `mlx-audio-swift-master/Sources/MLXAudioCore/DSP.swift`
4. `demucs-mlx/demucs_mlx/mlx_layers.py` and `qwen3-asr-swift/Sources/PersonaPlex/Conv.swift`
5. `demucs-mlx/demucs_mlx/mlx_hdemucs.py` + `mlx_htdemucs.py`
6. `demucs-mlx/demucs_mlx/wiener_mlx.py`
7. `demucs-mlx/demucs_mlx/model_converter.py` / `mlx_convert.py`
8. `mlx/docs/src/usage/compile.rst` and `custom_metal_kernels.rst`

## Mapping Into Current `demucs-mlx-swift`

Current files in `demucs-mlx-swift` already align to key areas:

- `Sources/DemucsMLX/SeparationEngine.swift`
  - Map to: `demucs-mlx/demucs_mlx/apply_mlx.py`, `demucs/apply.py`
  - Next focus: deterministic seeded shift behavior and exact overlap weighting parity.
- `Sources/DemucsMLX/AudioDSP.swift`
  - Map to: `demucs-mlx/demucs_mlx/spec_mlx.py`, `mlx-audio-swift-master/Sources/MLXAudioCore/DSP.swift`
  - Next focus: STFT/ISTFT framing/normalization parity, window and stride edge behavior.
- `Sources/DemucsMLX/MLXDemucsLayers.swift`
  - Map to: `demucs-mlx/demucs_mlx/mlx_layers.py`, `qwen3-asr-swift/Sources/PersonaPlex/Conv.swift`
  - Next focus: conv/group-norm axis/layout consistency checks (NCL <-> MLX native layouts).
- `Sources/DemucsMLX/MLXDemucsTransformer.swift`
  - Map to: `demucs-mlx/demucs_mlx/mlx_htdemucs.py`
  - Next focus: attention block parity and positional/timestep handling.
- `Sources/DemucsMLX/ModelProtocol.swift`
  - Map to: `demucs-mlx/demucs_mlx/mlx_registry.py`, `model_converter.py`
  - Next focus: model-selection/loading path parity with converted checkpoints.

## Practical Use of Each Project for This Effort

- Use `demucs-mlx` as the **behavioral source of truth** for MLX execution and checkpoint compatibility.
- Use `demucs` as the **algorithmic source of truth** when behavior in `demucs-mlx` is unclear.
- Use `mlx-audio-swift-master` + `qwen3-asr-swift` as the **Swift implementation templates** for DSP, chunking, and tensor-shape-safe layer wrappers.
- Use `mlx` + `mlx-swift` docs as the **framework authority** for compile, FFT, and custom kernel behavior.
- Use `mlx-examples` conversion patterns when refining checkpoint import/export tooling.

## Suggested Immediate Next Tasks

1. Lock parity tests for `split/overlap/shift` against `demucs-mlx` behavior.
2. Validate STFT/ISTFT numerical parity on fixed fixtures before tuning model blocks.
3. Finalize NCL/NCHW wrapper invariants with shape assertions in Swift layers.
4. Port/verify Wiener path for hybrid models before quality benchmarking.
