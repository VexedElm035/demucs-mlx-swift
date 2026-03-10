# Reference Comparison (2026-03-09)

Input:
`/Users/kylehowells/Developer/Github/demucs-mlx-swift/Sample Files/09 Still Into You.m4a`

## demucs-mlx vs demucs

- avg MAE: `0.006223`
- avg MSE: `0.000092`
- avg SDR (dB): `22.604`
- avg corr: `1.00209`
- avg max abs: `0.156288`

## swift_current vs demucs

- avg MAE: `0.084130`
- avg MSE: `0.014128`
- avg SDR (dB): `0.740`
- avg corr: `0.52876`
- avg max abs: `0.773697`

## swift_current vs demucs-mlx

- avg MAE: `0.084078`
- avg MSE: `0.014100`
- avg SDR (dB): `0.736`
- avg corr: `0.52876`
- avg max abs: `0.756073`

## Interpretation

- `demucs-mlx` closely tracks baseline `demucs` on this track.
- current Swift backend is not yet a true Demucs neural model and is far from reference quality.
