\# MotionGS — Improvement Branch



This branch corresponds to the course project:

\*\*"Reproduction and Stability Enhancement of MotionGS"\*\*



\## Summary of Improvements



\### Improvement A — Isotropic Gaussian Flow (Section 5.1 of report)

The original anisotropic Gaussian-flow formula suffers from catastrophic

cancellation at the warm-up boundary under modern hardware (PyTorch 2.4 +

CUDA 12.4 + RTX 4060). We replace it with an isotropic centre-displacement

formulation in `utils/flow\_utils.py`, restoring training stability and

achieving PSNR 29.04 dB vs. 7.71 dB (collapsed original).



\### Improvement B — Delayed Pose-Refinement Activation (Section 5.2 of report)

Modified `train.py` to postpone pose-refinement from iteration 3000 to 50%

of total iterations. Hypothesis was disproved: degraded PSNR by 0.7–2.2 dB.

Analysis in Section 7.1 of report.



\### Windows Compatibility Fixes

\- Path handling fix in `scene/dataset\_readers.py`

\- Backward CUDA kernel illegal memory access fix

\- Rasterizer antialiasing constructor argument fix



\## Environment

\- Python 3.10

\- PyTorch 2.4.0 + CUDA 12.4

\- Windows 11, RTX 4060 Laptop 8GB



\## Reproduce



\### Baseline (Deformable 3DGS, no flow)

```bash

python train.py -s data/plate \\

&#x20; --densify\_until\_iter 9000 \\

&#x20; --densify\_grad\_threshold 0.0003

```



\### Improvement A (isotropic flow)

```bash

python train.py -s data/plate --use\_depth\_and\_flow \\

&#x20; --use\_isotropic\_flow \\

&#x20; --densify\_until\_iter 9000 \\

&#x20; --densify\_grad\_threshold 0.0003

```



\### Improvement B (delayed pose refinement)

```bash

python train.py -s data/plate --use\_depth\_and\_flow \\

&#x20; --delayed\_pose\_refine \\

&#x20; --densify\_until\_iter 9000 \\

&#x20; --densify\_grad\_threshold 0.0003

```



\## Experiment Logs

See `logs/` directory for per-iteration diagnostics and VRAM usage data.

