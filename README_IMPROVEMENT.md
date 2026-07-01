# MotionGS Improvement Notes

This branch corresponds to the course project **Reproduction and Stability Enhancement of MotionGS**.

## Summary of Improvements

### Improvement A: Isotropic Gaussian Flow

The anisotropic Gaussian-flow formula can become unstable around the warm-up boundary on newer PyTorch/CUDA stacks. The flow calculation in `utils/flow_utils.py` now uses isotropic center displacement between projected Gaussian centers and aggregates it with the rasterizer weights.

### Improvement B: Optional Delayed Pose Refinement

`train.py` supports delaying pose refinement until 50% of the total training iterations via `--delayed_pose_refine`. This is optional because the project experiments found it could reduce PSNR on some scenes.

### Windows Path Compatibility

`scene/dataset_readers.py` now derives the Nerfies scene name with `os.path.normpath`, so both Windows and POSIX-style paths are handled consistently, including paths with trailing separators.

## Environment Used for Reproduction

- Python 3.10
- PyTorch 2.4.0 with CUDA 12.4
- Windows 11, RTX 4060 Laptop 8 GB

## Reproduce

### Baseline: Deformable 3DGS, No Flow

```bash
python train.py -s data/plate \
  --densify_until_iter 9000 \
  --densify_grad_threshold 0.0003
```

### Isotropic Flow

```bash
python train.py -s data/plate --use_depth_and_flow \
  --densify_until_iter 9000 \
  --densify_grad_threshold 0.0003
```

### Delayed Pose Refinement

```bash
python train.py -s data/plate --use_depth_and_flow --optimize_pose \
  --delayed_pose_refine \
  --densify_until_iter 9000 \
  --densify_grad_threshold 0.0003
```
