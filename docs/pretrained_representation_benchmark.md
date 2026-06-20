# RGB-pretrained representation benchmark

This benchmark tests which event representation best transfers through a detector pretrained on RGB
images. Every variant uses the same COCO-pretrained Faster R-CNN ResNet-50 FPN. A small learnable
adapter maps the selected event representation to three channels. The final classification and
bounding-box predictor is replaced for the seven DSEC-MOT classes.

Adapter components are normalized independently so lower-range representations such as EROS are not
suppressed by EF/VG channels. Pretrained BatchNorm running statistics remain frozen during
fine-tuning because the physical batch size is small.

## Variants

- `event_frame_single`
- `voxel_grid_single`
- `event_frame_voxel_grid_single`
- `event_frame_voxel_grid_multi_branch`
- `eros_single`
- `event_frame_eros_single`
- `event_frame_eros_multi_branch`
- `voxel_grid_eros_single`
- `voxel_grid_eros_multi_branch`
- `event_frame_voxel_grid_eros_single`
- `event_frame_voxel_grid_eros_multi_branch`

`single` processes all input channels jointly. `multi_branch` first processes EF, VG, and EROS
components independently and then fuses their features before generating the pseudo-RGB input.

## EROS

EROS is maintained continuously through each sequence and snapshotted at DSEC-MOT image timestamps.
It is not reset for each 50-ms window. Exact event-by-event precomputation is accelerated with Numba:

```bash
.venv/bin/pip install numba
.venv/bin/python -m src.data.eros_precompute
```

The default local decay follows the PUCK Algorithm 1 interpretation
`d = 0.3 ** (1 / k_EROS)` with `k_EROS = 10`.

## Pilot

Run EF and EROS first to measure memory usage and training duration:

```bash
.venv/bin/python -m src.experiments.pretrained_representation_benchmark \
  --variants event_frame_single eros_single \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --device cuda
```

## Full benchmark

```bash
.venv/bin/python -m src.experiments.pretrained_representation_benchmark \
  --epochs 20 \
  --freeze-backbone-epochs 2 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --num-workers 4 \
  --device cuda
```

The queue skips existing checkpoints and evaluation summaries. Results are evaluated at score
thresholds `0.05`, `0.10`, `0.25`, `0.50`, `0.75`, `0.85`, `0.90`, and `0.95`.

## Summary

Thresholds are selected independently for each variant using validation HOTA. The corresponding test
results can be printed with:

```bash
.venv/bin/python -m src.experiments.summarise_pretrained_representation_benchmark
```
