# Temporal Geometry State-valid Viewer

`temporal_geometry_oscd_cues_allframes_120` 체크포인트의 state-specific
DC/xyz/opacity/scale/rotation을 사용해 각 state의 `state_valid=True` Gaussian만
검사한다.

## 실행

```bash
conda run --no-capture-output -n oscd python viewer.py \
  --ref_ply data/Instance_1/scene_change1_2_3/reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply \
  --temporal_checkpoint outputs/instance1_scene_change1_2_3_temporal_geometry_oscd_cues_allframes_120/temporal_rchange_checkpoint.pt \
  --resolution 4 \
  --skip_inference_images \
  --port 8097
```

## 조작

- `Active Scene`: `Temporal State-valid GS`
- `Temporal State`: State 0, 1, 2 전환
- `State-valid highlight`: valid GS를 state별 단색으로 표시
- `Learned change DC`: 해당 state의 실제 학습 DC와 geometry로 change를 렌더링
- GS 클릭: state validity, lifespan, DC, activated geometry 및 geometry delta 표시

체크포인트의 valid GS 수는 State 0 `576,372`, State 1 `685,532`, State 2
`653,430`이다. 이는 전체 Gaussian을 의미하지 않으며, O-SCD support map에서
누적 raster contribution이 support threshold를 통과한 Gaussian이다.
