# DA3 root seed: 3D + 2D occupancy 비교 (2026-09-07)

## 결론

`ref → SC1 → SC2 → SC3` 연속 304프레임에서 신규 root birth를 **69,030 → 11,334개(−83.58%)**로 줄였다. Mean-frame mIoU/F1은 **0.6403/0.7522 → 0.6420/0.7535**였고, 최종 joint mask의 NEW GT recall도 **0.8758 → 0.8826**으로 유지됐다.

따라서 이번 단일 seed-0 실험에서는 **seed 과잉 생성을 크게 억제하면서 측정 성능을 거의 유지**했다. mIoU +0.00174는 작은 point estimate이므로 통계적으로 확실한 성능 향상이라고 주장하지 않는다. 서로 다른 깊이의 물체를 잘못 막을 수 있는 depth-blind 2D occupancy의 위험도 남는다.

## 비교 조건과 범위

- 두 조건을 현재 코드의 동일 snapshot에서 처음부터 각각 실행했다. 기존 BF30 soft 결과를 baseline으로 재사용하지 않았다.
- 데이터: `data/Instance_1/scene_change1_2_3`, SC1 95 + SC2 104 + SC3 105 = 304 frames.
- Scene 경계에서 state를 reset하지 않는 연속 실험이다. 독립 `ref → SC1/SC2/SC3` 실험 평균이 아니다.
- First-ever OPEN BF10, CLOSE/REOPEN BF30, Gaussian별 alpha-T 합산 후 `binary_q05`; pixel cue는 unchanged.
- Soft Q, Panel10 NEW/나머지 loss 분리, u120, seed0, causal sampled replay 유지.
- DA3 총 seed cap=0(무제한), root birth 최대 1,024/frame, stride4.
- 변경한 항목은 **DA3 신규 root seed 생성 시 2D 중복 검사**뿐이다.
- NEW/REMOVE/APPEARANCE 분류, 학습 loss, detector evidence, BF, seed birth → single joint detector 순서, NEVER_OPEN/CLOSED render 정책, densification/pruning 기준은 바꾸지 않았다.
- 생성 topology가 달라지므로 이후 detector event와 학습 결과까지 동일하다는 뜻은 아니다.
- 기본값은 과거 실행 재현을 위해 `3d_only`로 보존했다. 새 조건은 `--da3-birth-coverage 3d_plus_2d --da3-coverage-2d-sigma 2`로 선택한다.

## 구현

```text
생성 허용 = 기존 NEW 후보 gate
            AND 기존 3D support 밖
            AND 기존/같은 프레임 신규 seed의 2D footprint 밖
```

기존 `uncovered_by_learned_gaussian_support` 3D 검사(기존 최소 geometry updates=4, sigma=2)를 그대로 두고, 그 뒤에 2D 검사를 추가했다.

### 2D occupancy 대상

- OPEN: 현재 학습된 xyz/scale/rotation. Geometry update 횟수와 무관하게 검사한다.
- NEVER_OPEN: 고정 birth probe geometry로 자리를 예약한다.
- CLOSED, retired, future-born, base GS는 제외한다.
- DC, mask threshold, opacity 값으로 검사 여부를 제한하지 않는다. 검은 pending seed도 중복 생성을 막는다.
- 카메라 뒤, 화면과 겹치지 않는 footprint는 제외한다.

### 투영 및 같은 프레임 예약

```text
Σ_world  = R_quat · diag(scale²) · R_quatᵀ
Σ_camera = R_camera · Σ_world · R_cameraᵀ
Σ_2D     = J_projection · Σ_camera · J_projectionᵀ
occupancy(pixel) = (pixel − center)ᵀ · Σ_2D⁻¹ · (pixel − center) ≤ 2²
```

중심점만 찍거나 bounding box 전체를 채우지 않고 실제 2σ 타원을 rasterize한다. SPD 판정/역행렬 계산은 CPU float64를 사용하고, determinant floor로 작은 타원을 인위적으로 넓히지 않는다. 전체 H×W×N 배열 대신 각 타원의 잘린 국소 영역만 처리한다.

Q 우선순위로 후보 중심 픽셀을 검사하고, 통과한 root의 타원을 임시 occupancy에 즉시 합친다. 따라서 동일 frame의 신규 후보끼리도 자리를 예약한다. 후보 중심이 비어 있는지 검사하는 방식이며, **두 타원 사이에 면적 겹침이 조금이라도 있으면 전부 금지하는 방식은 아니다.** 객체 전체를 막지 않으므로 실제로 비어 있는 테두리/구멍은 허용된다.

깊이 일치 또는 가려짐 검사는 이번 실험에 넣지 않았다. 서로 다른 깊이에서 같은 pixel을 덮는 두 경우(앞/뒤 순서 모두)에는 3D-only는 허용하지만 3D+2D는 차단하는 regression test를 추가해 이 제한을 명시했다.

## 전체 성능

mIoU와 F1은 각각 frame별 foreground IoU/F1의 304-frame 평균이다. NEW recall은 **base+seed 최종 joint prediction이 ADD GT를 맞춘 비율**로, seed-only NEW IoU가 아니다.

| 지표 | 3D-only | 3D+2D | 변화 |
|---|---:|---:|---:|
| Mean-frame mIoU | 0.640305 | 0.642050 | +0.001744 |
| Mean-frame F1 | 0.752227 | 0.753465 | +0.001238 |
| Pooled IoU | 0.687184 | 0.688188 | +0.001004 |
| Pooled F1 | 0.814593 | 0.815298 | +0.000705 |
| Pooled precision | 0.737325 | 0.735607 | -0.001719 |
| Pooled recall | 0.909951 | 0.914353 | +0.004402 |
| NEW GT pooled recall | 0.875806 | 0.882649 | +0.006843 |
| NEW 존재 frame 평균 recall | 0.848459 | 0.856780 | +0.008322 |
| REMOVED GT pooled recall | 0.965696 | 0.966120 | +0.000424 |
| 신규 root birth | 69,030 | 11,334 | −83.58% |
| Density child 생성 | 2,468 | 2,545 | +77 |
| 전체 seed archive (retired 포함) | 71,498 | 13,879 | −57,619 |
| 최종 non-retired seed | 26,341 | 6,317 | −20,024 |
| 최종 OPEN seed | 11,800 | 2,170 | −9,630 |

전체 304 frame 중 IoU 상승 161 / 하락 124 / 동일 19였다. FP는 2,759,103 → 2,797,109로 **38,006 증가**, FN은 766,429 → 728,962로 **37,467 감소**했다. 즉 중복 birth 억제를 곧바로 최종 FP 억제와 동일시하면 안 된다.

### 연속 스트림의 구간별 결과

각 셀은 `3D-only → 3D+2D`이다.

| 구간 | Frames | mIoU | F1 | NEW GT recall | 신규 root birth |
|---|---:|---:|---:|---:|---:|
| SC1 | 95 | 0.5972 → 0.5990 | 0.6801 → 0.6817 | 0.8676 → 0.8691 | 17,546 → 3,578 |
| SC2 | 104 | 0.6968 → 0.6990 | 0.8143 → 0.8158 | 0.9148 → 0.9251 | 26,207 → 3,945 |
| SC3 | 105 | 0.6233 → 0.6245 | 0.7560 → 0.7566 | 0.8400 → 0.8483 | 25,277 → 3,811 |

세 구간 모두 NEW recall의 큰 하락은 관찰되지 않았다. 다만 global pooled recall이 특정 객체/가림 사례의 recall 손실까지 없음을 증명하지는 않는다.

### Birth 차단 원인

| 항목 | 3D-only | 3D+2D |
|---|---:|---:|
| 기존 NEW 후보 gate 통과 | 194,787 | 194,787 |
| 3D support 차단 | 124,355 | 101,213 |
| 기존 seed 2D footprint 추가 차단 | 0 | 70,019 |
| 같은 frame 신규 seed 2D 예약 차단 | 0 | 12,221 |
| frame/cap budget 차단 | 1,402 | 0 |
| 최종 root 생성 | 69,030 | 11,334 |

모든 frame에서 `후보 = 3D 차단 + 기존2D 차단 + 같은frame2D 차단 + budget 차단 + 생성`을 확인했다. 두 run의 topology가 달라져 3D 차단 수도 달라지므로, 82,240개의 추가 2D 차단 수를 root 감소 57,696개와 그대로 동일시하면 안 된다.

Frame 243(global t=242)에서는 root birth가 663 → 69였고, 새 조건은 기존 seed footprint로 789개, 같은 frame 예약으로 70개를 차단했다. 이는 frame 전체의 count이며 특정 텀블러만의 count는 아니다.

Birth 단계의 합계 시간은 16.29 → 44.71초(304 frame에서 약 +0.093초/frame)였다. 전체 runtime은 3,222.86 → 3,171.97초였지만, 두 run과 다른 GPU 작업이 자원을 공유했으므로 통제된 속도 개선 결과로 해석하지 않는다.

## 검증

- 동일 configuration 비교 통과: 허용 차이는 coverage mode / port / capture directory뿐.
- Frozen runtime Python 150개 SHA256 일치, 두 run의 사용 코드 SHA256 동일.
- 두 조건 각각 304 frame, exit code 0.
- 저장된 lossless prediction PNG를 다시 읽어 GT TP/TN/FP/FN, IoU/F1, NEW/REMOVED recall count를 모두 정확히 재계산했다.
- 매 frame **seed birth 이후 단 한 번의 joint evidence pass**. Raw cue는 detector에만 해당 계약대로 투입했다.
- Reference/probe/frozen parameter/frozen Adam drift, future view access, lifespan render violation, binary evidence mixed-row violation 모두 0.
- 실제 CUDA에서 base·seed DC row 갱신을 확인했다. DC gradient 연결을 단순 CPU interface 검사만으로 대체하지 않았다.
- 타원 projection/rotation/가림 위험/NEVER_OPEN 예약/CLOSED·retired·future 제외/빈 공간/Q 순서/동일 frame 예약/count identity 및 기존 viewer·joint detector·loss·seed topology·timeline 회귀: **189 tests passed**.
- 별도의 actual-CUDA 8-frame smoke도 통과했다.
- Independent code review에서 blocking issue 없음. Formal LSP/ruff/mypy는 설치되지 않아 AST parse, whitespace, scoped pytest, 수동 static review로 대체했다. 전체 repo test suite는 실행하지 않았다.

## 전체 프레임 영상

요청대로 기존 panel ID 순서를 아래처럼 바꾸고, 화면 번호는 새 위치에 맞춰 1–10으로 다시 매겼다. Live viewer의 배치를 바꾼 것은 아니고 저장 결과의 offline video 구성이다.

```text
위:  기존 1 · 2 · 3 · 4 · 5
아래: 기존 8 · 9 · 10 · 6 · 7
       ↓    ↓     ↓    ↓    ↓
표시:  6    7     8    9   10

새6 Signed SAM / 새7 Depth / 새8 NEW·REMOVE·Appearance
새9 최종 mask / 새10 GT
하단: timestamp별 OPEN/CLOSE + 움직이는 cursor
```

- [3D+2D 결과 MP4](../outputs/escd_seed_3d2d_occupancy_20260907/video_3d_plus_2d/viewer_10panels_lifecycle_all_frames.mp4)
- [3D-only 비교 MP4](../outputs/escd_seed_3d2d_occupancy_20260907/video_3d_only/viewer_10panels_lifecycle_all_frames.mp4)
- [정확도·birth·NEW recall 비교 그래프](../outputs/escd_seed_3d2d_occupancy_20260907/comparison_timeline.png)

두 영상 모두 **304 frames / 10 fps / 30.4초 / 1920×1916 / H.264**. 저장 RGB/Q/GT는 원래 dashboard 해상도에서 추출하고 나머지 7개는 원본 standalone PNG를 썼다. GT/prediction을 재추론하거나 바꾸지 않았다. OPEN은 REOPEN 포함 base+seed commit 합계이며, birth/density/prune count가 아니다. 전체 ffmpeg/OpenCV decode, frame 순서, 7개 raw panel의 재배치 픽셀 일치, CSV-summary event 일치, source hash 보존을 검증했다.

전체 구간 y축 범위와 scene 경계는 사후 시각화 문맥이고 detector 입력이 아니다. 궤적 자체는 현재 timestamp까지만 표시한다. 이전 `outputs/panel10_lifecycle_timeline_5x2_20260907/`의 BF30 soft 영상은 덮어쓰지 않았다.

## 변경 파일과 재현

- `temporal/seed_projected_occupancy.py`: 2D covariance 타원/union, lifecycle 대상 선택.
- `experiments/view_bayesian_detector_steps.py`: 옵션, root birth 연결, frame별 차단 통계.
- `experiments/compose_panel10_lifecycle_video.py`: 명시적 source panel 순서 및 표시 번호 재배치.
- 관련 회귀: `tests/temporal/test_seed_projected_occupancy.py`, `tests/experiments/test_panel10_seed_occupancy.py`, `tests/experiments/test_view_bayesian_detector_steps.py`, `tests/experiments/test_compose_panel10_lifecycle_video.py`.
- 새 외부 dependency 없음. Renderer나 loss의 새 계층을 추가하지 않고 기존 3D 검사 뒤의 root birth gate만 확장했다.

[실행·분석 artifact root](../outputs/escd_seed_3d2d_occupancy_20260907/)에는 `run_case.sh`, `run_evaluation.py`, `runtime_source/`, `runtime_source_sha256.json`, `compare_results.py`, `comparison.json`, `tests_final.txt`, 각 run의 `configuration.json`, `frame_metrics.csv`, `audit.json`, `summary.json`, `posthoc_verification.json`이 있다. 출력 디렉터리는 새 경로를 써야 한다.

```bash
ROOT=outputs/escd_seed_3d2d_occupancy_20260907
bash "$ROOT/run_case.sh" 3d_only "$ROOT/repeat_3d_only" 8361
bash "$ROOT/run_case.sh" 3d_plus_2d "$ROOT/repeat_3d_plus_2d" 8362

PYTHONPATH=. /home/rvl/miniforge3/envs/oscd/bin/python \
  -m experiments.compose_panel10_lifecycle_video \
  --run-dir "$ROOT/3d_plus_2d" --output-dir "$ROOT/repeat_video_3d_plus_2d" \
  --panel-order 1 2 3 4 5 8 9 10 6 7
```

각 `video_*/verification.json`과 `encode_provenance.json`에 full-decode 결과, panel 순서, MP4 hash 및 encode 설정을 남겼다. 이 실험은 ESCD 한 연속 stream/seed0 범위이며 **PASLCD 20-scene 재실험이나 다중 seed 통계 검증은 포함하지 않는다.**
