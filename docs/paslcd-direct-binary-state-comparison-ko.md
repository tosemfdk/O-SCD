# PASLCD direct binary-state lifespan 비교

## 1. 질문과 범위

기존 O-SCD의 PASLCD 결과와 direct binary-state lifespan controller를 같은
20개 장면에서 비교한다. PASLCD는 `Instance_1/2 × 10 scenes`, 장면당 25개
inference/GT frame으로 총 500 frame이다.

이 평가는 evolving-scene transition benchmark가 아니다. 각 PASLCD 장면은
하나의 고정된 post-change scene만 포함한다. 따라서 성능 mask 비교에는 쓸 수
있지만, 장면 도중 실제 `active -> inactive -> active`가 일어나는지를 평가하지는
못한다.

## 2. 비교 프로토콜

- 해상도: `--resolution 4`
- pose: 기존 O-SCD가 저장한 fixed camera
- cue: 원래 O-SCD의 pixel cue + SAM2.1 Hiera Tiny candidate map을 float32로
  저장하고, binary mode에서 `candidate_map > 0.5` 적용
- detector evidence: immutable reference Gaussian의 lifespan-agnostic alpha-T
- evidence count: capped
- seed: 0
- GT/manual boundary: causal loop에서 사용하지 않고 모든 frame 처리 후 평가에만 사용
- direct topology: 고정, densification/pruning 없음
- B0: state-local DC만 최적화
- B1: state-local DC/xyz/opacity/scaling/rotation 최적화
- closed row-slot과 inactive pair: 이후 optimizer update에서 정확히 동결

두 update budget을 따로 평가했다.

- `u16`: 기존 O-SCD online의 16 updates/frame과 update 횟수를 맞춘 조건
- `u120`: 기존 ESCD direct-state 실험에서 사용한 120 updates/frame 조건

기존 baseline은 다음 frozen artifact를 그대로 읽었다.

```text
/home/rvl/workspace/github/O-SCD/artifacts/baseline/metrics_summary.json
/home/rvl/workspace/github/O-SCD/artifacts/baseline/metrics_per_scene.csv
```

Baseline commit은 `3abfeaaaef112ba9666b03993edf7d5a91fcaf74`이다.

## 3. 결과

`mIoU/F1`은 500 frame의 per-frame metric을 장면별로 평균한 뒤 20개 장면을
동일 가중 평균했다. Precision/recall은 장면별 aggregate pixel metric의 평균이다.

| 조건 | mIoU | F1 | Precision | Recall | O-SCD online 대비 mIoU/F1 | O-SCD refined 대비 mIoU/F1 |
|---|---:|---:|---:|---:|---:|---:|
| O-SCD online, 16 updates/frame | 0.4887 | 0.6423 | - | - | 기준 | -0.0686 / -0.0586 |
| O-SCD refined/offline, 3000 updates/scene | 0.5573 | 0.7009 | - | - | +0.0686 / +0.0586 | 기준 |
| u16 B0, direct binary DC-only | 0.4686 | 0.6241 | 0.7044 | 0.5970 | -0.0201 / -0.0182 | -0.0887 / -0.0768 |
| u16 B1, direct binary all-geometry | 0.5077 | 0.6598 | 0.6354 | 0.7311 | +0.0190 / +0.0175 | -0.0496 / -0.0411 |
| u120 B0, direct binary DC-only | 0.5008 | 0.6524 | 0.6574 | 0.6872 | +0.0121 / +0.0101 | -0.0565 / -0.0485 |
| u120 B1, direct binary all-geometry | **0.5147** | 0.6573 | 0.5569 | **0.8350** | **+0.0260 / +0.0150** | -0.0426 / -0.0436 |

결론은 다음과 같다.

1. update 수를 맞춘 `u16 B1`은 O-SCD online보다 `+0.0190 mIoU`,
   `+0.0175 F1` 높다.
2. 최고 mIoU는 `u120 B1`의 `0.5147`이지만, O-SCD refined `0.5573`보다
   `0.0426` 낮다.
3. 따라서 **기존 O-SCD online은 소폭 넘었지만 offline refinement는 넘지
   못했다**가 정확한 결론이다.

## 4. Geometry와 update budget의 효과

동일 detector decision에서 B0를 B1으로 바꾸면:

- u16: mIoU `+0.0391`, precision `-0.0690`, recall `+0.1341`
- u120: mIoU `+0.0139`, precision `-0.1004`, recall `+0.1478`

즉 geometry는 false negative를 많이 줄여 recall을 높이지만, 더 넓은 영역을
change로 렌더링하여 precision을 낮춘다.

u16에서 u120으로 update를 늘리면:

- B0: mIoU `+0.0322`, F1 `+0.0283`
- B1: mIoU `+0.0070`, F1 `-0.0025`

B1은 120 updates에서 recall이 `0.7311 -> 0.8350`으로 증가했지만 precision은
`0.6354 -> 0.5569`로 하락했다. 더 많은 geometry update가 항상 더 좋은
segmentation을 만들지는 않으며, 일부 장면에서는 false positive를 키웠다.

대표적으로 u120 B1은 O-SCD online 대비 `Instance_1/Zen`에서 `+0.1451`,
`Instance_2/Porch`에서 `+0.1117` mIoU를 얻었지만,
`Instance_1/Playground`에서 `-0.1249`, `Instance_2/Playground`에서
`-0.1075` mIoU를 잃었다.

## 5. OPEN 직후 optimization 효과

| 조건 | pre-opt mIoU | post-opt mIoU | pre→post delta |
|---|---:|---:|---:|
| u16 B0 | 0.4259 | 0.4686 | +0.0427 |
| u16 B1 | 0.4529 | 0.5077 | +0.0548 |
| u120 B0 | 0.4481 | 0.5008 | +0.0527 |
| u120 B1 | 0.4063 | 0.5147 | +0.1084 |

모든 500 frame에서 적어도 한 Gaussian OPEN이 발생했다. 그러므로 이 표의
OPEN-frame cohort는 사실상 전체 frame이며, 희소한 실제 onset latency를 뜻하지
않는다. 특히 u120 B1은 optimization 전 mask가 가장 약하고, frame 내부 120회
update에 가장 크게 의존한다.

## 6. Lifecycle 진단

모든 B0/B1 및 u16/u120 조건에서 detector event structure가 같았다.

```text
OPEN                         922,714
CLOSE                        446,622
REOPEN                       109,847
KEEP                       4,010,200
same-scene repeated events   556,469
repeated Gaussian rows       381,587
active->active false split         0
reused closed slot                 0
```

PASLCD의 각 장면은 post-change 상태가 촬영 중 바뀌지 않는다. 따라서 이 많은
CLOSE/REOPEN은 실제 evolving state를 찾은 결과가 아니라, viewpoint/cue 변화에
따라 direct binary belief가 흔들리는 **detector chattering**이다.

Lifecycle 구현 자체는 계약을 지켰다. B0/B1의 event sequence는 20/20 장면에서
동일했고, u16/u120 사이에도 동일했다. 즉 geometry와 update 수가 immutable-
reference detector로 feedback되지 않았다. 성능 차이는 같은 OPEN/CLOSE 결정
위에서 temporal representation을 얼마나 최적화했는지의 차이다.

## 7. 무결성 결과

80개 scene-condition run 전체에서:

- immutable base tensor max drift: `0`
- CLOSED slot parameter/optimizer max drift: `0`
- inactive gradient violation: `0`
- active→active false split: `0`
- closed slot reuse violation: `0`
- GT/manual boundary causal-use violation: `0`
- checkpoint 저장: 생략

Cue metadata에는 reference PLY와 fixed-camera SHA-256을 기록하며, PASLCD
benchmark resume 시 current camera file과 다시 대조한다.

## 8. 공정성 및 제한

1. `u16`은 update 횟수를 맞춘 비교지만 representation은 동일하지 않다.
   기존 O-SCD는 mutable topology와 densification/pruning을 사용하고, direct
   방식은 fixed topology/state-local delta를 사용한다.
2. O-SCD refined는 25 frame을 본 뒤 3000 total update를 수행하는 offline
   결과다. causal direct runner와 동일한 시간 정보 조건이 아니다.
3. direct runtime은 cached fixed pose와 cached cue를 사용한 runner 시간이다.
   기존 O-SCD baseline의 end-to-end `10m12s`와 직접적인 속도 비교를 하지 않는다.
4. `Instance_1/Lounge`의 primary camera JSON은 다른 해상도로 덮여 있었다.
   호환되는 seed-0 resolution-4 ablation camera를 사용했다. Frozen baseline이
   camera hash를 기록하지 않아 2026-07-14 baseline pose와 bitwise 동일하다고
   증명할 수 없다. `Garden`도 camera file timestamp가 baseline 이후라 동일한
   제한이 있다.
5. PASLCD에는 실제 repeated evolution ground truth가 없으므로 여기서 높은
   REOPEN count를 성공으로 해석하면 안 된다.

## 9. 재현 명령과 출력

Cue cache:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.prepare_paslcd_fixed_pose_cues \
  --cue-root outputs/paslcd_fixed_pose_cues_res4_v1 \
  --resolution 4
```

Benchmark:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_paslcd_binary_state_benchmark \
  --cue-root outputs/paslcd_fixed_pose_cues_res4_v1 \
  --output-root outputs/paslcd_direct_binary_state_benchmark_res4 \
  --conditions B0_binary_dc,B1_binary_all_geometry \
  --update-budgets 120,16 \
  --resume
```

Machine-readable 결과:

```text
outputs/paslcd_direct_binary_state_benchmark_res4/comparison.json
outputs/paslcd_direct_binary_state_benchmark_res4/comparison.md
outputs/paslcd_direct_binary_state_benchmark_res4/paslcd_binary_state_scene_metrics.csv
```

`outputs/`와 cue tensor는 실험 artifact이므로 Git에 커밋하지 않는다.

## 10. 다음 판단

All-geometry가 online mask 성능을 높일 수 있다는 것은 확인했다. 그러나 single-
state PASLCD에서도 대규모 CLOSE/REOPEN이 발생했으므로, 현재 direct filter를
evolving-scene lifespan detector로 확정할 수는 없다. 다음 detector 실험은
geometry 변경보다 먼저 emission/transition calibration, visibility persistence,
또는 3D neighborhood consistency로 chattering을 낮추되 ESCD의 실제
OPEN/CLOSE/REOPEN recall을 보존하는지를 평가해야 한다.
