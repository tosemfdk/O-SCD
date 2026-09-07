# Bayesian + DA3 historical lifespan replay checkpoint — 2026-09-06

## 결론

2026-09-04 viewer의 replay는 sampled frame `k`의 camera/cue를 사용하면서도 base와
DA3 Gaussian population은 최신 frame `t`의 lifecycle로 렌더했다. SC3에서 SC1/SC2
frame을 뽑으면 다음과 같은 잘못된 조합이었다.

```text
이전 구현
  camera / cue       = historical k
  Gaussian lifespan = current t

수정 구현
  camera / cue       = historical k
  base lifespan      = historical k
  DA3 lifespan       = historical k
  optimizer rows     = historical k에서 valid하고 sampled view에 visible한 row
```

이 mismatch를 제거하고 원본 O-SCD representation amplitude `2Q`를 사용한
304-frame × 120-update full run에서 mean-frame mIoU/F1은 `0.5249/0.6588`이었다.
이전 mismatched replay의 `0.5162/0.6497`보다 `+0.0087/+0.0091` 높았고, 가장 큰
회복은 SC3 mIoU `0.4534 -> 0.4925`였다.

따라서 **historical replay 자체가 SC1/SC2 cue를 현재 SC3와 상쇄한 것이 아니다.**
상쇄처럼 보였던 원인은 historical target과 current population을 섞은 구현이었다.

## 현재 canonical 실행

```bash
./run_bayesian_detector_viewer.sh
```

Launcher가 다음 두 축을 명시적으로 고정한다.

```text
--representation-cue-amplitude 2
--dc-replay-mode sampled
```

여기서 `sampled`는 더 이상 “과거 camera/cue + 현재 lifecycle”을 뜻하지 않는다.
항상 sampled timestamp의 lifecycle을 함께 조회하는 historical lifespan replay다.

## Timestamp 계약

Online current timestamp를 `t`, 한 update에서 뽑힌 replay timestamp를 `k`라고 하면
항상 `0 <= k <= t`다.

```text
DC branch
  camera                 = I_k
  target                 = 2Q_k
  base OPEN              = start <= k < end
  base NEVER_OPEN        = k까지 OPEN 이력 없음, black occluder
  base CLOSED            = k에서 제외
  seed OPEN              = materialized <= k and start <= k < end
  seed NEVER_OPEN        = materialized <= k and k까지 OPEN 이력 없음
  seed CLOSED            = k에서 제외
  seed born after k      = 존재하지 않는 row로 취급

Geometry branch
  camera                 = I_k
  target                 = Q_k
  trainable OPEN/pending = 위와 동일한 k 기준 population

Current output
  render timestamp       = t
  output population      = t에서 current-valid한 row만
```

Half-open interval `[start,end)`를 사용하므로 CLOSE frame `end`에서는 이미 CLOSED다.
동일 row가 CLOSE 뒤 REOPEN되면 새 interval slot을 만들고 timestamp query는 해당 시점에
유효한 interval만 반환한다.

## 구현 변경

### Base와 seed interval history

`experiments/view_bayesian_detector_steps.py`의 `DetectorReplayLifespan`이 다음 tensor를
보존한다.

- `materialized_timestamp[N]`
- `state_start[N,S]`
- `state_end[N,S]`
- `state_valid[N,S]`
- `current_state_index[N]`
- `num_states[N]`

추가 query는 `materialized_mask(k)`, `get_active_state_indices(k)`, `active_mask(k)`,
`never_open_mask(k)`, `closed_mask(k)`다. Base는 처음부터 materialize되고, DA3 row는
실제 birth frame에 accepted될 때만 history와 representation sidecar에 append된다.

### Replay render와 optimizer

`_train_representation_update()`는 `current_timestamp=t`와 별도로 다음 timestamp를
사용한다.

- `geometry_timestamp = item.timestamp`
- `dc_timestamp = dc_item.timestamp`

기본 `sampled` mode에서는 둘 다 `k`다. `current` ablation에서만 DC timestamp를 `t`로
바꾸며 geometry replay는 계속 `k`를 사용한다.

`temporal/new_seed_gaussians.py`의 `build_concatenated_change_view()`에는 historical
seed OPEN/NEVER_OPEN mask를 명시적으로 전달한다. 이 view는:

- `k`의 OPEN seed에 learned DC를 사용한다.
- `k`의 NEVER_OPEN seed에 정확한 black `RGB2SH(0)`을 사용한다.
- `k`의 CLOSED와 future-born seed를 concatenate하지 않는다.
- OPEN seed DC에만 gradient를 연결한다.

Base와 seed optimizer도 render radius가 양수인 `k`-valid row만 갱신한다. Off-view,
`k`-CLOSED, future-born row의 parameter와 Adam moment는 그 update에서 보존된다.

### Detector 독립성

이 수정은 representation replay만 바꾼다.

- Base detector: 신규 frame 도착 직후 immutable reference의 learned-Q alpha-T evidence
- Seed detector: birth 다음 신규 frame부터 fixed birth-geometry probe의 untouched
  `P+S > 0.5` evidence
- Learned DC/geometry, replay render, post-optimization render: detector input으로 역류 안 함
- GT: step과 representation update가 끝난 뒤 평가에만 사용

## 304-frame full 결과

산출물:

- `outputs/bayesian_da3_historical_replay_20260906/B_h1_amplitude2/summary.json`
- `outputs/bayesian_da3_historical_replay_20260906/B_h1_amplitude2/frame_metrics.csv`
- `outputs/bayesian_da3_historical_replay_20260906/B_h1_amplitude2/controlled_state.pt`
- `outputs/bayesian_da3_historical_replay_20260906/B_h1_amplitude2/diagnostics/`

### GT ADD union REMOVE

| metric | 이전 mismatched B | 수정 historical B | delta |
|---|---:|---:|---:|
| mean-frame mIoU | 0.5162 | **0.5249** | **+0.0087** |
| mean-frame F1 | 0.6497 | **0.6588** | **+0.0091** |
| precision | 0.6536 | 0.6401 | -0.0135 |
| recall | 0.8008 | **0.8567** | **+0.0559** |
| cue-mask mIoU | 0.4746 | **0.5030** | **+0.0283** |

### Scene별 mean-frame mIoU

| scene | 이전 mismatched B | 수정 historical B | delta |
|---|---:|---:|---:|
| SC1 | 0.4782 | 0.4750 | -0.0032 |
| SC2 | 0.6144 | 0.6033 | -0.0111 |
| SC3 | 0.4534 | **0.4925** | **+0.0391** |

Current-only amplitude-2 control F의 overall mIoU는 `0.5237`이었다. Correct historical
B는 이를 `+0.0012` 넘었으며, multiview replay를 보존하면서 current-only와 같은 수준의
현재-state 성능을 얻었다. F1과 cue IoU는 F가 각각 `+0.0008`, `+0.0147` 높으므로
모든 지표에서 historical이 우월하다고 해석하지 않는다.

### Cue/DC 진단

- high-Q pixel learned mean: `0.6859`
- high-Q pixel 중 score `>=0.5`: `0.7307`
- low-Q pixel learned mean: `0.0285`
- learned / white-ceiling mass ratio: `0.8027`
- coverage-limited high-Q fraction: `0.1205`
- exact autograd finite-difference relative error: `0.00163`

### Lifecycle와 causal audit

- historical lifespan replay updates: `23,916`
- lifespan/future-row render violations: `0`
- future-view accesses: `0`
- base OPEN/CLOSE: `80,742 / 22,062`
- seed OPEN/CLOSE: `5,886 / 2,863`
- DA3 proposed/accepted/coverage-rejected: `65,679 / 5,853 / 59,826`
- runtime: `916.60 s`
- peak CUDA memory: `10,710,277,632 bytes`

Dynamic learned-Gaussian coverage 때문에 이전 B와 accepted seed topology가 bitwise 같지는
않다. 따라서 작은 delta는 topology 변동을 포함한 point estimate다. 반면 interval
query, future-born exclusion, OPEN-only gradient 경로는 regression test와 runtime audit로
직접 검증했다.

## Parameter sharing 계약

현재 viewer의 base `change_dc`와 DA3 DC/geometry는 **Gaussian row마다 지속되는
parameter**다. Historical replay는 `k`에서 살아 있던 row를 정확히 선택하고 그 row의
현재 parameter를 학습한다. SC1/SC2/SC3의 causal replay view가 같은 Gaussian row를
관측하면 같은 parameter와 Adam moment를 공동 최적화한다. 이것이 의도한 multiview
fusion이며 과거 시점의 parameter snapshot을 저장했다가 되감지 않는다.

따라서 이번 수정이 보장하는 것은 다음이다.

- sampled frame과 같은 timestamp의 Gaussian population
- sampled timestamp에 맞는 OPEN/NEVER_OPEN/CLOSED gate
- sampled timestamp 이후 태어난 seed의 완전한 제외
- sampled timestamp에서 visible하고 trainable한 row만 optimizer update

현재 full run의 interval history에는 base multi-interval row `3,862`개, seed
multi-interval row `842`개가 있으며 양쪽 모두 row당 최대 6개 interval이 있었다.
이 interval들은 같은 Gaussian row의 liveness 구간이며 동일 row parameter를 공유한다.
예를 들어 SC3 시점에 SC1 frame `k`를 replay하면 parameter 값을 SC1 당시 값으로
되돌리는 대신, `k`에서 ACTIVE였던 row들만 현재 shared parameter로 렌더하고 그
parameter를 SC1 관측에도 맞도록 추가 최적화한다.

## 검증

수정 직후 수행한 검증:

- 관련 unit/regression tests: `90 passed`
- changed Python modules `py_compile`: passed
- 14-frame CUDA smoke: historical updates `841`, violation `0`, future access `0`
- 304-frame × 120-update full run: historical updates `23,916`, violation `0`, future access `0`

핵심 regression은 다음을 고정한다.

1. Base OPEN→CLOSE→REOPEN half-open history query
2. DA3 materialization 이전 timestamp에서 future row 완전 제외
3. Historical OPEN seed만 DC gradient를 받고 NEVER_OPEN/future row는 받지 않음
4. Base attributes가 최신 state가 아니라 requested replay timestamp를 따름

## 이전 checkpoint 처리

- `docs/bayesian-da3-viewer-pipeline-checkpoint-20260904-ko.md`: 이전 mismatch 비교 기록
- `docs/bayesian-da3-viewer-pipeline-checkpoint-20260904.json`: superseded manifest
- 이 문서와 동명의 JSON: 현재 canonical replacement

## 후속 viewer-only geometry variant

2026-09-06에 다음 대화형 variant를 추가했다. 이는 위 `0.5249` full result를 다시
측정한 조건이 아니라 viewer 육안 점검용 조건이다.

```bash
./run_bayesian_detector_viewer.sh \
  --no-train-never-open-geometry \
  --geometry-cue-amplitude 2
```

- Base geometry: frozen
- DA3 NEVER_OPEN xyz/scale/rotation: frozen
- DA3 OPEN geometry: historical timestamp의 OPEN row만 학습
- DC target: `2Q_k`
- Geometry target: unclipped `2Q_k`
