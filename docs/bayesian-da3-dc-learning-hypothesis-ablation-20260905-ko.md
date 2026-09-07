# Bayesian + DA3 DC 학습 실패 가설 검증 — 2026-09-05

## 2026-09-06 후속 정정

이 문서의 2026-09-05 H4 진단대로 기존 `sampled replay`는 camera/cue만 과거 `k`를
사용하고 Gaussian lifecycle은 최신 `t`를 사용했다. 2026-09-06에 base와 DA3 모두
half-open interval history를 보존하고, sampled update를 다음 계약으로 수정했다.

```text
camera = I_k
DC target = 2Q_k
geometry target = Q_k
base render/optimizer population = L_base(k)
DA3 render/optimizer population = L_seed(k)
future-born DA3 rows = excluded
```

수정된 `B_h1_amplitude2`를 304 frames × 120 updates로 다시 실행했다.

| run | SC1 | SC2 | SC3 | mean-frame mIoU | F1 | cue IoU |
|---|---:|---:|---:|---:|---:|---:|
| 이전 mismatched B | 0.4782 | 0.6144 | 0.4534 | 0.5162 | 0.6497 | 0.4746 |
| 수정 historical-lifespan B | 0.4750 | 0.6033 | 0.4925 | **0.5249** | **0.6588** | **0.5030** |
| 수정 - 이전 | -0.0032 | -0.0111 | **+0.0391** | **+0.0087** | **+0.0091** | **+0.0283** |

- 실제 historical-lifespan update: `23,916`
- lifecycle/future-row render violation: `0`
- future-view access: `0`
- base OPEN/CLOSE: `80,742 / 22,062`
- seed OPEN/CLOSE: `5,886 / 2,863`

따라서 과거 replay가 본질적으로 DC를 상쇄한다는 해석은 폐기한다. 문제는 과거 target과
현재 lifecycle population을 섞은 구현이었다. 올바른 historical replay는 이전
current-only 상한 F의 mIoU `0.5237`도 `+0.0012` 넘어섰다. 현재 canonical은
`2Q + joint SSF + historical lifespan replay`다. 상세 구현·결과는
[`bayesian-da3-historical-lifespan-replay-20260906-ko.md`](bayesian-da3-historical-lifespan-replay-20260906-ko.md)에
정리한다.

현재 viewer는 row마다 하나의 learnable DC/geometry를 유지하며 historical/current
view가 이를 공동 최적화한다. 이는 의도한 multiview parameter sharing이다. Lifespan은
sampled timestamp마다 그 row의 render/optimizer 참여 여부만 결정하며 parameter
snapshot을 되감거나 interval별로 복제하지 않는다.

## 2026-09-05 실험 당시 결론

OPEN 자체는 정상인데 learned DC color가 cue를 충분히 따라가지 않은 가장 큰 원인은
**loss 식이 아니라 loss에 들어간 cue amplitude**였다.

현재 canonical viewer는 learned sigmoid 뒤의 `candidate_map = 2Q`를 detector용으로는
올바르게 사용하지만, representation replay에 저장할 때 다시 `cue_scale=2`로 나누어
`Q ∈ [0,1]`로 만든다. 그 뒤 원본 O-SCD SSF 식을 호출한다. 즉 **loss 코드는 원본이지만
loss input scale은 원본 O-SCD의 0..2 scale이 아니었다.**

304-frame full run의 가장 큰 단독 효과는 다음과 같았다.

- `Q -> 2Q`만 복원: mean-frame mIoU `0.2877 -> 0.5162`, `+0.2285`
- DC만 current view로 학습: `0.2877 -> 0.4656`, `+0.1779`
- projected seed-local BCE만 사용: `0.2877 -> 0.4048`, `+0.1170`

최고 condition은 `2Q + joint base/seed DC + current-only DC`인 `F_h1_h4`였다.

- overall mIoU/F1: `0.5237 / 0.6596`
- cue-mask mean-frame IoU: `0.5176`
- baseline 대비 mIoU/F1: `+0.2360 / +0.2654`

다만 current-only DC는 과거 view 일관성을 직접 평가한 방법이 아니다. 따라서 당장
canonical method로 확정하기보다는, **representation target amplitude는 2로 복원하고
replay는 current-only를 상한 control로 둔 뒤 current lifespan-compatible replay를 다음
기본 후보로 시험**하는 것이 맞다.

Projected seed-local BCE는 unit-Q failure를 완화하지만, `2Q`가 복원된 뒤에는 mIoU를
낮췄다. 최종 기본 loss로 채택하지 않는다.

## 질문과 가설

이번 실험은 다음 다섯 가지를 분리했다.

1. **H0 — autograd/optimizer 단절:** OPEN row가 보여도 FastGS DC gradient나
   `MaskedRowAdam` 경로가 실제 parameter까지 연결되지 않는가?
2. **H1 — cue amplitude mismatch:** 원본 O-SCD SSF에 `[0,1]` cue를 넣어 positive
   force가 global sparsity force를 충분히 이기지 못하는가?
3. **H2 — joint explaining-away:** base와 DA3가 같은 joint render에 들어가 DA3 DC의
   책임과 gradient가 약해지는가?
4. **H3 — coverage/occlusion ceiling:** DC를 아무리 희게 만들어도 현재 OPEN geometry가
   cue pixel을 덮지 못하거나 black NEVER_OPEN이 가리는가?
5. **H4 — replay timestamp mismatch:** 과거 camera/cue를 replay하면서 Gaussian
   population과 DC는 sampled timestamp가 아니라 현재 timestamp의 상태를 사용하여
   서로 다른 시점의 target과 representation을 결합하는가?

## 코드상 직접 원인

기존 경로는 다음 순서다.

1. learned sigmoid 출력은 `candidate_map = 2Q`로 저장된다.
2. `_normalized_cue_target()`이 이를 다시 `cue_scale=2`로 나누어 `Q`를 만든다.
3. `compute_ssf_loss(Q, render)`를 호출한다.

관련 위치:

- unit target 생성: `experiments/view_bayesian_detector_steps.py:1837`
- amplitude를 분리한 새 helper: `experiments/view_bayesian_detector_steps.py:455`
- 실제 DC loss 입력: `experiments/view_bayesian_detector_steps.py:2129`
- DC-only current/sample view 분리: `experiments/view_bayesian_detector_steps.py:2044`
- 새 CLI 축: `experiments/view_bayesian_detector_steps.py:3403`

원본 SSF의 pixel probability와 전체 평균을 각각 `pⱼ`, `μ`라고 두면 render scalar에
대한 gradient 부호는 공통 양의 계수를 제외하고 다음과 같다.

```text
pⱼ = sigmoid(meanRGBⱼ)
μ  = meanⱼ(pⱼ)

gradient sign ∝ 2μ / (1 + μ²) - Qⱼ
```

초기처럼 `μ ≈ 0.5`이면 DC를 white 방향으로 올리려면 대략 `Qⱼ > 0.8`이어야 한다.
unit-Q는 최대값 1에서도 여유가 작지만, 원본 amplitude인 `2Q`는 detection force를 크게
만든다. H1 full run이 이 예측과 일치했다.

## 실험 설계

### 공통 고정값

- 304 frames: `scene_change1` 95 + `scene_change2` 104 + `scene_change3` 105
- frame당 representation update 120
- immutable-reference learned-Q alpha-T BF30 detector
- untouched cached `P+S > 0.5` seed BF30 detector
- causal learned sigmoid cue와 동일 DA3 proposal checkpoint
- base geometry frozen
- DA3 OPEN/NEVER_OPEN geometry 학습과 geometry replay schedule 고정
- representation seed 0
- output threshold `>= 0.5`
- GT는 각 causal step과 representation update가 끝난 뒤에만 평가
- future-view access 0

### DC condition

| ID | DC target | DA3 DC gradient | DC view | 목적 |
|---|---|---|---|---|
| A | `Q` | joint SSF | sampled replay | saved canonical baseline |
| B | `2Q` | joint SSF | sampled replay | H1 only |
| C | `Q` | projected seed-local BCE | sampled replay | H2 practical intervention |
| D | `Q` | joint SSF | current | H4 only |
| E | `2Q` | projected seed-local BCE | sampled replay | H1 + H2 |
| F | `2Q` | joint SSF | current | H1 + H4 |
| G | `2Q` | projected seed-local BCE | current | H1 + H2 + H4 |

`dc_replay_mode=current`는 geometry sampler를 바꾸지 않는다. 같은 sampled view로 DA3
geometry loss를 계산하되 DC loss의 view만 현재 frame으로 바꾼다. 따라서 H4에서
geometry replay 정책 자체를 바꾸는 confound를 피했다.

H3는 별도 학습 condition 대신 매 frame 다음 counterfactual 두 개를 추가 렌더했다.

1. 현재 OPEN을 전부 RGB white로 바꾸고 black NEVER_OPEN occlusion은 유지
2. 현재 OPEN을 전부 RGB white로 바꾸고 NEVER_OPEN opacity를 0으로 만들어 제거

## 전체 결과

### GT ADD union REMOVE

| condition | SC1 mIoU | SC2 mIoU | SC3 mIoU | overall mIoU | F1 | precision | recall | cue IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 0.4218 | 0.3613 | 0.0935 | 0.2877 | 0.3942 | 0.8104 | 0.3560 | 0.2313 |
| B H1 amplitude 2 | 0.4782 | **0.6144** | 0.4534 | 0.5162 | 0.6497 | 0.6536 | 0.8008 | 0.4746 |
| C H2 seed-local | 0.4433 | 0.4439 | 0.3311 | 0.4048 | 0.5417 | 0.6112 | 0.6277 | 0.3811 |
| D H4 current DC | 0.4518 | 0.5648 | 0.3799 | 0.4656 | 0.5968 | **0.8617** | 0.5894 | 0.3742 |
| E H1 + H2 | 0.4487 | 0.5861 | 0.4616 | 0.5002 | 0.6381 | 0.6275 | 0.8161 | 0.4747 |
| F H1 + H4 | **0.4794** | 0.5900 | 0.4982 | **0.5237** | **0.6596** | 0.6313 | 0.8734 | **0.5176** |
| G all | 0.4486 | 0.5694 | **0.5034** | 0.5089 | 0.6476 | 0.6111 | **0.8893** | 0.5172 |

`cue IoU`는 learned score `>=0.5`와 learned Q `>=0.5` 사이의 mean-frame IoU다.

### Cue-following 진단

| condition | Q>=0.8에서 learned mean | Q>=0.8 중 pred>=0.5 | Q<=0.2에서 learned mean | Q MAE | learned/white-ceiling |
|---|---:|---:|---:|---:|---:|
| A | 0.3612 | 0.3008 | 0.0084 | 0.0688 | 0.4239 |
| B | 0.6446 | 0.6823 | 0.0245 | 0.0610 | 0.7567 |
| C | 0.5401 | 0.5406 | 0.0311 | 0.0723 | 0.6340 |
| D | 0.5080 | 0.5000 | **0.0054** | 0.0571 | 0.5962 |
| E | 0.6391 | 0.6978 | 0.0339 | 0.0670 | 0.7500 |
| F | **0.7425** | 0.7561 | 0.0243 | **0.0554** | **0.8714** |
| G | 0.7373 | **0.7720** | 0.0326 | 0.0601 | 0.8654 |

## 가설별 판정

### H0 — joint renderer/autograd 단절: 기각

t=13에서 실제 active DA3 seed parameter 하나에 대해 centered finite difference를
계산했다.

| condition | autograd | finite difference | relative error |
|---|---:|---:|---:|
| A joint unit-Q | `3.2351e-5` | `3.3081e-5` | 2.21% |
| B joint 2Q | `-5.5210e-5` | `-5.5730e-5` | 0.93% |
| C projected local | `-4.0749e-3` | `-4.0746e-3` | 0.0082% |

따라서 saved pipeline이 실제 사용하는 full base+seed joint graph와 optimizer 경로는
연결되어 있다.

단, 14-frame 음성 대조군에서 작은 DA3-only FastGS render로 원본 SSF를 직접 계산하면
autograd와 finite difference가 모두 정확히 0이었다. active seed 43개의 DC는 초기
RGB `0.5`에 그대로 머물렀다. 따라서 **naive seed-only FastGS SSF는 이 build에서
사용하면 안 된다.** H2의 본 실험은 이 때문에 이미 사용 경험이 있는 projected
per-seed BCE로 수행했다.

### H1 — cue amplitude mismatch: 강하게 지지, 1순위 원인

B는 A에서 representation DC target만 `Q -> 2Q`로 바꿨다.

- mIoU: `+0.2285`
- F1: `+0.2555`
- cue IoU: `+0.2433`
- high-Q learned mean: `0.3612 -> 0.6446`
- high-Q positive fraction: `0.3008 -> 0.6823`
- final OPEN seed intrinsic RGB `>=0.5` 비율: `0.0609 -> 0.2365`
- recall: `+0.4448`
- precision: `-0.1568`

Baseline 반복 mIoU 차이는 `0.00046`뿐이었다. H1 효과는 그 약 500배다.

즉 “원래 loss로 돌렸다”는 설명은 식에 대해서는 맞지만, 입력 amplitude까지 원래로
돌아간 것은 아니었다.

### H2 — joint explaining-away: unit-Q에서는 완화되지만 주원인은 아님

C는 base joint forward와 base gradient는 유지하되 seed DC의 joint gradient만 지우고
projected per-seed BCE로 교체했다.

- A -> C mIoU `+0.1170`
- cue IoU `+0.1498`
- seed intrinsic RGB `>=0.5` 비율 `0.0609 -> 0.2405`
- 마지막-update seed gradient 절댓값 평균은 약 89.3배 증가

그러나 이 intervention은 base 부재만 바꾼 순수 실험이 아니다. FastGS seed-only SSF가
0-gradient여서 projected BCE를 사용했으며, loss 좌표와 gradient scale도 함께 달라졌다.
따라서 C는 “seed-local supervision으로 failure를 완화할 수 있다”는 증거이지,
explaining-away만의 고유 효과량은 아니다.

더 중요한 interaction은 다음과 같다.

- B `2Q joint` -> E `2Q + seed-local`: mIoU `-0.0161`, cue IoU `+0.0001`
- F `2Q + current joint` -> G `all`: mIoU `-0.0149`, cue IoU `-0.0004`
- F -> G recall은 `+0.0159`지만 precision은 `-0.0202`
- low-Q learned mean은 F `0.0243`에서 G `0.0326`으로 증가

따라서 amplitude가 복원된 뒤에는 joint gradient 부족이 전체 성능의 blocker가 아니다.
Projected seed-local BCE는 기본값으로 채택하지 않는다.

### H3 — coverage/black occlusion ceiling: 존재하지만 2순위 이하

Baseline에서 `Q>=0.8` pixel을 기준으로 측정했다.

- learned score mean: `0.3612`
- 현재 OPEN을 전부 white로 만든 attainable ceiling: `0.8521`
- NEVER_OPEN까지 제거한 open-only ceiling: `0.9414`
- black NEVER_OPEN 평균 occlusion penalty: `0.0894`
- white로 만들어도 threshold 0.5에 못 미치는 high-Q pixel: `12.17%`
- GT-positive 중 white ceiling이 0.5 미만인 pixel: `5.56%`
- learned mass / white-ceiling mass: `42.39%`

특히 SC3는 learned high-Q mean `0.2265`, white ceiling `0.8917`인데 coverage-limited
high-Q pixel은 `8.12%`뿐이었다. 따라서 현재 DC failure의 대부분은 topology ceiling이
아니라 **도달 가능한 coverage를 DC가 사용하지 못한 것**이다.

Black NEVER_OPEN occlusion은 약 0.089 score만큼의 별도 penalty이므로 후속 개선 여지는
있지만, 이번 DC 저학습의 주원인으로 보기는 어렵다.

### H4 — historical-target/current-lifespan timestamp mismatch: 강하게 지지

여기서 결과를 "과거 replay 자체가 현재 DC를 희석한다"고 일반화하면 안 된다. 올바른
lifespan time-travel replay라면 sampled frame `k`의 camera/cue와 함께 `k`에서 ACTIVE였던
Gaussian population을 렌더해야 한다. Gaussian parameter 자체는 여러 causal view가
공동 최적화하는 row-level shared parameter를 그대로 사용한다. 그러나 당시 saved viewer는
sampled item이 `k`여도 `_train_representation_update(..., current_timestamp=t)`를 호출한다.
Base는 현재 `lifecycle.current_state_index`, DA3는 `active_mask(t)`, concatenated renderer도
`timestamp=t`를 사용한다. 따라서 실제 비교는 다음과 같다.

```text
camera/cue = historical k
OPEN population/DC = current t
```

즉 H4 condition은 generic replay ablation이 아니라 이 **timestamp mismatch를 current DC
target으로 제거한 alignment control**이다. Proper historical lifespan-aware replay가 나쁘다는
증거가 아니다.

Baseline DC update 중 실제 current frame을 사용한 비율은 `34.44%`였고, DC view의
평균 age는 `50.83 frames`였다. SC3만 보면 평균 age가 `84.72 frames`였다.

D는 geometry replay를 그대로 두고 DC view만 current로 바꿨다.

- mIoU `+0.1779`
- F1 `+0.2026`
- precision `+0.0513`
- recall `+0.2334`
- cue IoU `+0.1429`
- low-Q learned mean `0.0084 -> 0.0054`

H1을 이미 적용한 뒤에도 B -> F는 다음을 보였다.

- mIoU `+0.0075`
- cue IoU `+0.0430`
- high-Q learned mean `+0.0978`
- recall `+0.0726`

F의 mIoU 증가는 baseline repeat noise의 약 16.4배다. late evolving segment인 SC3는
B `0.4534 ->` F `0.4982`로 개선됐다. 반면 SC2는 `0.6144 -> 0.5900`으로 하락했다.
현재 frame 적합도와 기존의 mismatched historical replay 사이 차이를 보여준다.

이 후속 항목은 2026-09-06에 구현·검증했다. Sampled timestamp의 base/seed interval
population과 optimizer mask를 복원한 B는 overall mIoU `0.5249`, SC3 `0.4925`를 기록했고
historical update `23,916`회에서 future/lifecycle render violation은 0이었다. 즉 F가
보여준 이점의 상당 부분은 historical replay 제거 자체가 아니라 timestamp mismatch
제거에서 왔다.

## 재현성과 통제 감사

- immutable reference hash: 모든 full condition에서 동일
- base detector/lifecycle hash: 모든 full condition에서 동일
- base OPEN/CLOSE: 모든 condition에서 정확히 `80,742 / 22,062`
- future-view access: 전 condition 0
- main 7-condition runtime: 4,106.05 s
- baseline repeat 포함 runtime: 4,665.84 s
- 14-frame negative control 포함 총 runtime: 약 78.07 min
- main representation updates: `304 * 120 * 7 = 255,360`

Dynamic DA3 coverage suppression은 custom CUDA 비결정성 때문에 bitwise 동일하지 않았다.

- accepted seed: main condition에서 `5,494 .. 5,513`
- exact baseline repeat: `5,521`
- baseline 대 repeat accepted-source Jaccard: `0.9858`
- intervention 대 baseline Jaccard 범위: `0.9826 .. 0.9910`
- baseline repeat mIoU/F1 차이: `-0.00046 / -0.00066`

즉 dynamic topology의 실행별 미세 차이는 실제로 존재한다. 그러나 H1/H2/H4 단독 mIoU
효과는 baseline-repeat noise의 각각 약 `500x / 256x / 389x`여서 결론의 크기와 방향을
설명할 수 없다. E/F/G의 작은 interaction은 point estimate로 해석하되, 후속 채택 전
반복 seed 검증이 필요하다.

## 권고

### 2026-09-06 현재 유지할 것

- Detector input은 계속 unit learned-Q로 유지한다.
- Immutable-reference pre-optimization evidence 계약을 유지한다.
- DA3 seed detector와 geometry target도 이번 amplitude ablation에서 바꾸지 않는다.
- Canonical `run_bayesian_detector_viewer.sh`는 `2Q`와 lifespan-correct sampled replay를
  명시적으로 고정한다.

### representation DC 기준

현재 기본 method와 current-only control은 다음과 같다.

```bash
# 기본 method: 2Q + historical lifespan replay
./run_bayesian_detector_viewer.sh

# current-view-only ablation
./run_bayesian_detector_viewer.sh \
  --dc-replay-mode current
```

기본 연구 method는 수정된 H1 historical replay로 확정한다. F는 historical
multiview consistency를 제거한 current-only ablation으로만 남긴다. 필요하면 다음
sampling 정책을 이 올바른 lifespan render 위에서 비교한다: 최근 K-view window,
current-view 비중, 같은 interval 내부 sampling, cue sign/visibility 호환 sampling.

Projected seed-local BCE는 H1/F 위에 추가하지 않는다. H3 개선도 DC amplitude와 replay를
고친 뒤 남는 5--12% coverage-limited 영역에 대해 별도로 다룬다.

## 산출물

- Full per-condition output:
  `outputs/bayesian_da3_dc_hypotheses_20260905/`
- Machine summary:
  `outputs/bayesian_da3_dc_hypotheses_20260905/all_conditions_summary.json`
- Condition table:
  `outputs/bayesian_da3_dc_hypotheses_20260905/condition_metrics.csv`
- Per-frame CSV, diagnostic images, controlled state:
  각 condition directory의 `frame_metrics.csv`, `diagnostics/`, `controlled_state.pt`
- Baseline repeat:
  `outputs/bayesian_da3_dc_hypotheses_20260905/R_baseline_repeat/`
- Naive seed-only renderer negative control:
  `outputs/bayesian_da3_dc_hypotheses_20260905/N_seed_only_ssf_renderer_control/`
- Runner:
  `experiments/evaluate_bayesian_da3_dc_hypotheses.py`
- Summarizer:
  `experiments/summarize_bayesian_da3_dc_hypotheses.py`
- Suite preset:
  `run_bayesian_da3_dc_hypotheses.sh`
- Corrected full B output:
  `outputs/bayesian_da3_historical_replay_20260906/B_h1_amplitude2/`

현재 코드로 corrected suite 실행:

```bash
./run_bayesian_da3_dc_hypotheses.sh
```

기본 output은 `outputs/bayesian_da3_historical_replay_20260906/`이며 sampled condition은
모두 lifespan-correct semantics를 사용한다. 따라서 위 명령은 2026-09-05의 mismatched
artifact를 덮어쓰지 않는다. 본 condition만 실행하려면 `RUN_CONTROLS=0`을 지정한다.

단일 condition smoke:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/rvl/miniforge3/envs/oscd/bin/python \
  -m experiments.evaluate_bayesian_da3_dc_hypotheses \
  --condition B_h1_amplitude2 \
  --max-frames 14 \
  --updates-per-frame 120 \
  --output-root outputs/bayesian_da3_historical_replay_smoke_20260906
```
