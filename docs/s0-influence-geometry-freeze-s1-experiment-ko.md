# S0 influence-valid geometry 영구 동결 후 S1 학습 실험

## 질문

Shared geometry forgetting run의 State 0이 끝난 시점에서 실제 change render에
signed influence를 갖는 Gaussian의 geometry를 영구 동결하면, 나머지 Gaussian이
State 1을 학습하면서 S0 retention을 보존할 수 있는가?

이 실험은 동적 online locking을 다루지 않는다. 정확히 동일한 S0 boundary
checkpoint에서 influence를 한 번 계산하고, S1 동안 그 row만 고정하는 격리
ablation이다.

## 설정

- 시작 checkpoint:
  `instance1_scene_change1_2_3_temporal_shared_geometry_forgetting_oscd_cues_allframes_120/state0_complete_checkpoint.pt`
- supervision: O-SCD pixel cue + SAM2.1 feature cue
- fixed pose, manual boundary `[95, 199]`
- S1 frame: 104개
- frame당 update: 120회, 총 12,480 update
- topology 고정, densification/pruning/replay 없음
- S0 DC와 S1 이외 DC slot은 수정하지 않음
- S1 DC는 기존과 동일하게 학습

S0 boundary checkpoint의 95개 view에서 기존 signed-opacity influence 판정을
사용했다.

```text
soft_mask = sigmoid((gray - 0.5) / 0.05)
influence = candidate_opacity * d(sum(soft_mask)) / d(effective_opacity)

min per-view |influence| = 1e-6
min mean additive or occluding influence = 1e-4
min contributing views = 1
```

그 결과는 다음과 같다.

| 구분 | GS |
|---|---:|
| 전체 influence-valid, 영구 동결 | 14,477 |
| additive | 9,211 |
| occluding | 6,484 |
| additive와 occluding 모두 | 1,218 |
| S1 `state_valid`와 동결 집합의 교집합 | 12,369 |
| S1에서 움직일 수 있는 `state_valid` GS | 673,163 |

동결 집합은 전체 1,283,501개 Gaussian의 약 `1.13%`다.

## 구현 계약

동결 row는 S0가 최초 소유하며 별도 state anchor를 만들지 않는다.

```text
owner_state[i] = 0
anchor[i] = geometry_after_S0[i]

for every S1 update:
    grad(shared_geometry[i]) = 0
    optimizer.step()
    shared_geometry[i] = anchor[i]
```

고정한 속성은 `xyz`, `opacity`, `scaling`, `rotation` 전부다. Gradient mask 뒤
anchor projection도 수행해 optimizer 동작과 무관하게 equality constraint를
강제했다.

감사 결과:

- S0 DC max drift: `0`
- frozen xyz max drift: `0`
- frozen opacity max drift: `0`
- frozen scaling max drift: `0`
- frozen rotation max drift: `0`
- 실제 mask 전에 관측된 frozen-row gradient는 네 속성 모두 nonzero였으므로,
  동결 조건이 단순히 원래 gradient가 없던 row를 고른 것은 아니다.

## 결과

GT는 학습에 사용하지 않고 아래 confusion 평가에만 사용했다.

### Mean per-frame IoU

| checkpoint | S0 re-render | S1 re-render |
|---|---:|---:|
| S0 boundary | **0.5829** | - |
| S1 학습, unfrozen shared geometry | **0.4561** | **0.6404** |
| S1 학습, S0 influence geometry frozen | **0.3160** | **0.5443** |

### Mean per-frame F1

| checkpoint | S0 re-render | S1 re-render |
|---|---:|---:|
| S0 boundary | **0.6763** | - |
| S1 학습, unfrozen shared geometry | **0.5794** | **0.7674** |
| S1 학습, S0 influence geometry frozen | **0.4446** | **0.6748** |

IoU delta:

- 기존 unfrozen S0 forgetting: `-0.1268`
- influence-freeze S0 forgetting: `-0.2669`
- unfrozen 대비 S0 retention: `-0.1401`
- unfrozen 대비 S1 plasticity: `-0.0961`

따라서 이 hard-freeze 정책은 S0 retention과 S1 성능을 모두 악화시켰다.

## Geometry 이동 감사

동결 row는 정확히 움직이지 않았지만, S1-valid이면서 동결되지 않은 row는 다음과
같이 이동했다.

| 속성 | mean | p95 | max |
|---|---:|---:|---:|
| xyz | 0.00763 | 0.03383 | 0.58996 |
| opacity raw delta | 0.77059 | 4.13841 | 22.24588 |
| scaling raw delta | 0.31440 | 1.54954 | 14.79274 |
| rotation raw delta | 0.05814 | 0.27695 | 2.13553 |

## 후속 구현: GS의 persistent `geometry_frozen` 속성

외부 `frozen_mask`와 anchor projection 대신 shared temporal model 자체에 다음
checkpoint buffer를 추가했다.

```text
geometry_frozen[N] bool
```

S0 influence 판정 직후 `geometry_frozen |= influence_valid`를 수행하고, S1의 매
backward 뒤 네 shared geometry tensor에 다음 조건만 적용했다.

```text
grad[geometry_frozen] = 0
```

- 별도 geometry anchor tensor 없음
- optimizer 이후 projection 없음
- S1 boundary에서 Adam을 새로 생성
- weight decay 없음
- S1 DC gradient는 차단하지 않음
- legacy shared checkpoint에는 loader가 all-false buffer를 주입

새 checkpoint에는 `geometry_frozen=True`가 정확히 14,477개 저장됐으며 S1 학습 후
해당 row의 `xyz/opacity/scaling/rotation` drift는 모두 `0`이었다.

### 구현별 결과

| 조건 | S0 mean frame IoU | S1 mean frame IoU |
|---|---:|---:|
| S0 boundary | 0.5829 | - |
| unfrozen shared geometry | 0.4561 | 0.6404 |
| 외부 mask + anchor projection | 0.3160 | 0.5443 |
| model `geometry_frozen` + gradient mask | **0.2771** | **0.5291** |

두 freeze run은 동일한 14,477개 mask를 사용했고 frozen drift도 모두 정확히 0이다.
Model-buffer run은 외부 projection run보다 S0 `-0.0389`, S1 `-0.0152` 낮았다.

다만 이는 buffer 저장 방식 자체의 효과로 해석하면 안 된다. 동일 gradient sequence를
두 구현에 입력하는 Adam 회귀 테스트에서는 최종 geometry parameter가 bitwise
동일했다. 실제 FastGS CUDA 전체 학습은 rasterizer backward의 atomic 연산 때문에
별도 run 사이에서 bitwise deterministic하지 않고, 작은 초기 차이가 장기 최적화에서
다른 free-GS 해로 증폭될 수 있다. 중요한 공통 결론은 두 run 모두 unfrozen
baseline보다 retention과 plasticity가 낮았다는 점이다.

## 동일 S0 checkpoint의 전체 geometry freeze 대조군

원본 DC-only와 partial freeze는 출발 geometry가 다르므로, 동일한 shared-geometry
S0 checkpoint에서 다음 대조군을 추가했다.

```text
source: shared geometry after S0
geometry_frozen: 1,283,501 / 1,283,501
optimizer: S1 state_change_dc only
schedule: 104 frames × 120 updates = 12,480
```

S1 학습 뒤 shared xyz/opacity/scaling/rotation과 S0 DC는 source checkpoint와
bitwise 동일했다.

| 조건 | S0 mean frame IoU | S1 mean frame IoU |
|---|---:|---:|
| S0 boundary | 0.5829 | - |
| 동일 S0: geometry 전체 고정, S1 DC-only | **0.5829** | 0.4250 |
| 동일 S0: influence 14,477개 고정 | 0.2771 | 0.5291 |
| 동일 S0: geometry 전체 unfrozen | 0.4561 | **0.6404** |
| 원본 base geometry 고정 DC-only | 0.6137 | 0.6464 |

동일 S0 checkpoint 안에서는 예상대로 trade-off가 확인된다.

- 전체 고정은 S0를 정확히 보존하지만 S1 표현력이 가장 낮다.
- partial freeze는 전체 고정보다 S1을 `+0.1041` 개선하지만 S0를 `-0.3057`
  악화시킨다.
- unfrozen은 전체 고정보다 S1을 `+0.2154` 개선하지만 S0를 `-0.1268`
  악화시킨다.

원본 DC-only가 더 좋은 이유는 같은 제약의 결과가 아니라 **geometry basis 자체가
다르기 때문**이다. 원본 DC-only는 reference base geometry를 처음부터 고정하지만,
이 대조군은 S0 cue에 맞춰 이미 변형된 shared geometry를 S1에서도 그대로 사용한다.

### 두 valid 정의

이 실험에서 혼동하기 쉬운 두 집합은 서로 다른 의미다.

| 집합 | S0 | S1 | S0∩S1 |
|---|---:|---:|---:|
| `state_valid`: cue support 내부 누적 raster contribution ≥ 1 | 576,372 | 685,532 | 542,045 |
| S0 signed-influence valid: soft mask 면적에 대한 평균 절대 influence ≥ `1e-4` | 14,477 | - | support 교집합 내 12,369 |

따라서 partial-freeze 실험은 S0·S1 support-valid 교집합 542,045개를 모두 잠근
것이 아니다. 그중 현재 S0 mask에 유의한 signed removal/insertion effect가 있었던
12,369개를 포함해 전체 14,477개만 잠갔다. 나머지 support-valid GS는 이후 state에서
geometry가 계속 학습된다.

## 결론

> 과거에 influence가 있던 GS의 geometry drift만 막는 것으로는 forgetting을 막을
> 수 없다.

실패 원인은 두 가지로 분리된다.

1. S0 시점에 influence-invalid였던 free GS도 S1에서 이동하거나 커지면 S0 view에
   새 influence를 획득할 수 있다. Static S0 influence mask는 이 새 간섭을 막지
   않는다.
2. S0 influence-valid GS는 공간적으로 중요한 capacity다. 이를 완전히 막으면 S1이
   같은 영역을 설명할 때 사용할 수 있는 geometry가 줄어 current-state 적합도도
   떨어진다.

즉 이 결과는 동결 구현 실패가 아니다. 동결 invariant는 정확히 만족했지만,
**row-level static ownership 자체가 output retention을 보장하지 못한 것**이다.

동일 S0 checkpoint 전체-freeze 대조군과 원본 base-geometry DC-only 결과까지
종합한 현재 연구 결정은 다음과 같다.

> Reference 3DGS의 `xyz/opacity/scale/rotation`은 시간에 따라 덮어쓰지 않고
> immutable spatial basis로 유지한다. 현재 변화 상태는 state별 DC, lifespan,
> validity/confidence로 표현한다.

원본 base geometry DC-only가 S0/S1 모두에서 가장 안정적이었고, S0 cue에 맞춰
한 번 변형된 shared geometry를 이후 고정해도 S1 DC-only 성능이 `0.4250`으로
낮았다. 따라서 현재 단계에서는 dynamic geometry optimization을 기본 경로에서
제외한다. 향후 fixed reference geometry로 표현할 수 없는 위치가 확인될 경우에도
reference를 수정하기보다 별도의 state-specific residual/new-GS 표현으로 분리해
검토하며, 이는 이 실험 checkpoint의 범위에 포함하지 않는다.

## 산출물

```text
outputs/instance1_scene_change1_2_3_temporal_shared_geometry_s0_influence_freeze_s1_oscd_cues_allframes_120/
  state0_complete_checkpoint.pt -> 기존 S0 checkpoint
  state1_complete_checkpoint.pt
  temporal_rchange_checkpoint.pt -> state1_complete_checkpoint.pt
  summary.json
  influence_freeze_comparison.json
  signed_influence/
    state0_signed_influence.pt
    state0_signed_influence_summary.json
    state0_multiview_valid_render/
  confusion_after_s1/
  freeze_vs_unfrozen_after_s1/
  s0_boundary_vs_freeze_after_s1/

outputs/instance1_scene_change1_2_3_temporal_shared_geometry_s0_geometry_frozen_attr_s1_oscd_cues_allframes_120/
  state1_complete_checkpoint.pt
  summary.json
  influence_freeze_comparison.json
  geometry_freeze_implementation_comparison.json
  confusion_after_s1/
  model_frozen_vs_unfrozen_after_s1/

outputs/instance1_scene_change1_2_3_temporal_shared_geometry_s0_all_geometry_frozen_s1_dc_only_oscd_cues_allframes_120/
  state1_complete_checkpoint.pt
  summary.json
  confusion_after_s1/
  s0_geometry_freeze_scope_comparison.json
  all_frozen_vs_partial_after_s1/
```

대표 GIF:

```text
```

## 재현

```bash
PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.export_state_signed_influence \
  --checkpoint outputs/instance1_scene_change1_2_3_temporal_shared_geometry_forgetting_oscd_cues_allframes_120/state0_complete_checkpoint.pt \
  --summary outputs/instance1_scene_change1_2_3_temporal_shared_geometry_forgetting_oscd_cues_allframes_120/summary.json \
  --output-dir outputs/instance1_scene_change1_2_3_temporal_shared_geometry_s0_influence_freeze_s1_oscd_cues_allframes_120/signed_influence \
  --state 0

PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.train_s0_influence_freeze_s1

PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.render_temporal_confusion_maps \
  --run-dir outputs/instance1_scene_change1_2_3_temporal_shared_geometry_s0_influence_freeze_s1_oscd_cues_allframes_120 \
  --checkpoint-path outputs/instance1_scene_change1_2_3_temporal_shared_geometry_s0_influence_freeze_s1_oscd_cues_allframes_120/state1_complete_checkpoint.pt \
  --output-dir outputs/instance1_scene_change1_2_3_temporal_shared_geometry_s0_influence_freeze_s1_oscd_cues_allframes_120/confusion_after_s1 \
  --max-frames 199 --overwrite

PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.summarize_s0_influence_freeze_s1

PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.train_s0_geometry_frozen_attr_s1

PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.compare_geometry_freeze_implementations

PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.train_s0_all_geometry_frozen_s1_dc_only

PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.compare_s0_geometry_freeze_scopes
```
