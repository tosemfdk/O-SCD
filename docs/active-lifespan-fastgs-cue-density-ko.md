# Active lifespan + FastGS-style change-cue density 실험

## 1. 질문

이번 실험은 독립적인 `ref -> SC1`, `ref -> SC2`, `ref -> SC3` 스트림에서 다음 결합이 실제 성능을 높이는지 확인한다.

```text
immutable R_ref alpha-T evidence
    -> direct binary Bayesian state filter
    -> view-consistent K3/BF3 OPEN/CLOSE
    -> 현재 OPEN이며 현재 view에 보이는 row-slot만 all-geometry 학습
    -> K=10 causal change-cue VCD/VCP
    -> 다음 frame
```

Detector는 binary cue를 사용하고 density controller는 이전 실험에서 정한 soft raw ref--inference cue를 사용한다. 두 경로는 분리되어 있다.

## 2. 고정 설정

- Dataset: `data/Instance_1/scene_change1_2_3`
- Scope: SC1 95 frames, SC2 104 frames, SC3 105 frames를 각각 reference부터 독립 실행
- Seed: 0
- Resolution: 4
- Updates/frame: 120
- Detector cue: `candidate_map > 0.5`
- Detector evidence: immutable reference alpha-T, capped count
- Filter: direct two-state Bayesian filter
- Lifecycle controller: view-consistent `K=3`, minimum Bayes factor `3`
- Thaw: `dc,xyz,opacity,scaling,rotation`
- Optimizer mask: `current OPEN row-slot AND current render radius > 0`
- Density view bank: current view + 이미 처리된 causal view 중 deterministic-random sample, 최대 10개
- Density update: 각 frame의 local update index 4
- GT: 전체 causal loop가 끝난 뒤 평가에만 사용

Baseline과 density 조건은 detector, cue cache, camera, seed, optimizer schedule이 같다. 실제 `Gaussian index / timestamp / action / old slot / new slot` lifecycle event sequence도 세 scene 모두 완전히 일치했다. Posterior 실수 값의 최대 차이는 CUDA atomic reduction 수준인 약 `7e-7`이고 어떤 lifecycle action도 바뀌지 않았다.

## 3. Topology-safe 결합

`TemporalGeometryChangeModel`은 본래 고정 `[N,S,...]` 구조이므로 immutable reference 자체를 FastGS 함수로 resize하면 detector identity와 temporal state alignment가 깨진다.

이번 구현은 다음 두 topology를 분리한다.

```text
Detector reference prefix
    N = 1,283,501
    영구 고정
    alpha-T Bayesian evidence 전용

Temporal render anchor
    동일한 immutable prefix N
    + active episode에서 생성된 residual child rows
```

Residual child는 다음 lineage를 가진다.

```text
stable_id
parent_stable_id
immutable root Gaussian index
episode slot
creation timestamp
generation
```

Child의 lifespan은 생성 당시 root의 OPEN slot에 고정된다. Root slot이 CLOSE되면 같은 episode의 모든 child도 같은 decision timestamp에 CLOSE된다. Root가 나중에 새 slot으로 REOPEN되어도 이전 child는 다시 열리지 않는다.

Pruning 가능한 대상은 **현재 OPEN residual child**뿐이다.

- immutable reference-prefix row: 절대 prune하지 않음
- CLOSED historical row: 절대 prune하지 않음
- 현재 OPEN residual row: cue-consistency 조건을 만족할 때만 prune 가능

## 4. K=10 change-cue score

현재 active topology를 과거 causal view `j`에 투영하고 soft cue `C_j(p)`로 alpha-T VJP를 계산한다.

\[
e^+_{i,j}=\sum_p \alpha_i(p)T_i(p)C_j(p)
\]

\[
e^-_{i,j}=\sum_p \alpha_i(p)T_i(p)(1-C_j(p))
\]

현재 episode가 시작되기 전 view는 그 row의 score에 포함하지 않는다. FastGS-style importance는 다음과 같다.

\[
S_i^{change}=\frac{1}{K}\sum_{j=1}^{K}e^+_{i,j}
\]

Densification은 다음 교집합만 허용한다.

```text
current OPEN
AND at least 3 observed sampled views
AND S_change > 5
AND FastGS position-gradient qualifier
```

Small row는 clone, large row는 absolute-gradient split 후보이다. 다만 immutable prefix split source는 제거할 수 없으므로 prefix를 유지하고 residual child 두 개를 추가한다. Residual split source는 FastGS처럼 child 두 개로 교체한다.

## 5. Change-cue VCP

각 sampled view에서

\[
q_{i,j}=\frac{e^+_{i,j}}{e^+_{i,j}+e^-_{i,j}+\epsilon}
\]

를 계산하고 `q >= 0.5`인 observed view를 supporting view로 센다.

Residual pruning 조건은 다음과 같다.

```text
current OPEN residual
AND age >= 10 frames
AND visible in at least 8 sampled views
AND supporting views <= 2
```

즉 change score가 높아서 prune하는 FastGS original VCP 방향을 그대로 사용하지 않는다. 여러 view에서 실제로 보였지만 change cue와 일치한 view가 소수인 residual만 제거한다.

## 6. 결과

### 6.1 Primary metrics

| Scope | Condition | mIoU | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| SC1 | all-geometry baseline | 0.5555 | 0.6538 | 0.7555 | 0.9146 |
| SC1 | + K10 VCD/VCP | **0.5623** | **0.6580** | 0.7536 | 0.9147 |
| SC2 | all-geometry baseline | **0.6144** | **0.7433** | 0.6476 | 0.9430 |
| SC2 | + K10 VCD/VCP | 0.6141 | 0.7433 | 0.6496 | 0.9396 |
| SC3 | all-geometry baseline | **0.5919** | **0.7253** | 0.6816 | 0.8748 |
| SC3 | + K10 VCD/VCP | 0.5886 | 0.7231 | 0.6839 | 0.8649 |

Frame-count weighted result:

| Metric | Baseline | K10 density | Delta |
|---|---:|---:|---:|
| mIoU | 0.5882 | 0.5891 | +0.0009 |
| F1 | 0.7091 | 0.7096 | +0.0005 |

SC1은 개선됐지만 SC2는 사실상 동률이고 SC3는 하락했다. 따라서 seed 0 기준으로 density integration이 일관된 성능 향상을 만들었다고 결론낼 수 없다.

### 6.2 Topology

| Scope | Split children | Clone children | VCP pruned | Split-source removed | Final residual |
|---|---:|---:|---:|---:|---:|
| SC1 | 400 | 0 | 41 | 42 | 317 |
| SC2 | 884 | 0 | 42 | 74 | 768 |
| SC3 | 842 | 0 | 22 | 102 | 718 |
| 합계 | 2,126 | 0 | 105 | 218 | 1,803 |

모든 densification은 split이었다. 현재 `dense_fraction * scene_extent` 아래에 들어가는 중요 gradient candidate가 없어서 clone은 한 번도 발생하지 않았다. 최종 residual 증가는 원래 128만 Gaussian의 약 `0.025%--0.060%` 수준이다.

VCP는 실제로 동작했지만 2,126 split child에 비해 105개만 제거했다. 현재 조건에서는 pruning보다 split에 의한 capacity 증가가 지배적이다.

### 6.3 Lifecycle

Density on/off lifecycle action은 동일하다.

| Scope | OPEN | CLOSE | REOPEN | same-scene repeated transition events |
|---|---:|---:|---:|---:|
| SC1 | 33,246 | 5,078 | 1,403 | 6,481 |
| SC2 | 66,995 | 13,767 | 2,748 | 16,515 |
| SC3 | 67,948 | 20,762 | 3,937 | 24,699 |

독립 one-step scene인데도 CLOSE/REOPEN이 많다. Density controller는 representation capacity만 바꾸며 이 detector chattering을 해결하지 않는다. 오히려 child가 root episode에 묶이므로 root가 false CLOSE되면 유용한 residual도 함께 닫힌다.

### 6.4 Runtime / memory

| Scope | Baseline runtime | K10 runtime | Overhead | Baseline peak | K10 peak |
|---|---:|---:|---:|---:|---:|
| SC1 | 277.1 s | 290.1 s | +4.7% | 7.72 GiB | 9.93 GiB |
| SC2 | 304.7 s | 323.2 s | +6.1% | 7.81 GiB | 10.02 GiB |
| SC3 | 315.0 s | 331.6 s | +5.3% | 7.84 GiB | 10.05 GiB |

K=10 VJP와 dynamic topology가 약 5% runtime, 약 2.2 GiB peak memory를 추가했다.

## 7. 해석

1. **Detector와 representation의 분리는 성공했다.** Density on/off에서 lifecycle action sequence가 동일하므로 mIoU 차이는 detector 변경이 아니라 residual representation 차이이다.
2. **SC1에서는 특정 frame에서 split child가 under-represented footprint를 보완했다.** SC1은 46 frames에서 개선, 32 frames에서 하락했고 최대 단일-frame 개선은 `+0.1539`였다.
3. **SC3에서는 precision이 조금 오르고 recall이 크게 하락했다.** Split이 큰 Gaussian support를 더 작은 child로 분산시키면서 일부 view의 coverage가 줄어든 것으로 해석할 수 있다. 이는 source evidence가 아니라 결과에 근거한 추론이다.
4. **현재 정책은 사실상 split-only ablation이다.** Clone이 0이므로 “새 capacity 추가”와 함께 scale 축소 및 source replacement 효과가 섞여 있다.
5. **Pruning은 아직 약하다.** VCP가 제거한 row가 적어서 one-view spike 제거 효과를 primary metric에서 분리하기 어렵다.
6. **가장 큰 blocker는 여전히 detector chattering이다.** Stable one-step scope에서도 수천 REOPEN이 발생한다. Density tuning 전에 false CLOSE/REOPEN을 줄이지 않으면 child lifespan도 불안정하다.

## 8. 무결성 audit

세 baseline과 세 density run 모두 다음을 만족했다.

```text
future density view access                 = 0
GT/manual boundary use in causal loop      = 0
immutable detector-reference max drift     = 0
render-anchor immutable prefix max drift   = 0
CLOSED prefix slot max drift               = 0
CLOSED residual slot max drift             = 0
inactive/current-view-invisible grad errors = 0
reused lifespan slot violations            = 0
active->active false split                  = 0
```

## 9. 실행 명령

Baseline 예:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_binary_lifespan_active_density \
  --condition baseline \
  --scope scene_change1 \
  --updates-per-frame 120 \
  --output-dir outputs/escd_independent_binary_active_density_u120_seed0_20260825/baseline/scene_change1 \
  --skip-checkpoint
```

K=10 density 예:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_binary_lifespan_active_density \
  --condition active_cue_vcd_vcp \
  --scope scene_change1 \
  --updates-per-frame 120 \
  --output-dir outputs/escd_independent_binary_active_density_u120_seed0_20260825/active_cue_vcd_vcp/scene_change1 \
  --skip-checkpoint
```

## 10. 출력

```text
outputs/escd_independent_binary_active_density_u120_seed0_20260825/
  comparison.json
  comparison.md
  comparison_timelines.png
  baseline/{scene_change1,scene_change2,scene_change3}/
  active_cue_vcd_vcp/{scene_change1,scene_change2,scene_change3}/
```

각 run은 `summary.json`, `frame_metrics.csv`, `lifecycle_events.jsonl`, `density_events.jsonl`, `residual_lineage.jsonl`을 가진다. 생성 output은 Git에 포함하지 않는다.

## 11. 제한 및 다음 판단

- 이번 결과는 seed 0 한 번이며 K ablation과 multi-seed repeat를 하지 않았다.
- Independent ref-to-scene만 실행했고 `ref -> SC1 -> SC2 -> SC3` continuous integration은 아직 실행하지 않았다.
- Prefix split source를 제거할 수 없으므로 FastGS original split과 정확히 같지 않다.
- New-object 위치에 reference support가 전혀 없으면 reference-root child density만으로는 해결할 수 없다.

현재 결과만 보면 다음 우선순위가 타당하다.

```text
false CLOSE/REOPEN 감소
-> clone/split threshold를 분리해 split-only 현상 해소
-> VCP support threshold/grace ablation
-> 동일 detector decision을 고정한 multi-seed repeat
-> 그 뒤 continuous evolving stream
```
