# ref → SC1 FastGS-style soft alpha-T change-cue density control

## 1. 실험 질문

FastGS의 multi-view consistent densification(VCD)을 `R_change`에 적용하되,
RGB reconstruction error 대신 O-SCD가 이미 계산한 ref--inference change cue를
사용했다.

이번 checkpoint의 범위는 다음으로 제한한다.

- 데이터: `Instance_1/scene_change1_2_3`의 `scene_change1` 95 frames
- 변화 흐름: `ref → SC1`만 사용
- cue: cached O-SCD pixel + SAM ref--inf cue의 **연속값 [0,1]**
- topology: temporal lifespan sidecar가 아닌 별도의 mutable `R_change` bank
- density operation: clone/split만 사용
- cue-based pruning: 사용하지 않음
- GT: 95 frames의 causal 처리가 전부 끝난 뒤 평가에만 사용

## 2. FastGS 공식 구현 확인

확인한 upstream은 [FastGS repository](https://github.com/fastgs/FastGS)의
commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`이다.

확인 파일:

- `utils/fast_utils.py`
- `train.py`
- `scene/gaussian_model.py`
- `gaussian_renderer/__init__.py`
- `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu`

공식 구현은 K개의 training view에서 normalized RGB L1 map을 threshold하여
integer `metric_map`을 만들고, rasterizer가 high-error pixel에 기여한
Gaussian별 count를 누적한다. 이 count를 `sum / K` 후 floor한 importance와
position gradient qualifier를 **AND**하여 작은 Gaussian은 clone하고 큰
Gaussian은 absolute-gradient로 split한다. 즉 VCD가 gradient 조건을
대체하지 않는다.

## 3. O-SCD에 이미 있던 FastGS port

기존 저장소에는 다음 구현이 이미 있었다.

- `gaussian_renderer.render(..., get_flag, metric_map)`과
  `accum_metric_counts`
- `scene/gaussian_model.py`의 FastGS clone/split/prune와 optimizer-safe
  tensor resize
- `utils/fast_utils.py`의 RGB L1 기반 multi-view score
- `update.py` selective reconstruction의
  `compute_gaussian_score_fastgs()` → `densify_and_prune_fastgs()`

그러나 기존 `update.py` 경로는 offline reconstruction update이고, random
training RGB view의 L1 error를 사용한다. 온라인 `oscd.py`의 inference
`R_change`는 frame당 update 4에서 vanilla gradient-only clone/split을
사용한다. 기존 `compute_gaussian_score_fastgs_pixel_changed()`도 raw
candidate cue가 아니라 이미 생성한 eroded/dilated change mask를 사용한다.

따라서 FastGS topology 코드는 재구현하지 않고, **online causal raw cue
score와 FastGS gradient/importance 교집합만 새로 연결**했다.

## 4. Soft alpha-T score

FastGS CUDA `metric_map`은 integer map이므로 `0.2`, `0.7` 같은 cue 값을
그대로 누적할 수 없다. 이번 실험은 기존 `change_evidence.py`의 color-probe
VJP를 재사용했다.

Gaussian `i`, sampled causal view `j`, pixel `p`에 대해:

\[
e^+_{i,j}=\sum_p \alpha_{i,j}(p)T_{i,j}(p)C_j(p)
\]

\[
e^-_{i,j}=\sum_p \alpha_{i,j}(p)T_{i,j}(p)(1-C_j(p))
\]

그리고 densification importance는 FastGS의 multi-view `sum / K` 형태로:

\[
S_i^{change}=\frac{1}{K}\sum_{j\in V_t}e^+_{i,j}
\]

로 정의했다. `candidate_map > 0.5`를 적용하지 않는다. `e^-`와
`e^+/(e^++e^-)`는 diagnostic으로만 저장하고 첫 densification gate에는
사용하지 않는다.

### Causal view sampling

timestamp `t`에서:

```text
current view t + processed past views 중 random K-1개
```

를 사용한다. 아직 K개가 없으면 가능한 view를 전부 사용한다. K=1은
정확히 current-view alpha-T score이고, `view index > t` 접근은 모든 실행에서
0이었다. Density sampling은 local RNG를 사용해 training-view schedule의
random state를 바꾸지 않는다.

## 5. Densification contract

기본 조건은 FastGS와 같은 구조다.

```text
clone = small GS
        AND regular xyz gradient >= 2e-4
        AND soft change importance > 5

split = large GS
        AND absolute xyz gradient >= 1.2e-3
        AND soft change importance > 5
```

- frame당 updates: 16
- density decision: local update 4
- evaluation mask threshold: 0.5
- seed: 0, 1, 2 반복
- densification 후 parameter/Adam moment/gradient accumulator/radius 길이를
  모두 audit
- immutable reference PLY SHA-256를 실행 전후 비교

세 조건을 구분했다.

1. `baseline`: 기존 online O-SCD gradient-only clone/split
2. `fastgs_gradient_only`: FastGS gradient/size/topology 조건은 쓰되 importance를
   항상 통과시킨 matched control
3. `cue_vcd_k*`: 동일 FastGS 조건에 soft alpha-T cue importance를 AND

따라서 cue gate 자체의 효과는 `fastgs_gradient_only`와 `cue_vcd_k*`를
비교해야 한다. `baseline`은 기존 O-SCD density policy 전체와의 참고 비교다.

## 6. ref → SC1 결과

아래 값은 seed 0/1/2의 mean ± sample standard deviation이다.

| Condition | K | mean-frame mIoU | F1 | Precision | Recall | Final GS | Splits | Runtime(s) | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | - | 0.5857 ± 0.0105 | 0.6825 ± 0.0084 | 0.8285 ± 0.0028 | 0.8441 ± 0.0189 | 1,283,749 ± 27 | 242 ± 24 | 37.29 ± 0.63 | 2.310 ± 0.003 |
| FastGS gradient-only | - | 0.6063 ± 0.0108 | 0.6989 ± 0.0076 | 0.8253 ± 0.0044 | 0.8755 ± 0.0276 | 1,286,653 ± 128 | 3,152 ± 127 | 26.36 ± 0.15 | 2.098 ± 0.001 |
| soft cue-VCD | 1 | 0.6077 ± 0.0132 | 0.6990 ± 0.0099 | 0.8267 ± 0.0023 | 0.8852 ± 0.0216 | 1,285,746 ± 70 | 2,245 ± 70 | 26.93 ± 0.36 | 2.120 ± 0.001 |
| soft cue-VCD | 3 | 0.6076 ± 0.0124 | 0.6995 ± 0.0090 | 0.8246 ± 0.0048 | 0.8883 ± 0.0142 | 1,285,804 ± 66 | 2,303 ± 66 | 27.76 ± 0.30 | 2.117 ± 0.005 |
| soft cue-VCD | 5 | 0.6122 ± 0.0066 | 0.7023 ± 0.0054 | 0.8274 ± 0.0011 | 0.8974 ± 0.0087 | 1,285,707 ± 46 | 2,206 ± 46 | 28.20 ± 0.06 | 2.121 ± 0.007 |
| soft cue-VCD | 10 | **0.6143 ± 0.0043** | **0.7043 ± 0.0022** | 0.8255 ± 0.0020 | **0.8974 ± 0.0087** | **1,285,579 ± 135** | **2,078 ± 135** | 30.10 ± 0.02 | 2.119 ± 0.005 |

Clone은 모든 cue-VCD run에서 0이었다. 현재 scene extent와 FastGS
`dense=0.001` 기준으로 실제 통과 Gaussian이 모두 large/split 쪽이었기
때문이다.

### Matched FastGS control 대비 K=10

- split: `3151.7 → 2078.3`, `-1073.3` (`-34.1%`)
- mean-frame mIoU: `0.6063 → 0.6143`, `+0.0080`
- F1: `0.6989 → 0.7043`, `+0.0054`
- recall: `0.8755 → 0.8974`, `+0.0220`
- precision: `0.8253 → 0.8255`, 거의 동일
- runtime: `26.36s → 30.10s`, 약 `+14.2%`
- peak memory: 약 `2.10 GiB → 2.12 GiB`

즉 ref→SC1에서는 raw cue를 alpha-T로 여러 view에 투영한 gate가 FastGS
gradient-only가 만든 Gaussian의 약 1/3을 제거하면서 mask 성능을 유지한
것이 아니라 소폭 높였다.

성능 증가는 주로 recall 방향이었다. 95 frames 합계의 3-seed 평균에서
K=10은 matched control 대비 FN을 약 `269,183 → 221,676`으로 줄였고,
FP는 약 `400,578 → 410,114`로 늘렸다. 즉 cue-VCD가 더 넓은 실제 change를
표현하면서 작은 FP 비용을 지불한 결과다.

## 7. K에 대한 해석

K가 커질수록 단순 importance candidate 수가 단조 감소하지는 않았다.
평균 importance candidate/frame은 대략:

```text
K=1: 1347
K=3: 1440
K=5: 1490
K=10:1495
```

이는 positive alpha-T footprint mass를 평균한 score가 여러 view에서
지속되는 영역뿐 아니라 **여러 view가 덮는 공간 범위**도 넓히기 때문이다.
따라서 이 score를 pure consistency probability로 해석하면 안 된다.

반면 gradient와 importance의 최종 교집합인 실제 split은:

```text
K=1: 2245
K=3: 2303
K=5: 2206
K=10:2078
```

로 K=10에서 가장 적었고, mIoU/F1 분산도 가장 작았다. 이 제한된
ref→SC1 결과에서는 K=10을 primary setting으로 선택할 근거가 있지만,
K 증가가 항상 noisy candidate를 단조 제거한다는 가설까지 증명한 것은
아니다.

## 8. Pruning을 이번 단계에 넣지 않은 이유

FastGS reconstruction VCP의 score 방향을 raw change support에 그대로
적용할 수 없다.

- 높은 RGB reconstruction-degradation score: pruning 후보로 사용할 수 있음
- 높은 change-support score: 실제 change를 설명하므로 **보존 대상**

이번 `S_i^{change}`는 footprint-size/view-coverage bias도 남아 있고,
`e^-`를 이용한 calibrated non-change support ratio도 아직 검증하지 않았다.
따라서 이 score로 pruning하면 실제 change Gaussian을 반대로 제거할 위험이
있다. 첫 단계는 cue-VCD densification-only로 종료하고, VCP는

```text
low multi-view change support
AND high multi-view non-change support
AND existing opacity/size condition
```

을 별도 synthetic/real validation한 뒤 진행해야 한다.

## 9. Temporal topology 제한

현재 `TemporalGeometryChangeModel`의 state tensor는 Gaussian identity와
고정된 `[N,S,...]` 구조다. Immutable base만 clone/split하면
`state_change_dc`, state geometry delta, lifecycle buffer의 row alignment가
깨진다.

따라서 이번 구현은 temporal sidecar를 resize하지 않고 mutable
`R_change` bank에서만 검증했다. Lifespan에 연결하려면 다음 중 하나가 먼저
필요하다.

1. current OPEN episode 전용 residual Gaussian bank
2. clone/split lineage를 모든 state-local tensor와 optimizer state에
   원자적으로 적용하는 topology transaction

CLOSED historical episode와 immutable reference는 어떤 경우에도 density
operation 대상이 되어서는 안 된다.

## 10. 무결성 결과

18개 full run 모두 다음을 통과했다.

- future view access: 0
- GT causal-loop use: false
- manual boundary use: false
- immutable reference PLY hash 변화: 0
- topology tensor length mismatch: 0
- optimizer parameter/moment length mismatch: 0
- cue-based prune: 0

CUDA rasterizer의 atomic accumulation 때문에 동일 seed도 완전한 bitwise
결정론을 보장하지 않는다. 따라서 단일 run을 결론으로 사용하지 않고
3-seed mean/std를 보고했다.

## 11. 실행 명령

개별 실행:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_ref_sc1_change_cue_density \
  --condition cue_vcd \
  --k-views 10 \
  --seed 0 \
  --output-dir outputs/ref_sc1_soft_alpha_t_vcd/seed0/cue_vcd_k10
```

반복 결과 집계:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.summarize_ref_sc1_change_cue_density \
  --root outputs/ref_sc1_fastgs_soft_alpha_t_vcd_repeats_20260824
```

주요 생성물:

```text
comparison.json
comparison.csv
comparison.md
ref_sc1_miou_comparison.png
ref_sc1_gaussian_growth_comparison.png
seed*/condition/summary.json
seed*/condition/frame_metrics.csv
seed*/condition/density_events.jsonl
seed*/condition/new_gaussian_lineage.npz
```

이 파일들은 experiment artifact이므로 Git에 포함하지 않는다.

## 12. 결론

ref→SC1 범위에서는 soft raw change cue + alpha-T VJP + causal K=10 gate가
FastGS gradient-only 대비 Gaussian split을 약 34% 줄이고 mIoU/F1을 소폭
높였다. 따라서 다음 단계의 기본 후보는 K=10이다.

다만 이것은 아직 lifespan-aware density control이 아니며, score가 pure
multi-view consistency probability도 아니다. 다음 연구 순서는

1. ref→SC1에서 positive/negative support ratio 기반 VCP를 별도 검증
2. PASLCD stable scene에서 redundant growth 감소 확인
3. ESCD `ref→SC1→SC2→SC3`로 확대
4. residual bank 설계 후 current OPEN lifespan에만 temporal integration

이어야 한다.

SC1 안의 semantic new-object identity annotation을 density decision이나
평가에 사용하지 않았으므로, 이번 결과만으로 new-object region recall이
개별적으로 개선됐다고 주장하지 않는다.
