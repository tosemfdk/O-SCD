# ESCD K=10 oracle state-local density-view ablation

## 1. 질문

`ref → SC1`에서 가장 좋았던 soft alpha-T cue-VCD K=10을 다음 조건으로
확장했다.

1. `ref → SC2` 독립 실행
2. `ref → SC3` 독립 실행
3. `ref → SC1 → SC2 → SC3` 연속 실행

연속 실행에서는 oracle change event를 안다고 가정하고, K=10 density score가
다른 state의 과거 frame을 뽑지 못하게 했다.

```text
SC1 timestamp:
    K views ∈ SC1 only

SC2 timestamp:
    K views ∈ SC2 only

SC3 timestamp:
    K views ∈ SC3 only
```

이 oracle 정보는 **density importance를 계산할 view bank 제한에만**
사용했다. Mutable `R_change` bank 자체는 reset하지 않았고, optimization
replay는 기존처럼 모든 이미 처리된 causal frame을 선택할 수 있다. 이
설계는 다음 질문만 분리한다.

> 연속 실행 성능 저하가 K-view density score에 이전 state frame이 섞이기
> 때문인가?

## 2. 공통 설정

- cue: O-SCD pixel + SAM raw ref--inf cue `[0,1]`
- binary threshold: 사용하지 않음
- importance:

\[
S_i^{change}=\frac{1}{K}\sum_{j\in V_t}\sum_p
\alpha_{i,j}(p)T_{i,j}(p)C_j(p)
\]

- K: 10
- density decision: frame당 16 updates 중 `update_index=4`에서 1회
- density operation: FastGS gradient AND cue importance clone/split
- cue-based pruning: 없음
- seeds: 0, 1, 2
- GT: causal inference가 모두 끝난 뒤 평가에만 사용

독립 SC2/SC3는 매 실행마다 immutable reference topology에서 새 mutable
`R_change` bank를 시작한다. 연속 실행은 하나의 mutable bank를 304 frames
동안 유지한다.

## 3. Oracle view-bank 동작 확인

연속 실행에서 density sample 수는 boundary 직후 다음처럼 reset됐다.

```text
timestamp 94  / SC1: 10 views, all SC1
timestamp 95  / SC2:  1 view,  all SC2
timestamp 96  / SC2:  2 views, all SC2
timestamp 104 / SC2: 10 views, all SC2

timestamp 198 / SC2: 10 views, all SC2
timestamp 199 / SC3:  1 view,  all SC3
timestamp 200 / SC3:  2 views, all SC3
timestamp 208 / SC3: 10 views, all SC3
```

3 seeds 전체에서:

- future density view access: 0
- cross-state density view access: 0
- oracle labels used for optimization replay: false
- oracle labels used for density sampling: true

따라서 이후의 연속 성능 저하는 K=10 importance에 다른 state frame이 직접
섞여서 생긴 결과가 아니다.

## 4. 독립 ref → state 결과

3-seed mean ± sample standard deviation이다.

| Condition | Frames | mIoU | F1 | Precision | Recall | Splits | Final GS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref → SC1 | 95 | 0.6143 ± 0.0043 | 0.7043 ± 0.0022 | 0.8255 ± 0.0020 | 0.8974 ± 0.0087 | 2,078 ± 135 | 1,285,579 ± 135 |
| ref → SC2 | 104 | **0.6843 ± 0.0038** | **0.8051 ± 0.0027** | 0.7083 ± 0.0034 | **0.9366 ± 0.0034** | 2,186 ± 118 | 1,285,687 ± 118 |
| ref → SC3 | 105 | 0.6303 ± 0.0036 | 0.7626 ± 0.0034 | 0.7493 ± 0.0029 | 0.8543 ± 0.0018 | 1,959 ± 168 | 1,285,460 ± 168 |

즉 각 state를 reference에서 독립적으로 시작하면 K=10 density control은
SC2와 SC3에서도 동작한다. 특히 SC2는 recall이 높아 가장 좋은 mIoU를
기록했다.

## 5. 연속 oracle-local K=10 결과

연속 304-frame 전체 결과:

| Metric | Result |
|---|---:|
| mean-frame mIoU | 0.5142 ± 0.0014 |
| F1 | 0.6336 ± 0.0008 |
| Precision | 0.7267 ± 0.0059 |
| Recall | 0.6533 ± 0.0098 |
| Total splits | 5,781 ± 280 |
| Final GS | 1,289,282 ± 280 |
| Runtime | 93.35 ± 0.54 s |
| Peak CUDA memory | 약 3.73 GiB |

연속 run을 state별로 나누면 다음과 같다.

| State | Independent mIoU | Continuous oracle-local K mIoU | Difference |
|---|---:|---:|---:|
| SC1 | 0.6143 ± 0.0043 | 0.6124 ± 0.0053 | -0.0019 |
| SC2 | 0.6843 ± 0.0038 | 0.5593 ± 0.0005 | **-0.1250** |
| SC3 | 0.6303 ± 0.0036 | 0.3805 ± 0.0072 | **-0.2498** |

SC1은 independent run과 거의 같지만, state가 누적될수록 성능이 크게
내려갔다. SC3의 연속 recall은 `0.3920 ± 0.0228`로 independent SC3의
`0.8543 ± 0.0018`보다 크게 낮았다.

## 6. 해석

### 확인된 것

Oracle로 density K-view bank를 완전히 state-local하게 만들어도 연속
SC2/SC3 degradation은 사라지지 않았다.

따라서 주된 문제는:

```text
K-view importance에 이전 state cue가 섞임
```

하나가 아니라:

```text
동일 mutable R_change parameter/topology가
SC1 → SC2 → SC3 optimization을 계속 공유함
```

에 있다.

K-view bank는 어떤 Gaussian을 추가로 split할지만 제한한다. 이미 SC1에서
학습한 DC/opacity/geometry와 생성된 topology를 닫거나 archive하지 않는다.
또한 이번 isolation 실험의 training replay는 oracle state-local하지 않아서,
SC2/SC3 optimization 중 과거 SC1/SC2 frame도 다시 선택될 수 있다.

즉 density sampling을 state별로 끊는 것만으로는 representation memory를
state별로 분리할 수 없다.

### Density 결과가 나쁘다는 뜻은 아님

독립 SC2/SC3 결과는 각각 `0.6843`, `0.6303`으로 정상적이다. 따라서 soft
alpha-T K=10 gate 자체가 SC2/SC3 cue를 표현하지 못한 것은 아니다.

이번 failure는 다음처럼 분류해야 한다.

- density evidence failure: 아님. Cross-state sampled density view는 0.
- topology operation failure: 아님. 모든 topology/optimizer audit 통과.
- evolving representation failure: 맞음. 하나의 mutable bank가 state별
  current representation을 분리하지 못함.

## 7. Pruning

이번 확장에서도 pruning은 실행하지 않았다.

Positive change alpha-T mass가 높다는 것은 제거 대상이 아니라 change
support가 강하다는 뜻이다. Oracle K-view restriction이 있어도 이 score를
FastGS reconstruction VCP 방향으로 뒤집어 쓰면 실제 change Gaussian을
지울 수 있다.

VCP를 추가하려면 별도로:

```text
low positive change support
AND high non-change support
AND existing opacity/size condition
```

을 검증해야 한다.

## 8. 다음 구현 방향

연속 evolving scene에는 density view bank뿐 아니라 mutable capacity 자체가
state-local해야 한다.

가장 안전한 다음 단계는:

1. immutable reference는 유지
2. current OPEN lifespan 전용 residual Gaussian bank를 별도로 둠
3. boundary/CLOSE에서 residual bank를 historical state로 archive
4. 다음 OPEN에서 새 residual bank/topology를 시작
5. optimizer replay도 current state의 causal views로 제한

이다.

기존 fixed `[N,S,...]` temporal sidecar를 base densification에 맞춰 조용히
resize하는 방식은 Gaussian identity와 closed history alignment를 깨므로
사용하면 안 된다.

## 9. 출력

```text
outputs/escd_soft_alpha_t_vcd_k10_scope_oracle_20260824/
├── scope_comparison.json
├── scope_comparison.csv
├── scope_comparison.md
├── independent_vs_continuous_state_miou.png
└── seed{0,1,2}/
    ├── ref_sc2/
    ├── ref_sc3/
    └── continuous_oracle/
```

각 run에는 `summary.json`, `frame_metrics.csv`, `density_events.jsonl`, lineage
NPZ와 diagnostic plot이 있다. 생성물은 Git에 포함하지 않는다.

## 10. 실행 예

독립 SC2:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_ref_sc1_change_cue_density \
  --condition cue_vcd --scope scene_change2 --max-frames 104 \
  --k-views 10 --seed 0 \
  --output-dir outputs/ref_sc2_k10/seed0
```

연속 oracle state-local K bank:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_ref_sc1_change_cue_density \
  --condition cue_vcd --scope continuous --max-frames 304 \
  --oracle-state-local-density-views --k-views 10 --seed 0 \
  --output-dir outputs/continuous_oracle_k10/seed0
```
