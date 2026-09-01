# BF30에서 local regularization + `G_prev` ablation

## 1. 질문

현재 BF30 lifespan 설정에서 원본 O-SCD의 화면 전체 sparsity regularizer를
다음 두 항으로 교체하면 어떻게 되는지 확인한다.

1. cue가 약한 **pixel에서만** change probability를 줄이는 local regularizer
2. 직전 frame update가 cue 밖에서 새로 키운 영역만 다시 억제하는 `G_prev`

특히 다음 현상을 함께 본다.

- `OPEN`인데 DC가 0.5 아래로 내려간 Gaussian이 줄어드는가?
- 연속 `ref -> SC1 -> SC2 -> SC3`의 current-state mask가 개선되는가?
- loss 변경만으로 잘못된 lifecycle transition까지 줄어드는가?

## 2. 통제 조건

두 조건은 loss 외에는 동일한 fresh paired run이다.

| 항목 | 값 |
|---|---|
| stream | 연속 `ref -> SC1 -> SC2 -> SC3`, 304 frames |
| seed / update | seed 0 / frame당 16 updates |
| detector | `single_candidate_beta` |
| change 확정 기준 | Bayes factor `30`, 즉 `log BF >= ln(30)` |
| detector evidence | mutable bank의 lifespan-agnostic alpha-T capped pseudo-count |
| renderer | `OPEN + NEVER_OPEN`; `CLOSED` 제외 |
| `NEVER_OPEN` | 고정 zero-DC occluder, 모든 parameter freeze |
| optimizer | 현재 view visibility와 무관하게 모든 `OPEN` row의 전 parameter 학습 |
| first OPEN DC | 기존 값 보존 (`preserve`) |
| replay | 현재 view 확률 `0.33`, 아니면 이미 처리된 view에서 causal sampling |
| density | ACTIVE-only O-SCD clone/split, update 4, prune off |

비교 조건은 다음과 같다.

```text
global:
  기존 O-SCD global positive sparsity

local + G_prev:
  local cue support + 직전 frame의 unsupported positive-growth replay
  lambda_g = 7.5
```

## 3. Loss 정의

### 3.1 공통 change probability

view `v`에서 렌더된 3-channel change image를 `R_v`라고 하면 SSF에서 사용하는
change probability는 다음과 같다.

$$
m_v(p)=\sigma\left(\frac{1}{3}\sum_{c=1}^{3}R_{v,c}(p)\right).
$$

cached O-SCD cue `C_v(p)`는 `0..2` 범위이므로 local support는 다음처럼 고정한다.

$$
X_v(p)=\operatorname{clip}\left(\frac{C_v(p)}{2},0,1\right).
$$

### 3.2 Global baseline

기존 조건은 화면 전체 change mass에 같은 축소 압력을 준다.

$$
L_{\mathrm{global}}(v)
=\operatorname{mean}_p[C_v(p)(1-m_v(p))]
+\log\left(1+\operatorname{mean}_p[m_v(p)]^2\right).
$$

따라서 cue로 충분히 지지되는 pixel도 global mean을 통해 음의 방향 압력을 받는다.

### 3.3 Local term

local base loss는 cue가 약한 pixel에서만 change probability를 줄인다.

$$
B(v)
=\operatorname{mean}_p[C_v(p)(1-m_v(p))]
+\operatorname{mean}_p[(1-X_v(p))m_v(p)].
$$

### 3.4 직전 frame positive-growth map

frame `t-1`의 lifecycle 처리 직후·최적화 직전 확률과 frame 최적화 직후 확률의
양의 차이만 저장한다.

$$
G_{t-1}(p)
=\operatorname{ReLU}\left(
m^{\mathrm{post}}_{t-1}(p)-m^{\mathrm{pre}}_{t-1}(p)
\right).
$$

`G_{t-1}`은 detach하며 `t-1` camera 좌표에 남긴다. 이 정의의 pre-render는 이미
해당 frame의 lifecycle 결정과 first-OPEN 초기화를 반영하므로, **같은 frame의
OPEN/초기화 점프는 `G_prev`에 포함되지 않는다.**

### 3.5 `local + G_prev`

일반 replay view가 선택되면 `B(v)`만 사용한다. sampler가 최신 view `t`를
선택하고 `t>0`이면 같은 optimizer step에서 `t-1`도 현재 lifespan timestamp로
다시 렌더하고 다음 loss를 사용한다.

$$
\begin{aligned}
L_{\mathrm{latest}}
&=\frac{1}{2}\left(B(t)+B(t-1)\right) \\
&\quad+\lambda_g\operatorname{mean}_p\left[
G_{t-1}(p)(1-X_{t-1}(p))m_{t-1}^{\mathrm{current}}(p)
\right],\\
\lambda_g&=7.5.
\end{aligned}
$$

추가 항은 직전 update에서 실제로 증가했고, cue support가 약하며, 현재
재렌더에서도 남아 있는 pixel만 억제한다.

## 4. 결과

| 지표 | global | local + `G_prev` | 차이 |
|---|---:|---:|---:|
| 전체 mean-frame mIoU | 0.1622 | **0.4230** | **+0.2608** |
| 전체 mean-frame F1 | 0.2117 | **0.5538** | **+0.3422** |
| precision | **0.7237** | 0.6102 | -0.1135 |
| recall | 0.2623 | **0.6300** | **+0.3677** |
| SC1 mIoU | 0.2707 | **0.4375** | +0.1668 |
| SC2 mIoU | 0.1721 | **0.4563** | +0.2843 |
| SC3 mIoU | 0.0544 | **0.3770** | +0.3226 |
| 최종 OPEN 중 intrinsic DC `<0.5` | 48.60% | **33.85%** | -14.75%p |
| frame 평균 OPEN 중 intrinsic DC `<0.5` | 48.25% | **31.94%** | -16.31%p |

Lifecycle과 topology는 다음처럼 변했다.

| 지표 | global | local + `G_prev` | 차이 |
|---|---:|---:|---:|
| OPEN | 35,058 | 43,237 | +8,179 |
| CLOSE | 988 | 1,968 | +980 |
| REOPEN | 44 | 396 | +352 |
| same-scene repeated events | 426 | 666 | +240 |
| representation events (`OPEN/CLOSE`) | 425 | 657 | +232 |
| 최종 Gaussian 수 | 1,283,994 | 1,286,203 | +2,209 |
| runtime | 159.1 s | 190.1 s | +31.0 s |
| peak CUDA memory | 6.90 GiB | 7.44 GiB | +0.54 GiB |

`local + G_prev`에서 joint previous-view step은 총 `1,659`회였다. Weight를
곱하기 전 raw growth replay loss 평균은 `0.000436`, frame별 `G_prev` 평균은
`0.001326`, 관측 최대값은 `0.2310`이었다.

## 5. 해석

### 5.1 개선된 부분

Global regularizer가 주던 화면 전체 축소 압력을 제거하자 change 영역이 훨씬 잘
유지되었다. 개선의 중심은 precision이 아니라 recall이다.

```text
precision: -0.1135
recall:    +0.3677
```

특히 global 조건에서 거의 사라진 SC3 mask가 `0.0544 -> 0.3770`으로 회복했다.
OPEN row 중 DC가 0.5 아래인 비율도 최종 기준 14.75%p 줄었다. 따라서 사용자가
지적한 “OPEN인데 global shrinkage를 받아 검게 되는 row”에는 실제 효과가 있다.

### 5.2 해결되지 않은 부분

이 loss는 detector posterior에 DC-cue consistency를 직접 넣지 않는다. 다만
detector가 mutable geometry/opacity의 alpha-T evidence를 보기 때문에 학습 결과가
다음 frame detector trajectory에 간접 feedback을 준다.

그 결과 OPEN/CLOSE/REOPEN과 same-scene 반복 전이가 모두 증가했다. 즉 이번
결과는 **mask representation의 큰 개선**이지, 잘못 OPEN된 Gaussian을 DC와 cue의
불일치로 CLOSE하는 detector를 얻었다는 뜻은 아니다.

또한 최종 intrinsic DC 분포의 q05는 다음처럼 더 낮아졌다.

```text
global q05:          0.287
local + G_prev q05: 0.062
```

DC<0.5 row의 전체 비율은 감소했지만, 남은 dark tail은 오히려 더 극단적이다.
따라서 local loss가 모든 OPEN DC를 양수로 고정한 것이 아니라 DC 분포를 더
양극화했다고 보는 편이 정확하다.

### 5.3 현재 결론

`local + G_prev`는 BF30의 current-state mask loss로는 global보다 명확히 낫다.
그러나 다음 문제는 그대로 남는다.

1. detector와 learned DC의 직접적인 정합성 부재
2. 증가한 lifecycle chattering
3. 늘어난 false-positive mass와 낮아진 precision
4. 더 큰 topology, runtime, CUDA memory

따라서 이 결과만으로 production 기본값을 교체하지는 않는다. 다음 detector
실험에서는 alpha-T cue evidence와 learned DC/opacity가 강하게 충돌하는 OPEN row를
별도의 probabilistic evidence로 반영해야 한다.

## 6. 무결성 및 검증

- 두 run 모두 CLOSED parameter/Adam 최대 drift: `0`
- 두 run 모두 inactive gradient violation: `0`
- 두 run 모두 future-view access: `0`
- targeted tests: `38 passed`
- CUDA 3-frame smoke: global/local 모두 통과

## 7. 구현과 산출물

- runner: `experiments/run_online_dynamic_active_oscd_density.py`
- tests: `tests/experiments/test_online_dynamic_active_oscd_density.py`
- shared loss helper: `temporal/fusion.py`
- 기존 `G_prev` 정의/선택 근거: `docs/part11-loss-locality-ablation-ko.md`

실행 결과는 저장소 밖 `/tmp`에 두었다.

```text
/tmp/escd_bf30_fixed_never_open_all_open_loss_ablation_20260831/
  global/
    summary.json
    frame_metrics.csv
    lifecycle_events.jsonl
  local_gprev_w7p5/
    summary.json
    frame_metrics.csv
    lifecycle_events.jsonl
  comparison.json
  comparison.csv
  comparison.md
  comparison_curves.png
```
