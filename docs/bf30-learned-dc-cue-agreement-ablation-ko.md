# BF30 learned-\(C_i\) / cue agreement detector ablation

## 목적

기존 learned-DC BF30에서는 DC가 cue loss로 학습되지만 lifecycle detector는
DC를 직접 보지 않는다. 이 ablation은 Gaussian별로 현재 학습된 change
magnitude \(C_i\)와 alpha-T로 귀속된 cue 비율 \(q_i\)를 비교하여
OPEN/CLOSE를 결정한다.

동시에 다음 failure hypothesis를 검증한다.

> OPEN에서 CLOSE 후보가 생겨도 DC optimizer가 \(C_i\)를 현재 cue 쪽으로
> 움직이면 mismatch가 줄어들어 CLOSE evidence가 사라질 수 있다.

## learned change magnitude

O-SCD의 `load_ply_change`는 raw DC를 0으로 초기화하지만 SH rasterizer는 이를
RGB 0.5로 변환한다. 따라서 intrinsic RGB를 그대로 확률로 사용하면 초기
Gaussian이 change/non-change 양쪽에 동일한 0.5 agreement를 가져 detector가
열리지 않는다.

이 실험에서는 raw-zero를 semantic change 0으로 만드는 positive magnitude를
사용한다.

\[
C_i = \operatorname{clamp}\left(
2\left(\operatorname{mean}(\operatorname{SH2RGB}(DC_i))-0.5\right),
0,1\right).
\]

현재 frame cue의 Gaussian별 alpha-T pseudo-count가
\(\Delta a_i,\Delta b_i\)라면

\[
q_i = \frac{\Delta a_i}{\Delta a_i+\Delta b_i}.
\]

## agreement evidence

현재 DC와 cue가 다를 확률을 FLIP, 같을 확률을 KEEP evidence로 사용한다.

\[
e_i^{\mathrm{flip}}
= q_i(1-C_i) + (1-q_i)C_i,
\]

\[
e_i^{\mathrm{keep}}
= q_iC_i + (1-q_i)(1-C_i).
\]

실제 pseudo-count는 위 비율에 기존 capped evidence mass를 곱한 값이다.
FLIP/KEEP block은 기존 single-candidate Beta log Bayes factor로 누적하며
threshold는 BF 30이다.

Candidate가 commit되면 lifecycle bit를 toggle한다. OPEN commit은 intrinsic
rendered DC를 1로, CLOSE commit은 O-SCD 초기값인 raw DC 0으로 맞추고 해당
DC Adam row state를 reset한다. CLOSED row는 계속 렌더링과 optimizer에서
제외된다.

## 비교 조건

- stream: continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames
- seed: 0
- updates per frame: 16
- BF threshold: 30
- render support: OPEN + fixed raw-zero NEVER_OPEN occluder
- optimizer: all OPEN rows, NEVER_OPEN/CLOSED frozen
- loss: local cue support + \(G_{prev}\), weight 7.5
- density: ACTIVE-only O-SCD clone/split at local update 4, pruning off

두 run의 유일한 의도적 차이는 live candidate의 DC optimizer 정책이다.

- `adapt`: candidate가 살아 있어도 DC를 계속 학습한다.
- `freeze`: candidate가 살아 있는 row의 DC와 DC Adam state만 보존한다.
  geometry/opacity/scaling/rotation은 계속 학습한다.

## 결과

| 방식 | mIoU | F1 | Precision | Recall | SC1 | SC2 | SC3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 learned DC detector | 0.4498 | 0.5814 | 0.6114 | 0.6586 | 0.4642 | 0.4902 | 0.3967 |
| lifespan gate | 0.4264 | 0.5698 | 0.4516 | 0.9570 | 0.4412 | 0.4371 | 0.4023 |
| agreement adapt | 0.3399 | 0.4755 | 0.3451 | 0.8658 | 0.2630 | 0.3013 | 0.4478 |
| agreement freeze | 0.3078 | 0.4427 | 0.2905 | 0.9414 | 0.2650 | 0.2342 | 0.4194 |

| 진단 | adapt | freeze | freeze-adapt |
|---|---:|---:|---:|
| OPEN | 124,084 | 170,284 | +46,200 |
| CLOSE | 18,970 | 38,321 | +19,351 |
| REOPEN | 5,299 | 9,438 | +4,139 |
| candidate commit | 143,054 | 208,605 | +65,551 |
| same-scene transition | 13,133 | 32,965 | +19,832 |
| final ACTIVE | 107,900 | 134,858 | +26,958 |
| mean rendered-positive fraction | 0.1415 | 0.1827 | +0.0413 |
| live-candidate gap before | 0.548835 | 0.644871 | - |
| live-candidate gap after | 0.547131 | 0.644871 | - |
| one-frame gap delta | -0.001704 | 0.000000 | - |

## 해석

### 1. 예상한 self-erasure는 실제다

`adapt`에서는 live OPEN candidate의 \(|C_i-q_i|\)가 한 frame의 16-step
optimization 동안 평균 0.001704 감소했다. `freeze`에서는 parameter와 Adam
state가 정확히 보존되어 변화가 0이었다.

작은 per-frame 차이가 candidate block 전체에 반복되면서 freeze의 CLOSE는
adapt보다 약 2배, REOPEN은 약 1.78배 많아졌다. 즉 학습되는 DC가 mismatch를
일부 흡수하여 CLOSE를 막고 있었다.

### 2. 그러나 DC freeze는 해답이 아니다

Mismatch를 그대로 남기면 recall은 0.8658에서 0.9414로 증가하지만 precision은
0.3451에서 0.2905로 하락한다. same-scene transition도 13,133에서 32,965로
증가한다. 특히 SC2 mIoU가 0.3013에서 0.2342로 내려가며 repeated-state
구간에서 false transition과 넓은 false-positive mask가 누적된다.

원인은 하나의 3D Gaussian이 view마다 서로 다른 foreground/background
pixel을 덮고 alpha-T responsibility도 occlusion과 geometry 변화에 따라
달라지기 때문이다. 따라서 per-view \(q_i\)와 하나의 view-independent
\(C_i\) mismatch를 그대로 lifecycle flip으로 해석하면 불일치가 너무 많다.

### 3. 다음 설계 방향

Parameter 자체를 freeze하기보다 candidate 시작 시점의 detached
\(C_i^{anchor}\)를 detector sidecar에 저장하고 optimizer는 계속 동작시키는
편이 낫다. 이때 commit은 한 view의 mismatch가 아니라 여러 관측 view에서
동일 방향 mismatch가 반복될 때만 허용해야 한다.

즉 다음 후보는 다음 조합이다.

1. detached candidate-start DC anchor,
2. trainable representation은 계속 최적화,
3. view-consistent mismatch confirmation,
4. commit 시에만 DC/lifespan boundary 정렬.

## 무결성

- CLOSED parameter/Adam drift: 0
- inactive gradient violation: 0
- future-view access: 0
- topology integrity: pass

실험 산출물은
`/tmp/escd_bf30_learned_dc_agreement_u16_20260901_v1/`에 보존한다.
