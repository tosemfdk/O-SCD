# Persistent Project Context

이 저장소에서 작업하는 모든 Codex 에이전트는 사용자가 달리 명시하지 않는 한 아래 내용을 프로젝트의 기본 연구 방향과 문제 정의로 간주한다.

## 수식 출력 규칙

현재 사용하는 Codex CLI TUI는 LaTeX/KaTeX를 렌더링하지 않는다. 대화 응답에서는 `$...$`, `$$...$$`, `\(...\)`, `\[...\]` 구문을 사용하지 않는다.

- 짧은 수식은 Unicode 수학 기호와 위·아래 첨자로 적는다. 예: `πᵢ = aᵢ / (aᵢ + bᵢ)`
- 복잡한 수식은 `text` 코드 블록에 ASCII/Unicode 분수선과 정렬을 사용한다.
- LaTeX 원문이 필요한 경우에만 별도로 제공하고, 항상 바로 읽히는 Unicode 버전을 먼저 제공한다.

## 우리가 푸는 문제

기존 online Scene Change Detection(SCD)은 고정된 기준 장면 `R_ref`와 순차적으로 들어오는 이미지 사이의 변화를 검출한다. O-SCD는 이전 프레임에서 얻은 변화 단서를 `R_change`에 누적하여 멀티뷰 일관성을 확보하지만, 하나의 관측 구간 동안 post-change scene이 고정되어 있다고 가정한다.

이 프로젝트가 다루는 **SCD in an on-the-fly evolving scene**은 그 가정을 제거한다. 스트림을 촬영하는 동안 동일한 영역의 상태가 다음처럼 여러 번 바뀔 수 있다.

```text
의자 원래 위치 -> 의자 이동 -> 의자 제거 또는 재이동
```

기존 O-SCD의 단순 누적 방식을 그대로 사용하면 과거 상태를 설명하는 change cue와 현재 상태를 설명하는 cue가 동일한 `R_change`에 함께 남는다. 그 결과:

- 이미 사라진 변화가 계속 검출될 수 있다.
- 서로 충돌하는 증거가 평균화될 수 있다.
- 현재 장면 상태를 정확히 표현하지 못할 수 있다.

## 목표 시스템

입력은 고정된 reference 3D Gaussian Splatting 표현 `R_ref`와 온라인 이미지 스트림 `I_1, I_2, ..., I_t`이다. 시스템은 미래 프레임을 사용하지 않고 매 시점 `t`마다 다음을 수행해야 한다.

1. 현재 관측이 reference와 어디에서 다른지 검출한다.
2. 동일한 Gaussian 영역에서 과거 change evidence와 현재 evidence가 충돌하는지 판단한다.
3. 실제 장면의 변화 상태가 다시 바뀌었다면 change point를 검출한다.
4. 이전 상태를 지지하던 Gaussian의 lifespan을 종료한다.
5. 새로운 상태를 위한 lifespan을 시작한다.
6. 과거 변화의 합집합이 아니라 **현재 시점에 유효한 변화만** change mask로 출력한다.

## 시간적 변화 표현

Gaussian마다 하나의 누적 change score만 유지하지 않는다. 각 Gaussian `g_i`는 개념적으로 다음과 같은 시간적 상태 또는 상태 이력을 가진다.

```text
g_i: {(change state, start time, end time, confidence)}
```

기존 `R_change`의 persistent memory와 멀티뷰 융합 장점은 보존하되, 오래되었거나 현재 관측과 충돌하는 evidence를 자동으로 닫거나 교체할 수 있어야 한다.

## 핵심 목적과 제약

- **Online causality:** 시점 `t`의 판단에는 `I_1 ... I_t`만 사용하며 미래 프레임을 사용하지 않는다.
- **Current-state validity:** 출력 mask는 지금 유효한 변화만 나타내야 한다.
- **Temporal consistency:** 3D 변화 표현은 멀티뷰 일관성을 유지하면서 상태의 시작과 종료를 추적해야 한다.
- **Outdated evidence handling:** 과거 변화 evidence와 현재 evidence를 구분하고, 오래된 evidence가 현재 결과를 오염시키지 않도록 해야 한다.
- **Repeated evolution:** 같은 Gaussian 영역이 changed, reverted, removed, 또는 changed-again 상태로 반복 전이할 수 있음을 전제로 한다.

## 현재 detector의 확정 범위와 입력 계약

현재 Gaussian lifespan detector는 Gaussian별 **binary reference-change validity**만
추적한다.

```text
inactive/reference-consistent -> active/reference-different -> inactive -> active
```

- 동일 Gaussian이 `SC1`과 `SC2`에서 모두 reference와 다르면 `active -> active`
  `KEEP`으로 처리한다. 두 changed appearance 사이의 의미적 상태 구분이나 새
  lifespan 분리는 현재 설계 범위가 아니며, 이는 의도된 동작이다.
- Detector는 각 신규 online frame에서 representation optimization을 시작하기 전에
  계산한 raw cue의 Gaussian별 alpha-transmittance evidence만 한 번 사용한다.
- 학습된 change DC `C_i`, `C_i`와 cue의 차이, optimizer가 `C_i`를 움직인 양,
  같은 timestamp의 post-optimization render, 과거 replay view는 detector evidence로
  다시 사용하지 않는다.
- Candidate가 살아 있어도 stable/candidate Beta에는 이후 신규 frame의
  pre-optimization raw evidence만 추가한다. Representation 학습 결과가 이미 처리한
  frame의 detector 판단으로 역류해서는 안 된다.
- Learned-DC agreement 및 detached-anchor/K3 detector는 비교 실험으로만 보존하며
  현재 기본 detector가 아니다.
- Representation renderer에서는 frozen `NEVER_OPEN` Gaussian의 geometry/opacity
  occlusion을 유지하되 change color는 RGB black으로 override한다. `CLOSED`는
  렌더링하지 않는다.

## 한 문장 요약

> Detect not only where the scene differs from the reference, but also when that difference itself changes.

## 현재 구현 및 연구 단계

현재 저장소 checkpoint는 기존 oracle-boundary lifespan/E3 control과 causal Bayesian lifespan + active-only state geometry 구현을 함께 포함한다.

- Lifespan separation: state별 DC와 half-open interval을 구현하고 naive persistent O-SCD보다 과거 state 보존이 개선되는 것을 확인했다.
- Corrected cue experiment: O-SCD pixel + SAM2.1 cue, fixed pose, 304 frames, 이미지당 120 updates로 DC-only lifespan을 검증했다.
- E3 geometry: Gaussian index와 topology는 고정한 채 state별 xyz/DC/opacity/scale/rotation delta와 S0 soft anchor를 학습한다.
- E3 결과: geometry는 SSF loss를 낮췄지만 DC-only보다 overall mIoU/F1이 소폭 낮았다. 상세 내용은 [`docs/instance1-state-geometry-lifespan-comparison-ko.md`](docs/instance1-state-geometry-lifespan-comparison-ko.md)에 기록한다.
- Bayesian lifespan: immutable reference의 lifespan-agnostic alpha-T evidence, bounded Beta-Bernoulli BOCD와 명시적 MAP-reset 근사, binary OPEN/KEEP/CLOSE/REOPEN lifecycle, row-slot masked Adam, causal frame-major runner를 구현한다. `active -> active`는 항상 같은 slot의 `KEEP`이다.
- Bayesian ESCD 평가: 독립 `ref -> SC1/SC2/SC3`와 상태를 유지한 연속 `ref -> SC1 -> SC2 -> SC3` 304-frame 실험을 완료했다. 연속 MAP-reset run은 OPEN `81,268`, CLOSE/REOPEN `0`으로 lifespan separation에 실패했다. DC-only mean-frame mIoU는 `0.4264`, all-geometry는 `0.6208`이었지만 geometry가 동일 OPEN slot을 morphing해 failure를 가린 결과이므로 lifespan 개선으로 해석하지 않는다. 상세 내용은 [`docs/escd-bayesian-lifespan-experiment-results-ko.md`](docs/escd-bayesian-lifespan-experiment-results-ko.md)에 기록한다.
- BOCD 진단 확장: full posterior/Adams--MacKay lineage diagnostic과 protected reset-candidate Beam-2 ablation을 보존한다. Beam-2 binary all-geometry는 mean-frame mIoU/F1 `0.6140/0.7323`, OPEN/CLOSE/REOPEN `153,458/39,852/10,041`을 기록했다.
- Direct binary-state ablation: BOCD run length 없이 `P(z_t=active)`를 직접 추론하는 two-state Bayesian filter/controller, causal B0/B1/S0/S1 runner, pre/post-OPEN metric, exact closed-pair audit와 diagnostic plot을 구현했다. Immutable-reference alpha-T detector는 temporal slot parameter/lifecycle과 bitwise 독립이다.
- View-consistent direct binary lifespan: PASLCD D1에서 marginal `p_active`와 lifecycle mutation의 결합이 same-scene chattering의 핵심임을 확인한 뒤, committed-state `BF_open/BF_close`를 연속 observed view에서 확인하는 controller를 추가했다. `K=3, BF>=3`은 PASLCD repeated extras를 `556,469 -> 43,153`, REOPEN을 `109,847 -> 8,255`로 줄였고, all-geometry mIoU `0.4973`으로 O-SCD online `0.4887`을 유지했다. ESCD `ref -> SC1 -> SC2 -> SC3`에서는 B0/B1 mIoU `0.5981/0.5996`, OPEN/CLOSE/REOPEN `123,476/64,619/22,212`를 기록해 실제 전환 반응을 유지했지만 Beam-2 all-geometry `0.6140`보다 낮다. 상세 내용은 [`docs/view-consistent-binary-lifespan-confirmation-ko.md`](docs/view-consistent-binary-lifespan-confirmation-ko.md)에 기록한다.
- Active-visible optimizer isolation: temporal optimizer selection을 전체 OPEN pair가 아니라 `OPEN AND current temporal render radius > 0`인 row-slot으로 제한하고 매 update마다 visibility를 다시 계산한다. Off-view OPEN pair는 parameter와 Adam moment가 정확히 보존된다. 동일 K3/BF3 lifecycle에서 ESCD B0 DC-only mIoU는 `0.5981 -> 0.5922`, B1 all-geometry는 `0.5996 -> 0.6021`이었다. 동일 Beam-2 lifecycle에서는 DC-only `0.6047 -> 0.5991`, all-geometry `0.6140 -> 0.6082`였으며, 수정 후에도 all-geometry가 DC-only보다 `+0.0091` 높았다. 상세 내용은 [`docs/active-visible-temporal-optimizer-ablation-ko.md`](docs/active-visible-temporal-optimizer-ablation-ko.md)에 기록한다.
- Direct binary ESCD 결과: binary B0 DC-only/B1 all-geometry mean-frame mIoU는 `0.6063/0.6057`, soft S0/S1은 `0.5976/0.6091`이었다. Lifespan/optimizer invariant는 유지됐지만 binary/soft same-scene repeated extra transition이 `170,574/58,426`으로 많다. 현재 핵심 failure는 lifecycle 구현이 아니라 direct detector의 chattering이며, 다음 단계는 별도 실험에서 emission/transition calibration과 spatial/visibility consistency를 검증하는 것이다. 상세 내용은 [`docs/direct-binary-state-lifespan-ablation-ko.md`](docs/direct-binary-state-lifespan-ablation-ko.md)에 기록한다.
- PASLCD 비교: 20 scenes/500 frames에서 direct binary all-geometry는 O-SCD online 대비 u16에서 mIoU/F1 `+0.0190/+0.0175`, u120에서 `+0.0260/+0.0150`이었지만 O-SCD refined보다는 낮았다. Single-state PASLCD에서도 OPEN/CLOSE/REOPEN `922,714/446,622/109,847`이 발생해 detector chattering이 재확인됐다. 상세 내용은 [`docs/paslcd-direct-binary-state-comparison-ko.md`](docs/paslcd-direct-binary-state-comparison-ko.md)에 기록한다.
- FastGS-style change-cue density ablation: mutable `R_change` bank에 raw O-SCD pixel+SAM cue의 soft alpha-T VJP와 causal K-view FastGS gradient/importance 교집합을 연결했다. `ref -> SC1` 3-seed 결과에서 K=10은 matched FastGS gradient-only 대비 split을 `3151.7 -> 2078.3`(`-34.1%`)로 줄이고 mean-frame mIoU를 `0.6063 -> 0.6143`으로 높였다. Temporal lifespan sidecar는 여전히 fixed topology이며 cue-based pruning은 지원하지 않는다. 상세 내용은 [`docs/ref-sc1-fastgs-soft-alpha-t-cue-density-ko.md`](docs/ref-sc1-fastgs-soft-alpha-t-cue-density-ko.md)에 기록한다.
- K=10 ESCD scope extension: 독립 `ref -> SC2/SC3` mIoU는 `0.6843/0.6303`이었다. Oracle boundary로 density view bank의 cross-state sample을 0으로 만든 연속 `ref -> SC1 -> SC2 -> SC3`에서도 state별 mIoU가 `0.6124/0.5593/0.3805`로 하락했다. 따라서 density view sampling만 state-local하게 만드는 것으로는 persistent mutable `R_change`의 state contamination을 해결할 수 없으며, current lifespan 전용 residual bank가 필요하다. 상세 내용은 [`docs/escd-k10-oracle-state-local-density-ko.md`](docs/escd-k10-oracle-state-local-density-ko.md)에 기록한다.
- Active-lifespan density integration: immutable reference prefix와 dynamic residual child topology를 분리하고, child를 root Gaussian의 현재 lifespan slot에 고정했다. 현재 OPEN/current-view-visible row-slot만 all-geometry를 학습하며, K=10 soft alpha-T cue VCD는 active row만 split/clone하고 VCP는 현재 OPEN residual만 제거한다. 독립 SC1/SC2/SC3 seed-0 u120에서 frame-weighted mIoU/F1은 density-off `0.5882/0.7091`에서 `0.5891/0.7096`으로 소폭 변했다. SC1은 `+0.0068`, SC2는 `-0.0003`, SC3는 `-0.0033`으로 일관된 개선은 아니었다. Reference/prefix/closed-slot drift, inactive gradient violation, future-view access는 모두 0이었다. 상세 내용은 [`docs/active-lifespan-fastgs-cue-density-ko.md`](docs/active-lifespan-fastgs-cue-density-ko.md)에 기록한다.
- Persistent direct R_change ablation: lifespan slot에서 DC/geometry delta를 제거하고 Gaussian마다 하나의 mutable change DC/xyz/SH-rest/opacity/scaling/rotation을 직접 유지한다. CLOSE는 opacity gate만 숨기며 값과 Adam moment를 보존하고, REOPEN은 새 interval slot을 할당하되 같은 값에서 학습을 재개한다. 동일 K3/BF3 detector의 독립 SC1/SC2/SC3 u120에서 lifecycle event는 state-local baseline과 정확히 같았고 frame-weighted mIoU/F1은 `0.5882/0.7091 -> 0.5897/0.7102`였다. Peak CUDA memory는 약 `62%`, runtime은 `35--39%` 감소했지만 chattering은 그대로이므로 detector blocker를 해결하지는 않는다. 상세 내용은 [`docs/persistent-direct-rchange-lifespan-ablation-ko.md`](docs/persistent-direct-rchange-lifespan-ablation-ko.md)에 기록한다.
- Equal-status dynamic ACTIVE O-SCD density ablation: reference-prefix/residual class를 없애고 모든 mutable `R_change` row를 동등하게 취급한다. ACTIVE-visible row만 16-step O-SCD loss로 직접 DC/geometry를 학습하고 local update 4에서 ACTIVE-only gradient clone/split을 수행한다. 새 row는 source의 Bayesian/controller/lifespan state를 한 번 복사한 뒤 독립적으로 갱신된다. 독립 SC1/SC2/SC3에서 densify-only weighted mIoU/F1은 fixed topology `0.5779/0.7044` 대비 `0.5768/0.7033`으로 거의 중립이었다. ACTIVE `opacity<0.4` prune을 추가하면 18k--41k row를 제거하며 `0.5715/0.6992`로 하락했다. 상세 내용은 [`docs/dynamic-active-oscd-density-ko.md`](docs/dynamic-active-oscd-density-ko.md)에 기록한다.
- Never-open occlusion/appearance ablation: OPEN-only renderer가 unchanged foreground occluder를 제거하는 문제를 분리했다. NEVER_OPEN row를 고정 zero-DC occluder로만 복원하면 weighted mIoU가 `0.5779 -> 0.4557`로 하락했지만, 해당 row의 geometry는 고정하고 DC/opacity만 visible-view에서 학습하면 `0.6351/0.7539`까지 회복했다. 여기에 원본 O-SCD식 update-4 gradient clone/split을 켜면 `0.6449/0.7618`로 올라 원본 독립 O-SCD online `0.6471/0.7619`와 `-0.0023/-0.0002` 차이였다. 그러나 동일 state를 유지한 연속 `ref -> SC1 -> SC2 -> SC3`에서는 mIoU/F1 `0.4833/0.6028`로 원본 O-SCD `0.5178/0.6401`보다 낮았다. NEVER_OPEN appearance의 lifespan 우회와 cross-state causal replay가 남은 누적 오염 후보다. CLOSED row는 계속 숨기고 모든 parameter/Adam state drift는 0이었다. 상세 내용은 [`docs/open-never-open-appearance-ablation-ko.md`](docs/open-never-open-appearance-ablation-ko.md)에 기록한다.
- Raw-cue mixture density + black-child pruning ablation: raw BF30 + black frozen NEVER_OPEN + all-OPEN u16 continuous에서 density-off / gradient-only / gradient+cue-mixture / mixture+child-prune mean-frame mIoU/F1은 `0.4903/0.6271`, `0.4874/0.6248`, `0.4839/0.6209`, `0.4852/0.6224`였다. Child pruning은 generation-zero와 same-event child를 보호하고, 1 frame 이상 지난 OPEN densified child 중 intrinsic DC가 0.5 미만인 일부만 hard-prune하며 직접 부모/sibling support가 모두 없으면 한 child를 보존한다. Continuous에서 8,539개를 삭제해 final black ACTIVE 비율을 `0.6625 -> 0.6446`으로 낮췄지만 density-off를 넘지 못했다. PASLCD 20 scenes/500 frames에서는 네 조건이 `0.4566/0.6125`, `0.4561/0.6120`, `0.4515/0.6077`, `0.4523/0.6084`였고 child 62,678개를 삭제해도 O-SCD online `0.4887/0.6423`보다 낮았다. Mixture 대비 `+0.0008` mIoU point estimate는 Cantina CUDA 반복 변동과 같은 규모라 개선 증거로 보지 않는다. Cue-mixture OR split과 child pruning은 기본 방법으로 채택하지 않는다. 상세 내용은 [`docs/bf30-black-cue-mixture-density-ablation-ko.md`](docs/bf30-black-cue-mixture-density-ablation-ko.md), [`docs/bf30-black-child-prune-paslcd-ablation-ko.md`](docs/bf30-black-child-prune-paslcd-ablation-ko.md)에 기록한다.
- CLOSE-only asymmetric candidate output gate: `lifespan_gate_beta` BF30의 detector evidence/commit과 hard-gated training/density는 유지하면서, committed OPEN의 CLOSE candidate만 normalized log-BF 진행도에 따라 output opacity를 줄이고 CLOSED의 OPEN candidate는 commit 전까지 숨긴다. 연속 304 frames에서 hard 대비 mean-frame mIoU/F1은 `0.4261/0.5696 -> 0.4403/0.5833`이었고 FP는 684,942 pixel 줄었다. PASLCD 20 scenes/500 frames에서도 `0.3711/0.5125 -> 0.3840/0.5257`, 19/20 scene 개선, FP `-7.86%`를 기록했다. 양방향 gate는 PASLCD mIoU `0.2094`로 무너졌다. CLOSE-only는 유효한 precision 보정이지만 PASLCD O-SCD online `0.4887/0.6423`보다 여전히 낮으므로 detector/representation 전체 해법으로 해석하지 않는다. 상세 내용은 [`docs/bf30-close-only-transition-opacity-gate-ablation-ko.md`](docs/bf30-close-only-transition-opacity-gate-ablation-ko.md)에 기록한다.
- Raw-cue learned-DC CLOSE-candidate output ablation: pre-optimization raw alpha-T cue 분포만 사용하는 `single_candidate_beta` BF30과 learned persistent DC를 다시 결합하고, committed OPEN 중 fresh candidate Beta mean이 `<=0.4`인 CLOSE 방향 row만 `1-clamp(logBF/log30,0,1)`로 output opacity를 감쇠한다. Learned DC/geometry/opacity, detector, training, replay, density, hard lifecycle은 변경하지 않으며 frozen black NEVER_OPEN은 full-opacity occluder로 유지한다. 동일 run의 동일 parameter/lifecycle에서 hard/soft render를 함께 평가한 결과 연속 304 frames mIoU/F1은 `0.4896/0.6263 -> 0.4938/0.6304`, PASLCD 20 scenes/500 frames는 `0.4566/0.6124 -> 0.4604/0.6159`였고 19/20 scene에서 mIoU가 증가했다. Detector의 learned-DC input, OPEN-candidate preview, CLOSED drift, wrong gradient, future-view access는 모두 0이었다. 이는 detector 개선이 아니라 candidate uncertainty를 이용한 output precision 보정으로 해석한다. 상세 내용은 [`docs/bf30-raw-cue-learned-dc-close-candidate-output-ablation-ko.md`](docs/bf30-raw-cue-learned-dc-close-candidate-output-ablation-ko.md)에 기록한다.
- E4d XFeat-anchored active NEW geometry: E4a의 causal SAM/PCA/posterior와 sparse XFeat-512 anchor birth는 고정하고 NEW sidecar에만 geometry optimization, covariance clone/split, causal pruning을 허용한 D0--D3 controlled ablation을 완료했다. 304-frame overall IoU는 D0/D1/D2/D3 `0.4388/0.3508/0.3463/0.3560`이었다. 과거 보고한 `0.2275/0.1052/0.1065/0.1235`는 전체 prediction을 NEW GT와 비교한 legacy cross-scope diagnostic이며 NEW-sidecar IoU로 사용하지 않는다. Densified child reprojection precision 저하와 negative-DC/high-opacity occlusion, 과도한 scale·depth 자유도가 핵심 failure였으며 reference 1,283,501-row bitwise audit와 GT/future-view causal audit는 모두 통과했다. 상세 내용은 [`docs/part15-xfeat-anchored-active-new-geometry-ko.md`](docs/part15-xfeat-anchored-active-new-geometry-ko.md)에 기록한다.
- E4e learned NEW DC isolation + root-anchor radius: 모든 NEW row의 DC를 학습하면서 joint base+NEW SSF 대 base-free projected-cue DC, unconstrained 대 fixed-XFeat-xyz/root-radius hinge를 C0--C3 2×2로 비교했다. Object metric은 overall=`전체 prediction vs 전체 GT`, NEW=`learned-DC NEW sidecar vs NEW GT`, REMOVE=`opposite-sign base vs REMOVED GT`로 분리한다. 304-frame overall IoU는 `0.3616/0.2589/0.3552/0.2137`, NEW IoU는 `0.0752/0.1182/0.0424/0.0636`이었다. 자유 geometry에서 NEW-only DC는 NEW recall을 `0.0799 -> 0.5665`로 높였지만 NEW precision을 `0.5604 -> 0.1300`, overall precision을 `0.6672 -> 0.3084`로 낮췄다. XFeat xyz bitwise 고정과 anchor 1회 densification은 통과했으나 soft xyz hinge는 SC3 C3 child distance-ratio median `33.2`, max scale `76.5`의 재귀 density/footprint 폭주를 막지 못했다. Reference/causal audit는 모두 통과했다. 상세 내용은 [`docs/part16-learned-new-dc-isolation-anchor-radius-ko.md`](docs/part16-learned-new-dc-isolation-anchor-radius-ko.md)에 기록한다.
- Signed lifespan score density: independent `ref -> SC3`, u120에서 `active * (2p_plus_is_NEW-1) * (plus-minus alpha-T) / lifespan_age`로 gradient density를 대체했다. Target NEW object 004의 positive growth와 REMOVED object 010의 negative suppression 정확도는 `96.26%/96.52%`였지만 fixed top-2048 budget이 103 frame에서 모두 포화되어 Gaussian이 `+210,944` 증가했다. Gradient baseline 대비 mean-frame mIoU/F1은 `0.5901/0.7291 -> 0.5822/0.7232`로 하락했으므로 sign routing은 유효하되 raw score의 growth allocation은 채택하지 않는다. 다음 단계는 64/128/256 budget, mass normalization, root/lifespan child cap을 분리 검증하는 것이다. 상세 내용은 [`docs/ref-sc3-signed-lifespan-score-density-ko.md`](docs/ref-sc3-signed-lifespan-score-density-ko.md)에 기록한다.
- DA3 depth-prior NEW seed feasibility: SC3의 past-only 8-view pose-conditioned Depth Anything 3 depth를 immutable reference GS depth에 positive scale-only로 robust 정렬하고, confirmed NEW/high-confidence/front-of-reference pixel만 3D seed로 만들었다. Frame 48/52/64/80/84에서 object004 seed precision 평균은 `0.7099`, any-NEW precision은 `0.8663`, object010 leakage는 `0.00077`이었다. 2 cm causal voxel merge 후 1,392 seed 중 evaluation-only object004 subset 926개가 원통 표면을 형성했으며, causal XFeat object004 reprojection hit `11/11/3/11/24` 대비 DA3는 `252/476/353/369/732`였다. 이는 seed geometry feasibility이며 아직 u120 representation 결과는 아니다. 상세 내용은 [`docs/part17-da3-depth-prior-new-seeding-ko.md`](docs/part17-da3-depth-prior-new-seeding-ko.md)에 기록한다.
- DA3 fixed-geometry DC-only u120: independent `ref -> SC3`에서 DA3 seed xyz/scale/rotation/opacity를 고정하고 seed DC만 base-free projected NEW cue로 학습했다. 12k seed-cap run은 base signed DC-only 대비 overall IoU/F1을 `0.5379/0.6995 -> 0.5921/0.7438`, mean-frame IoU를 `0.5182 -> 0.5639`로 높였다. NEW sidecar IoU/F1은 `0.3294/0.4956`, object004 active 45-frame mean IoU/recall은 `0.3530/0.7129`, object010 내부 NEW FP는 0이었다. Cap을 20k로 늘려 14,271 seed를 쓰면 overall/NEW IoU가 `0.5904/0.3286`으로 소폭 하락해 추가 late seed가 해답은 아니었다. Reference와 모든 seed geometry field는 bitwise unchanged였고 future/GT birth access는 0이었다. 상세 내용은 [`docs/part18-da3-fixed-geometry-dc-only-u120-ko.md`](docs/part18-da3-fixed-geometry-dc-only-u120-ko.md)에 기록한다.
- DA3 R_change free-space NEVER_OPEN ablation: independent `ref -> SC3`에서 base R_change 514,810 occupied voxels와 기존 seed voxel을 피해 DA3 row를 `NEVER_OPEN`으로 넣고, pre-optimization raw-cue `lifespan_gate_beta` BF30이 OPEN한 row만 DC-only u120으로 학습했다. Immediate-ACTIVE occupancy control 대비 overall IoU/F1은 `0.5921/0.7438 -> 0.6071/0.7555`, object004 mean IoU는 `0.3529 -> 0.3632`로 증가했다. NEW precision은 `0.5623 -> 0.6634`였지만 recall은 `0.4424 -> 0.3816`, pixel-weighted NEW IoU는 `0.3291 -> 0.3197`로 하락했다. Final OPEN/NEVER_OPEN/CLOSED는 `9,658/1,325/1,017`이고 OPEN/CLOSE는 `11,470/1,812`였다. Center-voxel occupancy alone은 frame79까지 46 birth만 지연했고 결국 동일 12k cap에 도달했으므로 자연 포화 정지에는 radius/coverage 기반 occupancy가 추가로 필요하다. 상세 내용은 [`docs/part19-da3-never-open-rchange-occupancy-ko.md`](docs/part19-da3-never-open-rchange-occupancy-ko.md)에 기록한다.
- 이 checkpoint에는 MCMC, SGLD, relocation, unrestricted image-space Gaussian birth는 포함하지 않는다. Production temporal model의 기본 topology는 fixed이며, dynamic topology는 active residual density와 equal-status mutable-bank ablation runner에서만 명시적으로 사용한다.
