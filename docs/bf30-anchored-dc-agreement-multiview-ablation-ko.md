# BF30 detached `C_i^{anchor}` + multi-view agreement ablation

## 1. 목적

기존 learned-`C_i`/cue agreement detector에서는 lifecycle candidate가 살아 있는 동안에도
optimizer가 `C_i`를 cue 쪽으로 이동시킨다. 그 결과 OPEN→CLOSE mismatch가 detector 안에서
확정되기 전에 작아지는 self-erasure가 관측됐다.

이번 ablation은 parameter를 freeze하지 않는다. 대신 candidate가 시작된 시점의 semantic
change value만 detector state로 분리해 저장한다.

\[
C_i^{anchor}=\operatorname{stopgrad}(C_i(t_0))
\]

여기서 `t_0`는 Gaussian `i`의 현재 candidate 시작 timestamp다. Representation optimizer는
계속 현재 `C_i`와 geometry/opacity를 갱신하지만, candidate의 Bayes factor는 commit 또는
reject까지 `C_i^{anchor}`만 사용한다.

## 2. Detector

현재 view에서 Gaussian `i`에 alpha-T로 귀속된 positive/negative cue mass를 각각
`delta_a_i`, `delta_b_i`라 하고,

\[
q_i=\frac{\Delta a_i}{\Delta a_i+\Delta b_i}
\]

로 둔다. Candidate가 없는 row는 현재 learned `C_i`로 첫 mismatch를 평가하고, candidate가
시작되는 순간 그 값을 `C_i^{anchor}`에 복사한다. 이후 candidate block에서는 다음
FLIP/KEEP pseudo-count를 사용한다.

\[
e_i^{flip}=q_i(1-C_i^{anchor})+(1-q_i)C_i^{anchor}
\]

\[
e_i^{keep}=q_iC_i^{anchor}+(1-q_i)(1-C_i^{anchor})
\]

RESET-vs-KEEP Bayes factor는 기존 O(1) single-candidate Beta detector와 동일하게 누적한다.

### 2.1 Strict multi-view confirmation

BF30만 넘는 것으로는 commit하지 않는다. 현재 lifecycle bit와 반대 방향의 mismatch가
연속으로 관측된 view에서만 support count를 올린다.

- 현재 CLOSED: `q_i > C_i^{anchor}`
- 현재 OPEN: `q_i < C_i^{anchor}`
- 한 observed view라도 같은 방향을 지지하지 않으면 candidate를 reject한다.
- `BF >= 30`이고 support view가 최소 `K=3`일 때만 commit한다.

각 online frame에서 detector update는 한 번만 수행되므로 support 3은 서로 다른 streaming
timestamp의 observed view 3개를 뜻한다. 미래 view는 사용하지 않는다.

### 2.2 Commit과 optimizer

- Candidate 동안 모든 OPEN row의 optimizer는 계속 동작한다.
- Detector의 `C_i^{anchor}`는 optimizer graph와 분리된 topology-aligned buffer다.
- OPEN commit: learned DC를 rendered white 1로 맞추고 해당 DC Adam row를 reset한다.
- CLOSE commit: raw DC zero로 복원하고 해당 DC Adam row를 reset한 뒤 row를 숨긴다.
- Densify clone/split child는 source의 live candidate와 anchor를 한 번 복사한 뒤 독립적으로
  갱신한다.

## 3. 실험 조건

- Stream: continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames
- Seed: 0
- Online optimization: 16 updates/frame
- Detector: BF30, detached anchor, strict `K=3`, directional margin 0
- Renderer: `OPEN union fixed zero-DC NEVER_OPEN`; CLOSED hidden
- Optimizer: 모든 OPEN row, sampled-view visibility와 무관; NEVER_OPEN frozen
- Loss: local support + `G_prev`, weight 7.5
- Density: ACTIVE-only O-SCD clone/split at local update 4; pruning disabled
- Output: `/tmp/escd_bf30_anchored_dc_agreement_u16_20260901_v1/`

## 4. 결과

| 방식 | mIoU | F1 | Precision | Recall | SC1 | SC2 | SC3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 BF30 learned-DC baseline | 0.4498 | 0.5814 | 0.6114 | 0.6586 | 0.4642 | 0.4902 | 0.3967 |
| learned agreement, DC adapt | 0.3399 | 0.4755 | 0.3451 | 0.8658 | 0.2630 | 0.3013 | 0.4478 |
| learned agreement, candidate DC freeze | 0.3078 | 0.4427 | 0.2905 | 0.9414 | 0.2650 | 0.2342 | 0.4194 |
| **detached anchor + strict K3** | **0.3684** | **0.5085** | **0.3753** | **0.8692** | **0.2699** | **0.3366** | **0.4892** |

| Lifecycle 진단 | agreement adapt | detached anchor + K3 | 변화 |
|---|---:|---:|---:|
| OPEN | 124,084 | 110,739 | -13,345 |
| CLOSE | 18,970 | 16,559 | -2,411 |
| REOPEN | 5,299 | 4,630 | -669 |
| Candidate commit | 143,054 | 127,298 | -15,756 |
| Same-scene transition | 13,133 | 11,398 | -1,735 |
| Final ACTIVE | 107,900 | 97,062 | -10,838 |

Candidate start/reject/commit은 각각 `2,754,243 / 2,439,222 / 127,298`이었다. 모든 commit
event의 candidate visible-observation count는 최소 3이었고, K3 미만 commit은 0이었다.

## 5. 가설 검증

### 5.1 Optimizer는 실제로 계속 동작했다

Live OPEN candidate에서 현재 learned `C_i`와 cue의 평균 gap은 한 frame optimization 동안

\[
0.713130 \rightarrow 0.710687
\]

로 `-0.002443` 감소했다. DC gradient max는 `0.00105679`였고 candidate DC gradient를
강제로 제거한 값은 0이었다. 즉 freeze ablation과 달리 representation 학습은 계속됐다.

### 5.2 Detector mismatch는 지워지지 않았다

동일 row의 detector-side anchor는 optimizer와 density update 전후에 정확히 drift 0이었다.
따라서 현재 `C_i`가 cue 방향으로 적응해도 이미 열린 candidate가 보는 기준은 변하지 않았다.

### 5.3 Multi-view gate는 부분적으로만 효과가 있었다

Agreement-adapt 대비:

- mIoU/F1: `+0.0285 / +0.0330`
- Precision: `+0.0302`
- Same-scene transition: `13,133 -> 11,398`
- Candidate commit: `143,054 -> 127,298`

따라서 self-erasure를 없애고 K3를 적용한 방향 자체는 freeze보다 낫고, 기존 agreement-adapt의
false transition도 일부 줄였다.

그러나 기존 BF30 learned-DC baseline보다는 mIoU/F1이 `-0.0814 / -0.0729` 낮다. 특히
SC1/SC2 precision이 낮고 recall이 과도하게 높아 over-opening 성향이 남았다.

## 6. 해석과 다음 판단

이번 결과는 다음 두 문제를 분리한다.

1. **Self-erasure는 실제 문제였고 detached anchor로 제거됐다.**
2. **하지만 동일 방향 mismatch 3회는 올바른 3D state transition의 충분조건이 아니다.**

Candidate start가 275만 회, reject가 243만 회라는 점은 adjacent view에서 동일 부호 cue가
반복돼도 많은 경우 transient misattribution임을 보여준다. 현재 K3는 Gaussian identity에
귀속된 cue 방향만 확인하며, 서로 다른 view에서 그 Gaussian이 동일한 changed surface를
실제로 설명하는지까지 검증하지 않는다. 따라서 다음 단계가 있다면 단순히 K를 키우기보다
view별 visibility/footprint overlap 또는 rendered responsibility 일관성을 candidate support에
포함하는 편이 타당하다.

## 7. 무결성

- CLOSED parameter/Adam drift: 0
- Inactive/wrong gradient violation: 0
- Future-view access: 0
- Detector anchor optimizer drift: 0
- Topology integrity: pass
- 전체 테스트: 470 passed
