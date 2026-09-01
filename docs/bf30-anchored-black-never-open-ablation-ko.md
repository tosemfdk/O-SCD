# BF30 anchored K3 + black NEVER_OPEN occluder ablation

## 1. 목적

기존 `open_or_never_open` renderer는 NEVER_OPEN Gaussian의 reference geometry와 opacity를
보존하여 OPEN change Gaussian 뒤의 transmittance를 가린다. 다만 change bank의 초기 raw
DC가 0이므로 FastGS degree-zero SH 변환 뒤 Gaussian 색은 RGB 0.5가 된다.

이번 ablation은 occlusion geometry/opacity는 동일하게 유지하면서 NEVER_OPEN의 change
color만 RGB 0으로 강제한다.

\[
C_i^{raw}=\operatorname{RGB2SH}(0)=-\frac{0.5}{C_0}
\]

따라서 NEVER_OPEN은 검은 occluder로 합성되지만 모든 parameter와 Adam state는 계속
freeze된다. OPEN은 learned DC/all-geometry를 학습하고 CLOSED는 완전히 숨긴다.

## 2. 구현 계약

- 새 render mode: `open_or_never_open_black`
- OPEN: 현재 learned DC와 현재 mutable geometry/opacity
- NEVER_OPEN: RGB-zero DC override + 초기 geometry/opacity, 전체 gradient detach
- CLOSED: opacity gate 0으로 렌더링 제외
- Detector: 기존 detached `C_i^{anchor}` BF30 + strict K3
- Detector alpha-T probe 자체에는 color를 직접 사용하지 않는다.
- 다만 black render가 optimizer gradient를 바꾸고, 갱신된 geometry/opacity가 다음 frame의
  alpha-T evidence를 바꾸므로 representation feedback은 존재한다.

## 3. 실험 조건

- Continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames
- Seed 0, 16 updates/frame
- BF30, detached candidate anchor, strict 3 observed views, directional margin 0
- 모든 OPEN row 학습, NEVER_OPEN freeze
- Local support + `G_prev`, weight 7.5
- ACTIVE-only O-SCD density update 4, pruning disabled
- 비교 간 의도된 설정 차이: `render_support_mode` 하나뿐
- Output: `/tmp/escd_bf30_anchored_black_never_open_u16_20260901_v1/`

## 4. 결과

| 방식 | mIoU | F1 | Precision | Recall | SC1 | SC2 | SC3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 BF30 learned-DC | 0.4498 | 0.5814 | 0.6114 | 0.6586 | 0.4642 | 0.4902 | 0.3967 |
| Anchored K3 + neutral NEVER_OPEN | 0.3684 | 0.5085 | 0.3753 | 0.8692 | 0.2699 | 0.3366 | 0.4892 |
| **Anchored K3 + black NEVER_OPEN** | **0.4481** | **0.5903** | **0.4847** | **0.9458** | **0.4447** | **0.4553** | **0.4441** |

Black override는 neutral 대비:

- mIoU: `+0.0797`
- F1: `+0.0818`
- Precision: `+0.1094`
- Recall: `+0.0766`
- Mean predicted-positive fraction: `0.1306 -> 0.1100`

기존 BF30 learned-DC baseline과 비교하면 mIoU는 `-0.0017`, F1은 `+0.0089`다.

## 5. Lifecycle 진단

| 진단 | Neutral | Black | 변화 |
|---|---:|---:|---:|
| OPEN | 110,739 | 192,907 | +82,168 |
| CLOSE | 16,559 | 55,848 | +39,289 |
| REOPEN | 4,630 | 16,965 | +12,335 |
| Candidate commit | 127,298 | 248,755 | +121,457 |
| Same-scene transition | 11,398 | 48,915 | +37,517 |
| Final ACTIVE | 97,062 | 139,848 | +42,786 |

따라서 mask 성능 향상을 lifespan detector 안정화로 해석하면 안 된다. 오히려 lifecycle
chattering은 크게 증가했다. Black occluder가 neutral 0.5 contribution을 제거하여 image-space
mask를 정리하는 동시에, 달라진 SSF gradient와 mutable geometry/opacity feedback이 더 많은
transition을 유발한 것으로 해석된다.

SC1/SC2는 크게 개선됐지만 SC3는 `0.4892 -> 0.4441`로 하락했다. 즉 black support는 누적
false-positive를 줄이는 데 유리하지만, 후반 state에서 필요한 visible change support도 일부
가릴 수 있다.

## 6. 결론

NEVER_OPEN을 neutral gray가 아니라 black occluder로 렌더링하는 것은 **현재 mask 품질에는
명확히 유효**했다. Anchored detector의 기존 성능 손실을 거의 전부 회복했고 F1은 기존 BF30
baseline을 소폭 넘었다.

하지만 lifecycle separation 관점에서는 실패가 남아 있다. 다음 비교에서는 black renderer를
유지하되 detector evidence가 optimizer로 변형된 mutable geometry/opacity에 얼마나 의존하는지
분리하거나, detector probe geometry를 immutable reference에 고정하는 control이 필요하다.

## 7. 무결성

- CLOSED parameter/Adam drift: 0
- Inactive/wrong gradient violation: 0
- Future-view access: 0
- Candidate anchor optimizer drift: 0
- K3 미만 commit: 0
- Topology integrity: pass
- Tests: 472 passed
