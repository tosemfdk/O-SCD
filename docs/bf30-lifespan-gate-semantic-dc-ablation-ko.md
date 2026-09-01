# BF30 lifespan-gate semantic DC ablation

## 1. 질문

기존 learned-DC detector는 Gaussian의 학습된 DC와 현재 change cue를 직접
비교하지 않는다. 그 결과 cue evidence가 active로 남아 있으면, 실제 DC가 검게
학습된 OPEN Gaussian도 CLOSE되지 않을 수 있다.

반대로 persistent DC가 1인 상태에서 CLOSE한 뒤 raw DC를 그대로 예상값으로
사용하면, cue가 다시 1일 때 `DC=1, cue=1`이라 surprise가 없어 REOPEN을 놓칠 수
있다. 이를 피하기 위해 raw DC 대신 **현재 lifespan gate 자체를 semantic change
color**로 쓰는 binary control을 검증했다.

## 2. Semantic change color

Gaussian `i`의 half-open lifespan interval들로 현재 상태를 정의한다.

$$
z_i(t)=
\begin{cases}
1,& t\text{가 현재 OPEN interval에 포함됨},\\
0,& \text{otherwise}.
\end{cases}
$$

이번 ablation의 per-Gaussian render color는 다음처럼 고정한다.

$$
C_i^{\mathrm{semantic}}(t)=z_i(t).
$$

- OPEN: precomputed RGB `(1,1,1)`
- NEVER_OPEN: precomputed RGB `(0,0,0)` occluder
- CLOSED: compositor에서 숨김

따라서 raw SH DC와 SH-rest는 renderer에서 완전히 우회된다. 저장된 parameter와
Adam state는 덮어쓰지 않으며, 실제 관측된 DC/SH-rest gradient는 정확히 `0`이다.
Mask의 연속값과 경계는 geometry, opacity, alpha-transmittance가 담당한다.

## 3. Flip/KEEP Beta evidence

기존 alpha-T VJP가 만드는 cue-positive/negative capped pseudo-count를 그대로
사용한다.

```text
delta_positive: Gaussian footprint의 cue=change 책임
delta_negative: Gaussian footprint의 cue=non-change 책임
```

단, Beta의 의미를 active/inactive에서 **현재 lifespan bit의 FLIP/KEEP**으로
변경한다.

$$
\Delta a_i^{\mathrm{flip}}
=(1-z_i)\Delta_i^{+}+z_i\Delta_i^{-},
$$

$$
\Delta b_i^{\mathrm{keep}}
=(1-z_i)\Delta_i^{-}+z_i\Delta_i^{+}.
$$

즉 다음과 같다.

| 현재 bit | cue-positive | cue-negative |
|---|---|---|
| CLOSED `z=0` | FLIP→OPEN 후보 | KEEP CLOSED |
| OPEN `z=1` | KEEP OPEN | FLIP→CLOSE 후보 |

Committed validity prior와 fresh reset candidate prior는 각각 다음과 같다.

```text
stable: Beta(flip=1, keep=10)
reset candidate: Beta(flip=1, keep=1)
commit threshold: BF >= 30
```

Candidate가 살아 있는 동안 stable Beta는 freeze한다. Candidate marginal
likelihood가 stable KEEP 설명을 BF30 이상으로 이기면 lifecycle bit를 토글한다.
토글 전 FLIP evidence는 토글 후 새 상태의 KEEP evidence이므로, candidate block의
FLIP/KEEP 축을 뒤집어 새 stable Beta로 승격한다.

```text
CLOSED + sustained positive cue -> OPEN
OPEN + sustained negative cue   -> CLOSE
CLOSED + positive cue again     -> REOPEN
```

## 4. Paired run 조건

두 조건은 `detector_mode`와 `change_color_mode` 이외의 인자, source PLY hash,
fixed-camera hash가 모두 같다.

| 항목 | 값 |
|---|---|
| stream | 연속 `ref -> SC1 -> SC2 -> SC3`, 304 frames |
| seed / updates | seed 0 / frame당 16 updates |
| Bayes factor | 30 |
| renderer support | OPEN + 고정 NEVER_OPEN occluder, CLOSED hidden |
| optimizer | 모든 OPEN row 선택, NEVER_OPEN freeze |
| loss | local cue support + `G_prev`, `lambda_g=7.5` |
| density | ACTIVE-only O-SCD clone/split, prune off |

비교 조건:

```text
learned_dc_baseline:
  cue-distribution single-candidate Beta + learned persistent DC

lifespan_gate:
  flip/KEEP single-candidate Beta + binary lifespan semantic color
```

## 5. Mask 결과

| 지표 | learned DC | lifespan gate | gate - baseline |
|---|---:|---:|---:|
| mean-frame mIoU | **0.4498** | 0.4264 | -0.0234 |
| mean-frame F1 | **0.5814** | 0.5698 | -0.0116 |
| precision | **0.6114** | 0.4516 | -0.1598 |
| recall | 0.6586 | **0.9570** | +0.2984 |
| SC1 mIoU | **0.4642** | 0.4412 | -0.0230 |
| SC2 mIoU | **0.4902** | 0.4371 | -0.0531 |
| SC3 mIoU | 0.3967 | **0.4023** | +0.0056 |

Semantic gate는 거의 모든 change pixel을 덮으면서 recall을 `+0.2984` 높였지만,
false-positive 영역도 크게 증가했다. 전체 predicted-positive fraction 평균은
`0.0607 -> 0.1195`로 약 두 배가 되었다.

Frame 최적화 전후의 mean-frame mIoU 개선은 다음과 같다.

```text
learned DC:    0.4185 -> 0.4498  (+0.0313)
lifespan gate: 0.4144 -> 0.4264  (+0.0119)
```

Hard color가 DC 학습 자유도를 없앴기 때문에 geometry/opacity optimization만으로
cue 경계를 교정하는 능력도 더 낮았다.

## 6. Lifecycle 결과

| 지표 | learned DC | lifespan gate | 차이 |
|---|---:|---:|---:|
| OPEN | 43,406 | 222,490 | +179,084 |
| CLOSE | 1,990 | 71,270 | +69,280 |
| REOPEN | 441 | 22,928 | +22,487 |
| candidate commit | 6,299 | 293,760 | +287,461 |
| same-scene repeated events | 712 | 67,715 | +67,003 |
| final ACTIVE | 44,437 | 154,068 | +109,631 |

REOPEN이 `22,928`회 발생했으므로, retained raw DC 때문에 CLOSE 후 다시 열리지
않는 blind spot은 사라졌다. 그러나 해결 방식이 지나치게 민감했다.

첫 OPEN 뒤 첫 CLOSE latency도 크게 달라졌다.

| 진단 | learned DC | lifespan gate |
|---|---:|---:|
| first OPEN 수 | 42,965 | 199,562 |
| 이후 CLOSE | 1,107 | 57,915 |
| median CLOSE latency | 89 frames | 9 frames |
| 5 frames 이내 CLOSE | 0 | 18,006 |

빠르게 잘못 OPEN을 닫는 능력은 생겼지만, 대부분의 전이가 동일 scene 내부의
반복 toggle로 나타났다.

## 7. Chattering 원인

한 Gaussian의 cue-positive 비율은 실제 물체 상태가 같아도 view마다 변한다.

- foreground occlusion
- elongated Gaussian footprint
- change/non-change 경계 횡단
- alpha-T 경쟁 순서
- mutable geometry/opacity 변화

Learned-DC baseline의 candidate reset은 cue 분포가 바뀌어도 새 posterior가 여전히
active이면 `active -> active KEEP`으로 처리한다. 반면 이번 gate detector는
candidate commit 하나를 **무조건 binary bit toggle**로 해석한다.

```text
view A: OPEN Gaussian이 cue-positive를 많이 받음 -> KEEP
view B: 같은 Gaussian이 cue-negative를 많이 받음 -> CLOSE
view C: 다시 cue-positive -> REOPEN
```

이 때문에 candidate commit이 `6,299 -> 293,760`, same-scene event가
`712 -> 67,715`로 폭증했다. BF30만으로는 오래 누적된 stable Beta가 view-local
contradiction에 과민해지는 문제를 막지 못했다.

## 8. DC와 계산 특성

| 진단 | learned DC | lifespan gate |
|---|---:|---:|
| 최종 OPEN color `<0.5` | 33.06% | **0%** |
| DC gradient max | 0.003250 | **0** |
| SH-rest gradient max | 0 | 0 |
| runtime | 176.0 s | 215.3 s |
| peak CUDA memory | 7.47 GiB | **6.23 GiB** |

Semantic gate는 정의상 dark OPEN을 완전히 제거하고 SH gradient memory를 줄였다.
하지만 평균 ACTIVE row가 `35,580 -> 78,674`, 최종 ACTIVE가 3.47배 증가하여
runtime은 약 22% 증가했다.

## 9. 결론

이번 control은 두 질문에 명확히 답한다.

1. **CLOSE 후 REOPEN이 가능한가?** 가능하다. Lifespan gate가 CLOSED 예상값을
   항상 0으로 만들기 때문에 stored raw DC=1과 무관하게 cue=1이 새 mismatch가 된다.
2. **Binary semantic gate를 그대로 최종 detector/renderer로 쓸 수 있는가?** 아니다.
   Dark OPEN은 없애지만 view-dependent cue 책임을 state transition으로 오인하여
   chattering, false positive, ACTIVE inflation을 만든다.

따라서 이 raw binary gate는 learned-DC BF30 기본값을 대체하지 않는다. Gate
consistency를 후속으로 사용할 경우에는 단일 view의 mismatch를 즉시 toggle하지
말고, committed state에 대한 multi-view 확인이나 visibility/occlusion-aware
agreement를 먼저 적용해야 한다.

## 10. 구현·검증·산출물

- flip/KEEP detector/controller: `temporal/lifespan_gate_beta.py`
- semantic renderer integration: `experiments/run_online_dynamic_active_oscd_density.py`
- unit tests: `tests/temporal/test_lifespan_gate_beta.py`
- runner tests: `tests/experiments/test_online_dynamic_active_oscd_density.py`
- targeted tests: `28 passed`
- full repository tests: `454 passed`
- CUDA smoke: 3-frame, 16-frame 통과
- 두 full run 모두:
  - CLOSED parameter/Adam drift `0`
  - inactive gradient violation `0`
  - future-view access `0`
  - topology integrity pass

```text
/tmp/escd_bf30_lifespan_gate_ablation_20260901_v1/
  learned_dc_baseline/
    summary.json
    frame_metrics.csv
    lifecycle_events.jsonl
  lifespan_gate/
    summary.json
    frame_metrics.csv
    lifecycle_events.jsonl
  comparison.json
  comparison.csv
  comparison.md
  comparison_curves.png
```
