# View-consistent direct binary lifespan confirmation 실험

## 1. 질문과 결론

이 실험은 direct binary filter의 marginal belief `p_active`와 representation
lifecycle decision을 분리한다. PASLCD D1에서 동일한 physical post-change scene 안에서도
`OPEN -> CLOSE -> REOPEN`이 반복된 원인은 low mass 하나가 아니라 view-conditioned cue
변동과 `p_active` hysteresis의 직접 결합이었다.

첫 ablation은 view diversity, adaptive emission, geometry 변경을 넣지 않고 다음 한 가지만
검증한다.

> 현재 committed state를 뒤집는 observation Bayes factor가 여러 observed view에서
> 연속으로 지지될 때만 OPEN/CLOSE하면 single-scene chattering을 줄이면서 ESCD의 실제
> 변화에는 반응할 수 있는가?

선택한 `K=3, BF>=3`은 PASLCD repeated transition extras를 `556,469 -> 43,153`
으로 줄였다. 동일 lifecycle을 사용하는 all-geometry 표현은 PASLCD O-SCD online보다
mean-frame mIoU가 `+0.0086` 높았다. DC-only는 `-0.0419` 낮아, 안정적인 detector와
새 zero-init slot의 표현 지연은 별개의 문제임을 확인했다.

## 2. Causal contract

Detector 입력은 계속 immutable reference alpha-T evidence뿐이다.

```text
current RGB + immutable R_ref
        -> cached O-SCD cue
        -> immutable-reference alpha-T evidence
        -> direct binary Bayesian filter
        -> view-confirmed lifecycle decision
        -> current OPEN slot render/optimization
```

Temporal DC/xyz/opacity/scaling/rotation과 lifespan visibility는 evidence renderer에
feedback되지 않는다. GT와 manual boundary는 causal loop가 끝난 뒤 평가/plot에만 쓴다.

## 3. Transition confirmation 수식

Direct filter가 현재 observation에 대해 계산한 joint posterior를
`P00, P01, P10, P11`이라 한다. 현재 representation state가 inactive이면:

\[
O_{open}=\frac{P_{01}}{P_{00}},\qquad
BF_{open}=\frac{O_{open}}{p_{01}/(1-p_{01})}.
\]

현재 representation state가 active이면:

\[
O_{close}=\frac{P_{10}}{P_{11}},\qquad
BF_{close}=\frac{O_{close}}{p_{10}/(1-p_{10})}.
\]

즉 Markov transition prior odds를 제거하고 현재 observation의 flip-vs-stay likelihood
ratio만 확인한다. `p_active`는 저장하지만 lifecycle을 직접 flip하지 않는다.

선택 설정은 `eta=0.9`, capped strength `w<=1`, `BF>=3`, 연속 observed support
`K=3`이다. `w=1`이면 OPEN은 대략 `q>=0.75`, CLOSE는 `q<=0.25`에 해당한다.

- unobserved 또는 transition quality threshold 미만: counter와 lifecycle을 `HOLD`
- quality-valid non-support: 해당 consecutive counter reset
- inactive에서 K회 OPEN support: 새 zero-init slot `OPEN`
- active에서 K회 CLOSE support: 현재 slot `CLOSE`
- active에서 close 미확정: 같은 slot `KEEP`
- closed slot 재사용 금지, active-to-active split 금지

## 4. 구현

- `temporal/view_consistent_binary_lifespan_controller.py`
  - committed-state branch BF와 consecutive counter
  - controller state checkpoint round-trip
  - zero-init/open/close/reopen lifecycle 재사용
- `experiments/replay_paslcd_transition_confirmation.py`
  - D1 sparse NPZ만 읽는 GT-free offline policy replay
- `experiments/run_online_binary_state_lifespan_thaw.py`
  - 기존 `posterior_hysteresis` default 보존
  - opt-in `--lifecycle-controller view_consistent`
  - BF/support counter event diagnostics

## 5. PASLCD offline policy replay

20 scenes, 500 frames의 동일 D1 evidence를 `K={1,2,3}`, `BF={1,3}`으로 replay했다.

| condition | OPEN | CLOSE | REOPEN | repeated extras | final active |
|---|---:|---:|---:|---:|---:|
| 기존 posterior hysteresis | 922,714 | 446,622 | 109,847 | 556,469 | 476,092 |
| K=1, BF>=1 | 3,906,023 | 3,347,980 | 1,827,332 | 5,175,312 | 558,043 |
| K=1, BF>=3 | 1,813,888 | 1,290,570 | 533,161 | 1,823,731 | 523,318 |
| K=2, BF>=1 | 1,131,043 | 726,616 | 216,745 | 943,361 | 404,427 |
| K=2, BF>=3 | 493,138 | 176,997 | 44,311 | 221,308 | 316,141 |
| K=3, BF>=1 | 511,019 | 210,728 | 40,849 | 251,577 | 300,291 |
| **K=3, BF>=3** | **234,891** | **34,898** | **8,255** | **43,153** | **199,993** |

`K=1`은 transition odds만으로는 single-view noise를 오히려 증폭했다. `K=3, BF>=3`은
repeated extras를 `92.2%` 줄여 약 `12.9x`, REOPEN을 약 `13.3x` 줄였다.

## 6. PASLCD 500-frame / 120-update 표현 비교

20 scenes/500 frames에서 B0/B1은 detector, cue, pose, seed, update schedule이 같고 thaw
parameter만 다르다. Lifecycle event hash는 전 20 scene에서 정확히 일치했다.

| condition | mIoU | F1 | precision | recall | OPEN/CLOSE/REOPEN | repeated extras |
|---|---:|---:|---:|---:|---:|---:|
| O-SCD online | 0.4887 | 0.6423 | - | - | - | - |
| 기존 direct B0 DC | 0.5008 | 0.6524 | 0.6574 | 0.6872 | 922,714/446,622/109,847 | 556,469 |
| 기존 direct B1 all geometry | 0.5147 | 0.6573 | 0.5569 | 0.8350 | 922,714/446,622/109,847 | 556,469 |
| K3/BF3 B0 DC | 0.4468 | 0.5867 | 0.7348 | 0.5685 | 234,891/34,898/8,255 | 43,153 |
| **K3/BF3 B1 all geometry** | **0.4973** | **0.6286** | **0.6150** | **0.7499** | **234,891/34,898/8,255** | **43,153** |

K3/BF3 B1은 O-SCD online 대비 mIoU `+0.0086`, F1 `-0.0138`이다. 기존 chattering
B1보다 mIoU는 `-0.0174` 낮지만 repeated transition은 크게 줄었다.

OPEN frame의 평균 pre/post optimization mIoU 개선은 B0 `+0.0558`, B1 `+0.1079`였다.
Detector가 같은데 B0만 크게 떨어진 것은 conservative OPEN과 zero-init slot의 lag를
DC만으로 120 update 안에 회복하기 어렵다는 representation failure다. All geometry는 더
큰 표현 자유도로 상당 부분을 회복하지만 detector correctness를 바꾸지는 않는다.

모든 PASLCD run에서 base drift, closed-slot drift, inactive gradient violation,
active-to-active split, reused-slot violation은 0이었다.

## 7. ESCD `ref -> sc1 -> sc2 -> sc3`

동일한 fixed camera/cue/seed, binary capped evidence, 120 updates/frame, `max_states=16`으로
304 frames를 causal order로 처리했다.

| condition | mIoU | F1 | precision | recall | OPEN | CLOSE | REOPEN | repeated extras |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 direct B0 DC | 0.6063 | 0.7283 | 0.7162 | 0.8817 | 299,673 | 178,355 | 78,731 | 170,574 |
| K3/BF3 B0 DC | 0.5981 | 0.7201 | 0.7273 | 0.8673 | 123,476 | 64,619 | 22,212 | 51,434 |
| K3/BF3 B1 all geometry | 0.5996 | 0.7204 | 0.6847 | 0.9251 | 123,476 | 64,619 | 22,212 | 51,434 |
| Beam-2 B1 all geometry | 0.6140 | 0.7323 | 0.6856 | 0.9297 | 153,458 | 39,852 | 10,041 | 19,305 |

K3/BF3 B0는 기존 direct B0 대비 mIoU `-0.0082`로 거의 유지하면서 OPEN/CLOSE/REOPEN을
각각 약 `58.8%/63.8%/71.8%` 줄였다. 동일 scene segment 안의 repeated extras도
`170,574 -> 51,434`로 `69.8%` 줄었다.

K3/BF3 B1은 같은 lifecycle에서 mIoU/F1 `0.5996/0.7204`였다. B0보다 mIoU는
`+0.0015` 높고, precision/recall은 `0.7273/0.8673 -> 0.6847/0.9251`로 이동했다.
즉 ESCD에서는 geometry가 전체 score를 크게 올리기보다 recall을 높이는 대신 precision을
낮췄다. Segment별 B1 mIoU는 SC1/SC2/SC3 `0.5546/0.6132/0.6269`이다.

Beam-2 all-geometry는 이 ESCD 조건에서 mIoU/F1과 same-segment repeated extras가 모두
더 좋았다. 따라서 view-confirmed direct filter는 PASLCD failure 원인과 confirmation 효과를
검증한 독립 ablation이지, Beam-2를 대체하는 최종 detector가 아니다.

첫 전환 뒤에는 새로운 OPEN이 frame `97`에서 크게 증가해 boundary `95` 대비 약 2-frame
confirmation delay가 관찰됐다. 두 번째 전환 주변에도 OPEN/CLOSE가 모두 발생했다. 이는
PASLCD에서 안정성만 높이고 ESCD 전환을 완전히 죽인 설정은 아님을 보여준다. 다만
per-Gaussian true transition label이 없으므로 이 수치를 정확한 detection-delay metric으로
과장하지 않는다.

ESCD B0/B1 모두 base/closed-slot/inactive-gradient/slot-reuse/active-to-active-split drift와
violation은 전부 0이었다. 한 Gaussian의 최대 allocated episode 수는 8이었다. Runtime과
peak CUDA allocated memory는 B0 `495.1 s / 3.71 GB`, B1 `881.6 s / 7.21 GB`였다.

## 8. 결론과 다음 단계

1. D1의 핵심 가설은 맞았다. `p_active` marginal threshold와 lifecycle mutation을 분리하면
   PASLCD false transition을 order-of-magnitude로 줄일 수 있다.
2. 단일 branch-odds frame(`K=1`)은 부족하고 consecutive observed support가 필요하다.
3. 안정성 비용은 under-open/zero-init lag로 나타나며, DC-only보다 all-geometry에서 회복이
   크다. 이는 detector와 representation을 분리해 보고해야 한다.
4. PASLCD acceptance와 ESCD actual-transition response의 첫 gate를 통과했으므로 원인 분리를
   위해 이번 checkpoint에 view diversity와 adaptive emission을 동시에 추가하지 않는다.
5. 남은 문제는 ESCD same-segment repeated extras `51,434`이다. 다음 ablation이 필요하면
   OPEN/CLOSE confirmation을 비대칭으로 분리하거나 view diversity를 추가하되, 현재 결과를
   frozen baseline으로 둔다.

## 9. 실행과 산출물

```bash
PYTHONPATH=. python -m experiments.replay_paslcd_transition_confirmation \
  outputs/paslcd_d1_view_consistency_diagnostic \
  --output-dir outputs/paslcd_d2_transition_confirmation_replay

PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_paslcd_binary_state_benchmark \
  --output-root outputs/paslcd_view_consistent_k3bf3_benchmark_res4 \
  --update-budgets 120 -- \
  --lifecycle-controller view_consistent \
  --transition-confirmation-views 3 \
  --min-transition-bayes-factor 3
```

주요 산출물:

- `outputs/paslcd_d2_transition_confirmation_replay/summary.json`
- `outputs/paslcd_view_consistent_k3bf3_benchmark_res4/comparison.json`
- `outputs/escd_view_consistent_k3bf3_b0_dc_u120_20260821/summary.json`
- `outputs/escd_view_consistent_k3bf3_b1_all_geometry_u120_20260821/summary.json`

`outputs/`의 checkpoint, CSV, JSON, NPZ, PNG는 Git에 커밋하지 않는다.
