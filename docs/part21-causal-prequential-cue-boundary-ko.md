# Part 21: Causal prequential Stage-2 cue boundary

## 1. 문제 수정

기존 `outputs/stage2_l1_power_sigmoid_boundary/learned_boundaries.json`은 전체
304-frame stream을 먼저 train/validation으로 나눠 학습했다. 따라서 미래 frame이
앞선 frame의 `tau,width`에 영향을 주며 online 실험에 사용할 수 없다.

이번 수정에서는 frame `t`를 다음 순서로 처리한다.

```text
1. theta_(t-1)와 현재 cue histogram_t로 tau_t,width_t를 예측한다.
2. Q_t를 확정하고 현재 frame 처리에 사용한다.
3. 현재 출력이 확정된 뒤 teacher_t를 공개한다.
4. teacher_t로 theta_t를 한 번 업데이트한다.
5. theta_t는 frame t+1 이후에만 사용한다.
```

첫 frame은 학습 전 고정값 `tau=0.25`, `width=0.10`을 사용한다. 과거에 말한
`40 epoch` warm-up은 제거했다. 현재 causal runner의 `40`은 epoch 수가 아니라
처음 40개 도착 frame 동안 histogram trunk를 고정하고 작은 boundary head만
점진적으로 업데이트하는 frame 단위 warm-up이다.

## 2. 구현과 인과성 감사

- 구현: `experiments/train_stage2_cue_boundary_causal.py`
- 테스트: `tests/experiments/test_train_stage2_cue_boundary_causal.py`
- artifact:
  `outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/learned_boundaries_causal.json`

감사 결과:

- future teacher access: `0`
- current teacher used for current prediction: `false`
- frame 0에서 prediction 전에 볼 수 있는 마지막 teacher index: `-1`
- frame `t`에서 prediction 전에 볼 수 있는 마지막 teacher index: `t-1`
- 마지막 frame teacher를 바꿔도 이미 출력된 모든 frame의 `tau,width`가 동일함을
  regression test로 확인했다.

## 3. Cue-only replay 결과

아래 수치는 representation을 다시 학습한 결과가 아니라 저장된 histogram으로
계산한 근사 cue-only GT diagnostic이다. GT는 학습과 boundary 선택에 쓰지 않았다.

| Metric | 고정 tau=0.25 | Causal prequential | Delta |
|---|---:|---:|---:|
| Mean-frame IoU | 0.5871 | 0.6073 | +0.0202 |
| Mean-frame F1 | 0.7094 | 0.7259 | +0.0165 |
| Aggregate IoU | 0.6119 | 0.6398 | +0.0279 |
| Aggregate precision | 0.6263 | 0.6580 | +0.0317 |
| Aggregate recall | 0.9637 | 0.9586 | -0.0051 |

예측된 값의 범위는 다음과 같다.

- `tau`: mean `0.2673`, min `0.1947`, max `0.3444`
- `width`: mean `0.1130`, min `0.0719`, max `0.1788`

이 결과는 전체 stream post-hoc 학습 결과를 대체하는 올바른 online replay다. 다만
teacher는 저장된 current-frame post-update O-SCD mask이므로 실제 배포에는 동등한
causal teacher 또는 별도 self-supervised signal이 필요하다.

## 4. 해석 범위

현재 결과로 확인된 것은 과거와 현재만 사용하는 online `tau,width` 갱신이
가능하며 cue-only histogram metric이 고정 boundary보다 좋아졌다는 점이다.
Part 20의 DA3 u120 결과는 미래를 본 post-hoc boundary artifact를 사용했으므로
여전히 deployment-causal 성능으로 해석하지 않는다. DA3 representation 성능을
주장하려면 이 causal artifact로 u120을 다시 실행해야 한다.
