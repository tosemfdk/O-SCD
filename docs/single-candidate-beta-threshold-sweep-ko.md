# Single-candidate Beta changepoint threshold sweep

## 목적

Gaussian마다 full BOCD run-length posterior를 유지하지 않고 다음의 O(1)
상태만으로 evidence distribution reset을 검출할 수 있는지 확인했다.

- 확정된 현재 run의 `stable Beta(a, b)`
- 하나의 미확정 `candidate` 구간에 누적된 fractional pseudo-count `(A, B)`
- candidate 시작 timestamp

이 실험은 `h = ln(10), ln(30), ln(100), ln(300)`을 비교한다. CLI에는
각각 ordinary Bayes factor `10, 30, 100, 300`을 주고, detector 내부에서
한 번만 로그로 변환했다.

## detector

Candidate block 전체의 정확한 Beta marginal을 두 가설 아래에서 비교한다.

```text
log_keep  = log B(stable_a + A, stable_b + B) - log B(stable_a, stable_b)
log_reset = log B(1 + A, 1 + B) - log B(1, 1)
score     = log_reset - log_keep
```

정책은 다음과 같다.

1. 첫 observed evidence는 changepoint 없이 stable Beta를 초기화한다.
2. candidate가 없고 `score > 0`이면 candidate를 시작한다.
3. candidate가 살아 있는 동안 stable Beta는 freeze한다.
4. `score <= 0`이면 KEEP으로 판정하고 candidate block 전체를 stable에
   병합한다.
5. `score >= h`이면 RESET으로 확정하고 `Beta(1 + A, 1 + B)`를 새 stable
   run으로 승격한다.
6. reset 전후 binary label이 같으면 Beta run만 교체하고 기존 lifespan
   slot은 유지한다. `active -> active`는 `KEEP`이다.
7. hazard, Markov transition probability, run-length posterior, decay,
   concentration cap은 사용하지 않는다.

### concentration 반영 확인

`Beta(10,1)`과 `Beta(100,10)`은 평균이 같다. 따라서 첫 failure의 predictive
probability도 둘 다 `1/11`로 같다. 하지만 failure 두 개의 block marginal은
다르다.

```text
Beta(10,1):   P(FF) = (1*2)/(11*12)   = 1/66
Beta(100,10): P(FF) = (10*11)/(110*111) = 1/111
fresh Beta(1,1): P(FF) = 1/3
```

따라서 두 failure 뒤 RESET-vs-KEEP BF는 각각 `22`와 `37`이다. 즉 이
구현은 첫 관측에서는 평균만 보지만 반복 관측의 block marginal을 통해
Beta concentration 차이를 반영한다. 이 동작은 단위 테스트로 고정했다.

## 실험 조건

- stream: continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames
- shared mutable Gaussian bank: detector와 mask optimizer가 동일한 현재 GS를 봄
- detector probe scaling: native
- render support: `open_or_never_open_dc_opacity`
- density: active-only original O-SCD clone/split, local update 4
- online updates: 16/image
- replay: current view probability `0.33`, 아니면 이미 처리된 view에서 uniform
- pruning: disabled
- seed: 0
- threshold 외 조건은 direct-binary continuous baseline과 동일
- 모든 산출물은 `/tmp`에 기록

## 결과

### mask 품질

| detector | SC1 mIoU | SC2 mIoU | SC3 mIoU | overall mIoU | overall F1 |
|---|---:|---:|---:|---:|---:|
| direct binary baseline | 0.6228 | 0.5232 | 0.3175 | **0.4833** | **0.6028** |
| `h=ln(10)` | 0.5411 | 0.3466 | **0.3303** | 0.4018 | 0.5085 |
| `h=ln(30)` | 0.5524 | **0.4421** | 0.3247 | **0.4360** | **0.5466** |
| `h=ln(100)` | 0.3886 | 0.1434 | 0.0001 | 0.1705 | 0.2074 |
| `h=ln(300)` | 0.5329 | 0.3588 | 0.2968 | 0.3918 | 0.4999 |

Single-candidate 방식 중에는 `h=ln(30)`이 가장 좋았지만 direct baseline보다
overall mIoU/F1이 `-0.0473/-0.0561` 낮았다. BF30의 aggregate
precision은 `0.7345`로 baseline `0.6831`보다 높았지만 recall은 `0.5530`으로
baseline `0.6397`보다 낮았다. 즉 더 보수적인 mask가 되었고 유효 change를
많이 놓쳤다.

### lifecycle과 candidate

| detector | OPEN | CLOSE | REOPEN | same-scene repeated | candidate START | REJECT | COMMIT | final live |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct binary | 55,886 | 13,954 | 3,295 | 9,007 | - | - | - | - |
| `h=ln(10)` | 58,351 | 4,799 | 738 | 1,820 | 669,556 | 435,447 | 14,250 | 220,437 |
| `h=ln(30)` | 56,106 | 3,105 | 365 | 765 | 719,692 | 480,187 | 10,109 | 230,685 |
| `h=ln(100)` | 50,189 | 1,359 | 26 | 165 | 525,478 | 323,305 | 4,447 | 198,172 |
| `h=ln(300)` | 52,072 | 1,772 | 65 | 152 | 713,095 | 472,478 | 5,069 | 236,723 |

표의 same-scene repeated는 동일 binary label의 Beta reset `KEEP`을 제외하고
실제 `OPEN/CLOSE` representation mutation만 센 값이다. BF30은 baseline 대비
CLOSE, REOPEN, same-scene repeated event를 크게 줄였다.
그러나 transition 억제가 mask 정확도 개선으로 이어지지는 않았다. 특히 각
run 마지막에 약 20만~24만 candidate가 미확정 상태로 남았다.

## 해석

1. **O(1) detector 자체는 동작한다.** Stable Beta freeze, candidate reject
   merge, reset promotion, same-label slot preservation과 dynamic topology
   state 복제 invariant가 모두 유지됐다.
2. **순수 Beta 누적은 지나치게 보수적이다.** Unbounded stable concentration과
   높은 reset threshold가 결합되어 precision은 오르지만 recall이 크게
   떨어졌다.
3. **단일 candidate가 장기간 미확정으로 남는다.** `score > 0`이지만 `h`에는
   도달하지 않는 row가 많이 쌓였다. Hazard/prior odds가 없는 현재 설계에는
   candidate 지속 자체에 대한 확률 비용이 없다.
4. **threshold 효과는 단조롭지 않다.** 이 runner에서는 detector가 결정한
   ACTIVE set이 이후 parameter optimization과 densification을 바꾸고, 그
   mutable GS가 다음 detector evidence를 다시 바꾼다. 따라서 BF100과 BF300의
   차이는 동일 evidence trace에 threshold만 적용한 결과가 아니라 서로 다른
   causal representation trajectory의 결과다.
5. **chattering 감소만으로 충분하지 않다.** BF30은 repeated transition을
   크게 줄였지만 SC1/SC2 표현 학습과 recall을 잃었다. 현재 blocker는 단순한
   reset threshold 선택만으로 해결되지 않는다.

## invariant

네 threshold 모두 다음을 만족했다.

- CLOSED parameter/Adam drift: 0
- inactive/off-view gradient violation: 0
- dynamic topology alignment: pass
- future-view replay access: 0 (causal loop에서 즉시 검사)
- GT 사용: causal loop 이후 평가에서만 사용

## 산출물

- sweep root:
  `/tmp/escd_single_candidate_beta_h_sweep_continuous_u16_seed0_20260826/`
- per-run summary/raw/mask/event capture: `bf10/`, `bf30/`, `bf100/`, `bf300/`
- comparison JSON/CSV/plot: `report/`
- raw R_change + thresholded R_change + OPEN/CLOSE timeline videos: `videos/`
