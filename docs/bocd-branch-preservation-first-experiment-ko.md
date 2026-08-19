# BOCD reset-branch 보존 1차 실험

## 1. 이번 변경의 목적

연속 `Ref -> SC1 -> SC2 -> SC3` 실행에서 현재 `MAPResetBernoulliFilter`는
`OPEN=81,268`, `CLOSE=0`, `REOPEN=0`이었다. 이번 단계에서는 geometry와 renderer를
더 바꾸지 않고 다음 한 가지만 분리한다.

> 한 프레임에서 reset posterior가 threshold보다 낮더라도 reset 후보를 버리지 않고
> 다음 frame까지 유지하면, capped unit evidence에서도 `CLOSE/REOPEN`이 생기는가?

이 질문이 가장 먼저인 이유는 다음과 같다.

- Bayesian evidence는 immutable reference에서 계산되므로 state-local geometry가 detector
  입력을 직접 바꾸지 않는다.
- 현재 MAP 방식은 threshold 아래 reset branch를 즉시 폐기한다.
- controller는 accepted reset 없는 label 반전을 `UNCERTAIN`으로 유지한다.
- 따라서 branch 보존만 바꾼 detector-only 비교가 현재 `CLOSE=0` 원인을 가장 적은
  변수로 검증한다.

기존 production runner는 수정하지 않는다. 새 detector와 실험 adapter를 별도 파일로
추가하여 baseline 동작과 checkpoint 호환성을 그대로 보존한다.

## 2. 추가 파일

### `temporal/beam2_bocd.py`

`BeamTwoBernoulliFilter`를 추가한다. Gaussian마다 다음 두 hypothesis만 유지한다.

```text
incumbent branch: 현재 확정된 run
candidate branch: 지금까지 살아남은 가장 강한 reset 후보
```

관측 `(s_t, f_t)`가 들어오면 세 후보를 계산한다.

```text
1. incumbent growth
2. 기존 candidate growth
3. 현재 frame에서 새 reset candidate spawn
```

로그 가중치는 다음과 같다.

\[
\ell_{g,0}=\log w_0+\log(1-H)+\log p(s_t,f_t\mid a_0,b_0)
\]

\[
\ell_{g,1}=\log w_1+\log(1-H)+\log p(s_t,f_t\mid a_1,b_1)
\]

\[
\ell_{r}=\log\sum_k w_k+\log H+\log p(s_t,f_t\mid a_{prior},b_{prior})
\]

기존 candidate growth와 새 reset spawn 중 강한 하나만 candidate slot에 남기고,
incumbent와 candidate를 정규화한다. Candidate posterior가
`changepoint_probability` 이상이면 그 candidate를 incumbent로 commit한다.

이 방식은 full exact BOCD가 아니다. 그러나 threshold 아래 reset branch를 폐기하지
않는다는 핵심 차이를 O(N) 메모리로 직접 검사한다.

Serialized persistent state는 Gaussian당 float vector 9개와 int64 vector 8개다.
Instance_1의 `N=1,283,501`, float32 기준 예상 BOCD state는 약 `0.120 GiB`다.
현재 MAP estimator 약 `0.067 GiB`보다 크지만 bounded exact `R=128`의 약
`4.97 GiB`보다 훨씬 작다.

### `experiments/run_bocd_branch_preservation_smoke.py`

Renderer 없이 한 Gaussian에 capped evidence 최대치인 frame당 총 mass `1.0`을 넣는다.
Sequence는 다음과 같다.

```text
ACTIVE 40 frames -> INACTIVE 40 frames -> ACTIVE 40 frames
H = 0.01
CP threshold = 0.5
OPEN/CLOSE thresholds = 0.6/0.4
```

동일 controller를 사용해 실제 `OPEN/CLOSE/REOPEN` lifecycle action까지 검사한다.

실행:

```bash
PYTHONPATH=. python -m experiments.run_bocd_branch_preservation_smoke \
  --output-dir outputs/bocd_branch_preservation_smoke
```

생성 파일:

```text
summary.json
frame_metrics.csv
lifecycle_events.jsonl
```

### `experiments/run_online_bocd_branch_ablation.py`

기존 `run_online_bayesian_lifespan_thaw.py`를 수정하지 않고, 실험 실행 동안에만 다음
세 hook을 임시로 확장한다.

```text
validate_run_config: beam2 허용
estimated_bocd_state_bytes: beam2 메모리 계산
make_bocd_filter: beam2 factory 추가
```

`finally`에서 세 함수를 모두 원래 객체로 복원한다. 두 mode는 detector-only이며
pose, cue, alpha-T evidence, threshold, stream order가 동일하다.

실행:

```bash
PYTHONPATH=. python -m experiments.run_online_bocd_branch_ablation \
  --output-root outputs/escd_bocd_branch_ablation -- \
  --source-path data/Instance_1/scene_change1_2_3
```

빠른 detector-only 확인에서는 post-inference GT 평가를 생략할 수 있다.

```bash
PYTHONPATH=. python -m experiments.run_online_bocd_branch_ablation \
  --output-root outputs/escd_bocd_branch_ablation_noeval -- \
  --source-path data/Instance_1/scene_change1_2_3 \
  --skip-post-inference-evaluation
```

결과:

```text
output-root/
  map_reset/
  beam2/
  comparison.json
  comparison.md
```

## 3. CPU smoke의 예상 결과

기본 설정에서 기대하는 deterministic 결과는 다음이다.

| Mode | Lifecycle events | CLOSE | REOPEN | Decision delay | Estimated start error |
|---|---|---:|---:|---:|---:|
| `map_reset` | `OPEN` | 0 | 0 | - | - |
| `beam2` | `OPEN -> CLOSE -> OPEN` | 1 | 1 | 각 1 frame | 각 0 frame |

구체적으로 beam-2는:

```text
CLOSE decision: t=41, estimated CP start=40
REOPEN decision: t=81, estimated CP start=80
```

이 결과는 frame당 10~20 count를 주는 기존 surprise test와 달리, production capped
mode와 같은 unit evidence에서 branch 보존 자체가 transition을 복원할 수 있는지
검사한다.

## 4. 304-frame real detector-only 실험의 예상 결과와 판정

### 결과 A: `map_reset CLOSE=0`, `beam2 CLOSE>0/REOPEN>0`

가장 기대하는 결과다.

해석:

```text
alpha-T evidence에는 상태 반전을 나타내는 정보가 있었다.
MAP-reset이 한 프레임 threshold 아래 branch를 폐기한 것이 직접 blocker였다.
```

다음 단계:

- beam-2를 production runner의 정식 `--bocd-mode`로 승격
- per-Gaussian transition proxy로 false CLOSE와 delay 측정
- hazard/threshold sweep
- 그 뒤 DC-only lifecycle과 geometry 조건 비교

### 결과 B: `map_reset CLOSE=0`, `beam2 CLOSE=0`

Branch 폐기만의 문제가 아니다.

해석 후보:

- reverted Gaussian에 충분한 negative alpha-T responsibility가 들어오지 않는다.
- capped mass가 너무 작거나 관측 간격이 부족하다.
- reference Gaussian footprint가 changed/reverted 영역을 올바르게 대표하지 않는다.

다음 단계:

- transition 근처 Gaussian subset의 `(delta_a, delta_b, total_mass)` 저장
- raw/capped count 비교
- min evidence mass와 observability audit
- exact BOCD subset replay

### 결과 C: beam-2가 transition 밖에서 많은 CLOSE/OPEN을 발생

Branch 보존은 성공했지만 detector가 과민하다.

다음 단계:

- expected run length 증가
- CP threshold 증가
- 최소 candidate visible count 및 evidence concentration 증가
- 3D neighborhood 또는 multi-view agreement gate 추가

### 2D metric 예상

Detector-only branch의 목표는 lifecycle recovery이지 geometry mIoU 향상이 아니다.
Beam-2가 stale active slot을 실제로 닫으면 DC-only current mask의 SC2/SC3 false-positive
잔상이 줄 가능성이 있다. 반대로 false CLOSE가 많으면 recall이 떨어질 수 있다.
따라서 첫 판정은 mIoU보다 다음 순서로 한다.

```text
1. CLOSE/REOPEN 존재 여부
2. transition 인접 decision delay
3. off-transition event 수
4. mean-frame IoU/F1
```

## 5. 테스트

추가 테스트:

```text
tests/temporal/test_beam2_bocd.py
tests/experiments/test_bocd_branch_preservation_smoke.py
tests/experiments/test_bocd_branch_ablation.py
```

검증 항목:

- unit capped evidence에서 MAP `CLOSE=0`, beam-2 transition 복원
- 두 branch posterior와 run-length posterior 정규화
- chunked update와 full-batch update 동일성
- unobserved row persistent state bitwise 보존
- persistent state가 `[N]` vector만 사용
- experimental runner hook이 종료 후 원래 함수로 복원
- CLI가 mode/output을 강제로 통제

로컬 CPU 검증:

```text
7 passed
```

전체 저장소 test와 CUDA 304-frame 실행은 PR branch를 checkout한 O-SCD 환경에서
추가로 수행해야 한다.
