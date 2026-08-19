# Bayesian lifespan + active geometry

## 범위와 이진 상태 계약

각 reference Gaussian `i`의 온라인 change 상태는 다음 이진 변수로만 해석한다.

- `z_i,t = 1`: 현재 `R_ref`와 다른 영역을 나타낸다.
- `z_i,t = 0`: 현재 `R_ref`와 다른 change가 없다.

따라서 `change A -> change B`에서 A와 B가 모두 reference와 다르면 representation lifespan 전이가 아니다. BOCD가 evidence 분포의 reset을 검출하더라도 이진 label이 `active -> active`이면 action은 반드시 `KEEP`이다. 기존 slot을 닫거나 새 slot을 만들지 않고 같은 interval에서 DC와 geometry를 계속 학습한다.

Representation lifecycle은 아래 표만 따른다.

| 이전 label | 새 label | action | 표현 변화 |
| --- | --- | --- | --- |
| inactive | inactive | `NONE` | closed 유지 |
| inactive | active | `OPEN` | 다음 unused slot 할당 |
| active | active | `KEEP` | 같은 slot과 interval 유지 |
| active | inactive | `CLOSE` | 현재 interval 종료 |

나중에 다시 `inactive -> active`가 되면 이전 closed slot을 덮어쓰지 않고 새 slot을 연다. Slot capacity가 부족하면 affected Gaussian 수를 포함한 error를 발생시킨다.

## Detector, renderer, plasticity 분리

### 1. Lifespan-agnostic Bayesian observability

`temporal/change_evidence.py`는 immutable reference Gaussian만 probe한다. Evidence render에는 다음만 사용한다.

- detached base xyz;
- detached base opacity;
- detached base scaling/rotation;
- differentiable zero probe color.

Temporal `state_valid`, active opacity mask, state-local geometry delta는 사용하지 않는다. 그러므로 never-opened 또는 closed Gaussian도 alpha-transmittance mass가 있으면 evidence를 받고 `OPEN`/`REOPEN`할 수 있다.

### 2. Lifespan-gated current rendering

기존 temporal renderer의 half-open interval semantics를 유지한다. 현재 timestamp에서 유효한 slot만 `R_change`에 기여하고 inactive/closed/historical slot은 current output에서 숨긴다.

### 3. Active-only state-local plasticity

제안 경로는 `TemporalGeometryChangeModel`만 사용한다. Shared mutable geometry는 사용하지 않는다. Base의 `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation`은 frozen이며 run 전후 bitwise checksum으로 검증한다.

`MaskedRowSlotAdam`은 현재 OPEN인 `(Gaussian row, state slot)`만 update한다. Inactive pair의 parameter, first/second moment, AMSGrad max moment, pair-local step counter는 다른 row의 update 동안 정확히 보존된다. 새 slot을 OPEN할 때는 그 pair의 optimizer state만 reset한다.

## Alpha-T evidence

`render_change(..., override_color=probe_color, clamp_output=False)`의 한 번의 VJP를 사용한다.

\[
\rho_{i,t}(p)=\alpha_{i,t}(p)T_{i,t}(p)
\]

\[
e^+_{i,t}=\sum_p \rho_{i,t}(p)C_t(p),\qquad
e^-_{i,t}=\sum_p \rho_{i,t}(p)(1-C_t(p))
\]

VJP output weight는 channel 0에 `C_t`, channel 1에 `1-C_t`, channel 2에 0을 둔다. `probe_color.grad[:,0]`과 `[:,1]`이 각각 positive/negative evidence가 된다. CUDA test는 두 채널을 finite difference와 비교한다. FastGS gradient가 맞지 않으면 test가 실패하며 integer hit count fallback은 없다.

### Cue mode

- `binary`: `candidate_map > bayes_cue_threshold`. Source-faithful Beta-Bernoulli/B3-Seg observation mode이다.
- `soft`: `clamp(candidate_map / bayes_cue_scale, 0, 1)`. Fractional/power-likelihood extension이며 원래 Bernoulli observation model과 동일하다고 해석하지 않는다.

### Evidence count mode

`raw`는 `delta_a=e+`, `delta_b=e-`를 사용한다. `capped`는 다음을 사용한다.

\[
q_i=\frac{e_i^+}{e_i^+ + e_i^- + \epsilon},\qquad
w_i=\operatorname{clamp}\left(\frac{e_i^+ + e_i^-}{m_{sat}},0,1\right)
\]

\[
\Delta a_i=w_iq_i,\qquad \Delta b_i=w_i(1-q_i)
\]

`total_mass < min_evidence_mass`이면 unobserved이다. 이 경우 Beta tensor, BOCD run posterior/start/length, visible count, controller state와 lifecycle을 advance하지 않는다.

## Beta-Bernoulli BOCD

Fractional pseudo-count `(s,f)`의 integrated predictive score는 다음과 같다.

\[
\log p(s,f\mid a,b)=\log B(a+s,b+f)-\log B(a,b)
\]

`torch.lgamma`와 log-space normalization을 사용한다.

- `exact`: bounded/truncated `[N, max_run_length+1]` run-length posterior와 per-run Beta/state start/visible count를 유지한다.
- `map_reset`: 현재 MAP run 하나의 Beta sufficient statistics만 유지하는 명시적인 `O(N)` 근사 대안이다.

Runner는 exact persistent-state 예상량이 `--exact-bocd-memory-limit-gb`를 넘으면 error를 내고 자동으로 다른 알고리즘으로 바꾸지 않는다. `summary.json`과 frame CSV에는 실제 사용한 algorithm을 기록한다.

Controller는 changepoint가 검출된 새 run의 binary posterior가 충분히 확실해질 때까지 flip하지 않는다. Pending reset의 estimated start와 changepoint probability는 latch하지만 representation interval은 backdate하지 않는다. `state_start/state_end`에는 실제 decision timestamp만 기록한다.

## Lifecycle state와 checkpoint migration

`TemporalChangeModel`은 다음 lifecycle buffer를 유지한다.

- `state_status [N,S]`: `EMPTY=0`, `OPEN=1`, `CLOSED=2`;
- `num_states [N]`;
- `current_state_index [N]`.

Rendering의 authoritative source는 기존 `state_start`, `state_end`, `state_valid`이다. 기본 생성은 기존 manual-boundary 동작처럼 slot 0을 open 상태로 보존한다. 자동 runner만 `reset_all_lifespans_closed()`를 명시적으로 호출한다.

구 checkpoint에 lifecycle buffer가 없으면 interval에서 deterministic하게 추론한다.

- invalid -> `EMPTY`;
- valid + finite end -> `CLOSED`;
- valid + infinite end -> `OPEN`.

DC-only, state-specific geometry, shared-geometry legacy migration과 새 lifecycle round-trip을 test한다. 기존 manual-boundary runner는 변경하지 않았다.

## Causal runner

`experiments/run_online_bayesian_lifespan_thaw.py`는 global timestamp의 frame-major 순서로 다음을 수행한다.

1. fixed-pose view와 cached O-SCD pixel+SAM cue를 읽는다.
2. immutable reference의 alpha-T evidence를 계산한다.
3. observed Gaussian만 chunk 단위로 Beta/BOCD update한다.
4. binary `OPEN/KEEP/CLOSE/NONE/UNCERTAIN`을 적용한다.
5. current lifespan-gated representation을 render한다.
6. detector-only가 아니면 current OPEN pair만 update한다.
7. causal diagnostics와 prediction을 저장한다.

GT mask와 `[95,199]` diagnostic boundary는 online loop가 끝난 뒤에만 평가에 사용한다. Evidence, BOCD, lifecycle, DC/geometry update에는 사용하지 않는다. Detector decision은 immutable base evidence만 사용하므로 동일 seed/cue/pose/config를 사용한 DC-only와 geometry run은 같은 causal lifecycle decision을 공유한다.

### 주요 실행 예

합성 detector/controller smoke:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_bayesian_lifespan_thaw \
  --detector-only-smoke --output-dir /tmp/oscd-bayesian-synthetic-smoke
```

실제 데이터 detector-only 제한 smoke:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_bayesian_lifespan_thaw \
  --detector-only --max-frames 1 --max-states 2 \
  --skip-post-inference-evaluation \
  --output-dir /tmp/oscd-bayesian-detector-smoke
```

세 가지 비교 설정:

```bash
# A. detector only
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_bayesian_lifespan_thaw \
  --detector-only --output-dir outputs/bayesian-detector-only

# B. auto-lifespan DC only
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_bayesian_lifespan_thaw \
  --thaw-parameters dc --output-dir outputs/bayesian-dc

# C1. auto-lifespan DC + xyz
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_bayesian_lifespan_thaw \
  --thaw-parameters dc,xyz --output-dir outputs/bayesian-dc-xyz

# C2. auto-lifespan DC + all state-local geometry
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_bayesian_lifespan_thaw \
  --thaw-parameters dc,xyz,opacity,scaling,rotation \
  --output-dir outputs/bayesian-dc-all-geometry
```

각 learning rate는 `--dc-lr`, `--xyz-lr`, `--opacity-lr`, `--scaling-lr`, `--rotation-lr`로 노출한다.

## Output

Generated artifact는 commit하지 않는다. Runner는 지정한 output directory에 다음 machine-readable file만 쓴다.

- `summary.json`;
- `frame_metrics.csv`;
- `lifecycle_events.jsonl`;
- `per_frame_bayesian_stats.npz`;
- `checkpoint.pt`.

Frame diagnostics에는 observed count, positive/negative pseudo-count mass, change/CP probability quantile, action count, active lifespan, geometry-thawed row, base checksum, inactive drift audit, runtime과 CUDA peak memory가 포함된다. Event에는 Gaussian index, decision/estimated CP timestamp, old/new label, action, slot, binary posterior, CP posterior, concentration과 run-visible count가 포함된다.

## 검증 명령

```bash
PYTHONPATH=. conda run -n oscd python -m compileall -q \
  experiments poses temporal tests

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  conda run -n oscd pytest -q tests

git diff --check
```

## 실험 결과

독립 `ref -> SC1/SC2/SC3`와 연속 `ref -> SC1 -> SC2 -> SC3` 304-frame 결과, lifecycle failure 분석, raw-render GIF 경로는 [`escd-bayesian-lifespan-experiment-results-ko.md`](escd-bayesian-lifespan-experiment-results-ko.md)에 기록한다. 연속 MAP-reset run에서는 OPEN `81,268`, CLOSE/REOPEN `0`이었으며, all-geometry가 같은 OPEN slot을 morphing해 lifecycle 실패를 가리는 현상을 확인했다.

## 제한사항

- 연속 full run의 `MAPResetBernoulliFilter`는 changepoint branch를 threshold 아래에서 폐기하므로 exact BOCD와 동등하지 않다.
- Exact BOCD는 bounded이어도 1.28M Gaussian에서 수 GB의 persistent state가 필요하다. `map_reset`은 이름과 output에서 근사 알고리즘임을 명시한다.
- Inactive drift runtime audit는 `--inactive-audit-max-pairs`까지 closed pair를 exact 비교하며, 전체를 덮지 못하면 `exhaustive=false`를 기록한다. Unit test는 모든 inactive pair의 parameter와 optimizer state를 exact 검증한다.
- Global manual boundary에는 per-Gaussian binary lifecycle GT가 없으므로 false OPEN/CLOSE rate는 정확히 정의할 수 없다. Runner는 이를 `null`과 사유로 기록하며 geometry improvement를 주장하지 않는다.
- 새 위치에 reference support가 없는 경우는 이 경로의 범위 밖이다. 기존 XFeat NEW-seed module은 수정하지 않았다.
- MCMC, SGLD, relocation, densification/pruning, unrestricted birth, signed PCA, change identity classification, SSF formulation과 pose estimation은 변경하지 않았다.
