# Oracle-Boundary Fixed-Capacity MCMC Adaptation for Lifespan `R_change`

## 1. 결론

Manual boundary가 주어진 `S0=[0,95)`, `S1=[95,199)`, `S2=[199,304)`에서
고정된 Gaussian 수를 유지한 A0--A4를 구현하고 full-capacity seed-0
`oracle_stream` pilot을 실행했다.

핵심 결과는 다음과 같다.

1. **고정 capacity와 immutable state archive는 정상 동작했다.** 모든 mode에서
   `N_t = 1,283,501`이었고, 완료된 archive의 parameter/render checksum drift는
   정확히 0이었다.
2. **geometry Adam은 DC-only보다 좋아지지 않았다.** A2는 A0보다 posthoc mIoU
   `-0.0435`, F1 `-0.0347` 낮았다.
3. **SGLD는 A2 대비 최종 posthoc 성능을 소폭 회복했지만 online adaptation은
   악화했다.** A3는 A2보다 posthoc mIoU `+0.0070`, F1 `+0.0041`이지만
   seen-prefix mIoU는 `-0.0156`이었다.
4. **official live-target relocation은 이 실험에서 실패했다.** relocation 한 번의
   전후 energy perturbation은 매우 작았지만, 반복 relocation 후 A4 posthoc mIoU는
   `0.2296`으로 A3보다 `-0.3911` 낮아졌다.
5. **Gaussian이 많이 움직였다는 사실은 새 geometry coverage를 의미하지 않았다.**
   A4의 xyz displacement p95는 state별 `6.14--8.83`이었지만 평균 cue-positive
   mass coverage는 A3 `0.5301`에서 A4 `0.1877`로 감소했다.

따라서 이번 질문에 대한 답은 **“fixed capacity adaptation 자체는 구현 가능하지만,
official live-target relocation만으로 disconnected new-object support를 발견하는 것은
불가능하며 이 설정에서는 오히려 누적 표현을 붕괴시켰다”**이다.

> 3DGS-MCMC relocation은 기존 live support 내 capacity redistribution에는
> 효과적이지만, disconnected residual geometry에 대한 proposal mechanism은
> 제공하지 않는다.

이 보고서에서 relocation은 **approximate rendering-preserving
3DGS-MCMC-style relocation**으로만 부른다. MH correction이 없으므로 exact MCMC
posterior sampling 또는 exact stationary-distribution preservation을 주장하지 않는다.

## 2. 구현 범위

### 2.1 구현한 구성

- `temporal/mcmc_state.py`
  - `FixedCapacityChangeState`
  - `OracleBoundaryStateManager`
  - immutable CPU/disk `StateArchive`
  - state별 `xyz`, DC, change opacity, scale, rotation
- `gaussian_renderer/__init__.py`
  - `override_xyz`, `override_dc`, `override_opacity`, `override_scaling`,
    `override_rotation`
  - override가 없을 때 legacy renderer와 동일한 경로
- `temporal/mcmc_energy.py`
  - 기존 `compute_ssf_loss` 재사용
  - explicit `sum|mean` opacity/scale regularizer
  - optional non-official geometry anchor
- `temporal/mcmc_dynamics.py`
  - covariance/opacity-gated xyz-only positional noise
  - change opacity 기준 dead/live 판정
  - opacity-weighted live target sampling
  - grouped Eq. 9 opacity/scale correction
  - live target Adam moment reset, source moment retention
- `experiments/train_oracle_boundary_mcmc_rchange.py`
  - A0 `dc_only`
  - A1 `dc_opacity`
  - A2 `geo_adam`
  - A3 `geo_sgld`
  - A4 `geo_mcmc`
  - optional A5 `geo_mcmc_anchor`
  - `matched_exact`와 causal `oracle_stream`
- 별도 evaluator와 result summarizer
  - `experiments/evaluate_oracle_boundary_mcmc_rchange.py`
  - `experiments/summarize_oracle_boundary_mcmc_results.py`

구체적인 tensor ownership과 call path는
[`manual-boundary-mcmc-code-map.md`](manual-boundary-mcmc-code-map.md)에 기록했다.

### 2.2 공식 구현과의 대응

구현은 [3D Gaussian Splatting as Markov Chain Monte Carlo 논문](https://arxiv.org/abs/2404.09591)과
[공식 구현](https://github.com/ubc-vision/3dgs-mcmc)의 commit
`7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13`을 대조했다. dead opacity threshold,
opacity-weighted `torch.multinomial`, grouped relocation equation, target moment reset,
opacity/covariance-dependent position noise를 port했다. 공식 구현의 `add_new_gs`는 fixed-N
계약과 충돌하므로 의도적으로 제외했다.

### 2.3 명시적으로 구현하지 않은 것

BOCD, automatic boundary/lifespan, training 중 GT, GT-guided relocation,
residual-guided proposal, MH/MALA, append/delete, densification/pruning, opacity reset,
camera optimization, full RGB update, object tracking은 추가하지 않았다.

## 3. 검증 protocol

### 3.1 A0 matched-exact reproduction gate

- 304 frames
- resolution 4
- frame당 정확히 120 updates, 총 36,480 updates
- full `N=1,283,501`
- authoritative legacy O-SCD evaluator 결과:
  - mIoU: **0.6244687266**
  - F1: **0.7459780084**

이는 기존 corrected DC-only 결과 `0.6245 / 0.7460`을 재현하므로 A1--A4 pilot을
해석할 baseline gate를 통과했다.

### 3.2 Engineering smoke

- 각 state의 첫 10 frames
- frame당 4 updates
- deterministic first-50k capacity
- seed 0, `state-init=previous`
- 요청 resolution은 8이었으나 cached pixel+SAM cue artifact가 resolution 4로
  고정되어 있었다. Cue를 묵시적으로 resample하지 않고 validator가 실행을
  중단했으므로, smoke는 동일 artifact의 **resolution 4**에서 수행했다.

Smoke 결과:

- A0--A4 모두 finite
- 모든 iteration에서 `N_t=50,000`
- densification/pruning/reset 0회
- training GT pixels loaded 0
- archive parameter/render drift 0
- 6번의 real relocation audit
- absolute relative energy jump p95/max: **0.0005810 = 0.0581%**
- pilot start gate `<=5%` 통과

최종 verification에서는 `tests/mcmc`와 기존 `tests/temporal`을 함께 실행해
**145 tests passed**를 확인했다. CUDA renderer equivalence/gradient와 dynamics
behavior tests가 이 집합에 포함된다. 기계 판독 가능한 최종 gate 결과는
`outputs/oracle_boundary_mcmc/final_validation.json`에 저장했다.

### 3.3 Full-capacity oracle-stream pilot

- 304 frames를 timestamp 순서로 처리
- future frame 미사용
- manual target switch만 제공
- frame당 16 updates, 총 4,864 updates
- full `N=1,283,501`
- `state-init=previous`
- seed 0
- 모든 mode에서 동일 input hash
- GT는 training 종료 후 별도 evaluator에서만 사용

`seen-prefix`는 frame 도착 직후 저장된 prediction의 online metric이고,
`posthoc`은 각 state 최종 immutable archive를 모든 해당 frame에 replay한 metric이다.

## 4. A0--A4 정량 결과

| Mode | Posthoc mIoU | Posthoc F1 | Seen-prefix mIoU | Seen-prefix F1 | Runtime / frame | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| A0 `dc_only` | **0.6571** | **0.7717** | **0.6333** | **0.7470** | 0.251 s | 3.48 GiB |
| A1 `dc_opacity` | 0.6257 | 0.7482 | 0.5712 | 0.6895 | 0.187 s | 3.36 GiB |
| A2 `geo_adam` | 0.6136 | 0.7370 | 0.5298 | 0.6477 | 0.241 s | 3.47 GiB |
| A3 `geo_sgld` | 0.6207 | 0.7411 | 0.5142 | 0.6353 | 0.523 s | 3.47 GiB |
| A4 `geo_mcmc` | **0.2296** | **0.3130** | 0.5017 | 0.6180 | 0.849 s | 3.51 GiB |

Posthoc delta:

- A2 - A0: mIoU `-0.04351`, F1 `-0.03471`
- A3 - A2: mIoU `+0.00703`, F1 `+0.00409`
- A4 - A3: mIoU `-0.39106`, F1 `-0.42816`

A4 state별 posthoc 결과는 S0 `0.0000/0.0000`, S1 `0.4599/0.5952`,
S2 `0.2092/0.3165`였다. A4의 seen-prefix와 posthoc 차이는 초반 적응 성능이
일부 존재했어도 segment 후반의 반복 relocation/optimization 뒤 final archive가
그 표현을 보존하지 못했음을 뜻한다.

## 5. Fixed-capacity와 archive audit

모든 full pilot mode에서 관측된 Gaussian count 집합은 정확히
`{1,283,501}`이었다. Tensor append/delete, Parameter replacement,
densification, clone, split, prune, opacity reset은 호출되지 않았다.

A1--A4의 S0/S1/S2 archive에 대해 생성 직후와 전체 training 종료 후를 비교했다.

- parameter checksum drift: **0**
- deterministic audit-view render checksum drift: **0**
- optimizer는 target boundary에서만 재생성
- current state의 update/noise/relocation이 completed archive에 적용된 사례: **0**

따라서 성능 저하는 archive corruption 때문이 아니다.

## 6. SGLD와 geometry coverage

### 6.1 이동량

state별 xyz displacement-vs-base p95:

| Mode | S0 | S1 | S2 |
|---|---:|---:|---:|
| A2 Adam | 0.017 | 0.025 | 0.032 |
| A3 SGLD | 2.259 | 2.316 | 2.324 |
| A4 MCMC | 6.138 | 7.943 | 8.830 |

A3/A4에는 최대 약 `5.0e4`의 extreme displacement outlier도 있었다. 모든 값은
finite였지만, 이는 official RGB-3DGS용 noise scale을 sparse change-opacity pool과
under-constrained SSF target에 그대로 적용했을 때 dead Gaussian이 장면 bounds 밖으로
크게 이동할 수 있음을 보여준다.

### 6.2 coverage

전체 frame 평균 cue-positive mass coverage:

- A0: `0.5099`
- A1: `0.4989`
- A2: `0.5162`
- A3: `0.5301`
- A4: `0.1877`

A3는 A2보다 coverage를 `+0.0139` 늘렸지만 posthoc mIoU 개선은 `+0.0070`에
그쳤고 seen-prefix는 악화했다. A4는 displacement가 가장 큰데도 coverage가 크게
감소했다. 따라서 **slot movement 자체는 useful new-object geometry 발견의 증거가
아니다.**

## 7. Relocation audit

### 7.1 Synthetic Eq. 9 audit

opacity `{0.1,0.5,0.95}`, group size `{2,4,8}`, isotropic/anisotropic scale을
검사했다.

- NaN/Inf: 0
- 모든 case에서 MCMC relocation이 naive clone보다 작은 mean render error
- 최소 improvement ratio: `4.03x`
- opacity 0.1 case는 모두 mean error `<=1e-4`
- 전체 최대 mean error: `0.0250` (opacity 0.95, anisotropic, group 8)

즉 user-specified `1e-4` absolute target은 high-opacity case에서 충족되지 않았다.
원인은 Eq. 9가 중앙 alpha composition과 footprint를 근사 보존하는 식이지 모든
sample pixel의 response를 exact하게 보존하는 변환이 아니기 때문이다. 이 결과는
relocation을 exact-invariant라고 부르지 않아야 한다는 실험적 근거다.

### 7.2 Real pilot relocation

- aggregate relocation audits: 32
- source relocation events: 320,000
  - S0 100,000 / S1 110,000 / S2 110,000
- source-target distance: mean `7.984`, p50 `4.972`, p95 `13.049`, max `22,760.4`
- clone group size: mean `3.747`, p50 `3`, p95 `9`, max `38`
- target moment reset flag: 320,000/320,000 true
- source moment retained flag: 320,000/320,000 true
- absolute relative total-energy jump: p95 `0.0003181`, max `0.0004041`
  - 각각 약 `0.0318%`, `0.0404%`

따라서 A4 실패를 “한 번의 relocation이 render/energy를 크게 깨뜨렸기 때문”으로
설명할 수 없다. 즉시 perturbation은 작았지만, relocation destination은 이미 live인
Gaussian 위치로 제한되고, 이후 SGLD/Adam/opacity regularization이 동일 support에
집중된 clone 집합을 다시 변화시킨다. 이 누적 과정에서 A4 평균 rendered-alpha
coverage는 A3 `0.0711`에서 `0.0237`로 감소했다.

## 8. 질문별 답변

### 1. Geometry optimization만으로 DC-only보다 좋아졌는가?

아니다. A2는 A0보다 posthoc mIoU `0.0435`, F1 `0.0347` 낮았다. SSF mask만으로
xyz/depth/scale/rotation을 동시에 최적화하는 문제는 under-constrained였다.

### 2. SGLD positional exploration이 geometry Adam보다 좋아졌는가?

부분적으로만 그렇다. 최종 archive 기준 A3는 A2보다 mIoU `+0.0070`, F1
`+0.0041`이지만, seen-prefix mIoU/F1은 각각 `-0.0156/-0.0124`였다. 큰 outlier
displacement까지 고려하면 안정적인 우위라고 결론 내릴 수 없다.

### 3. Relocation이 SGLD-only보다 좋아졌는가?

아니다. A4는 A3보다 posthoc mIoU `-0.3911`, F1 `-0.4282` 낮았다.

### 4. Gaussian count는 정확히 유지됐는가?

그렇다. 모든 full pilot iteration에서 `N_t=N_0=1,283,501`이었다.

### 5. 이전 state archive는 완전히 보존됐는가?

그렇다. parameter checksum과 deterministic replay render checksum drift가 모두
정확히 0이었다.

### 6. 새 state에 없던 geometry coverage가 증가했는가?

A3에서 cue-positive mass coverage가 A2보다 소폭 증가했지만, A4에서는 크게
감소했다. 움직임 크기는 증가했지만 useful disconnected geometry coverage는
증가하지 않았다.

### 7. Relocation 전후 energy는 얼마나 변했는가?

32회 audit의 absolute relative total-energy jump는 p95 `0.0318%`, 최대
`0.0404%`였다. 순수 relocation은 local/instantaneous sense에서 충분히 작았다.

### 8. Official live-target relocation이 disconnected new object에 충분했는가?

아니다. Proposal destination이 기존 live position뿐이므로 그 위치 밖의 residual
surface를 직접 seed할 수 없다.

### 9. 실패 원인은 무엇인가?

근거의 우선순위는 다음과 같다.

1. **Relocation destination limitation:** 새로운 residual 위치가 proposal support에
   포함되지 않는다.
2. **Under-constrained SSF geometry:** A2조차 A0보다 낮고, 유사한 final SSF energy가
   매우 다른 GT generalization을 만든다.
3. **Opacity/capacity concentration과 누적 drift:** A4는 작은 instantaneous energy
   jump에도 반복 relocation 후 alpha/cue coverage가 붕괴했다.
4. **Official SGLD scale의 domain mismatch:** dead change Gaussian에서 매우 큰 finite
   displacement outlier가 생겼다.

Initialization 또는 archive corruption이 주원인이라는 증거는 없다. A2/A3가 같은
`previous` initialization에서 정상적인 범위의 prediction을 만들었고 archive drift도
0이기 때문이다.

### 10. 다음 단계에서 residual-guided proposal이 필요한가?

그렇다. 이번 negative result는 proposal support를 observed multi-view residual geometry로
확장할 필요성을 제공한다. 다만 이는 이번 범위에 포함하지 않았으며, 다음 단계에서는
depth/ray consistency, proposal ratio, 필요 시 MH correction을 별도 설계해야 한다.

## 9. 생성된 산출물

핵심 결과:

- `outputs/oracle_boundary_mcmc/comparison.json`
- `outputs/oracle_boundary_mcmc/smoke/validation.json`
- `outputs/oracle_boundary_mcmc/diagnostics/synthetic_relocation_audit.json`
- `outputs/oracle_boundary_mcmc/oracle_stream/{dc_only,dc_opacity,geo_adam,geo_sgld,geo_mcmc}`
- `outputs/oracle_boundary_mcmc/matched_exact/dc_only_seed0_reproduction`

각 run은 config/input hashes, energy, Gaussian counts, archive, online prediction,
relocation event/audit를 보존한다. A4 `relocation_events.jsonl`의 source event는 Gaussian
slot의 **representation-capacity lineage**이며 object motion trajectory가 아니다.

요청된 plot 10종은 `outputs/oracle_boundary_mcmc/diagnostics/oracle_stream/`에 있다.

1. `adaptation_curve.png`
2. `energy_adaptation_curve.png`
3. `cue_alpha_coverage_curve.png`
4. `alive_dead_gaussians.png`
5. `relocation_count_distance.png`
6. `relocation_delta_energy_histogram.png`
7. `xyz_displacement_histogram.png`
8. `state_confusion_contact_sheet.png`
9. `ablation_final_comparison.png`
10. `runtime_memory_comparison.png`

## 10. Stop decision

A4 pilot이 명확히 실패했으므로 요청한 gate에 따라 matched-exact A1--A4 및
3-seed full sweep은 시작하지 않았다. 이를 실행해도 현재 official live-target proposal의
구조적 destination 제약을 해결하지 못하고 큰 계산량만 소비한다.

이번 단계에서 주장하지 않는 항목:

- exact MCMC posterior sampling
- exact stationary-distribution preservation
- BOCD 성능
- automatic evolving-scene SCD
- object trajectory tracking
- new object의 complete RGB reconstruction
