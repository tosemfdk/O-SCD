# 보수적 multi-view residual relocation

## 목적

`geo_conservative_reloc`은 기존 A4 `geo_mcmc`를 변경하지 않는 별도 ablation이다.
목표는 **현재 state에서 충분히 관측됐지만 change cue를 한 번도 지지하지 못한
Gaussian capacity를, 아직 설명되지 않은 multi-view change cue의 3D 위치로
재배치하는 것**이다.

## 기존 A4와의 차이

| 항목 | `geo_mcmc` | `geo_conservative_reloc` |
|---|---|---|
| source | change opacity `< 0.005` | 충분히 관측 + cue support 0 |
| 과거 정보 | 별도 보호 없음 | 이전 state support slot 영구 보호 |
| destination | active/live Gaussian 위치 | 여러 residual ray가 합의한 3D 위치 |
| 이동 후 attribute | live target의 DC/opacity/scale 복제 | DC=0, 작은 tentative opacity, source scale/rotation 유지 |
| 검증 | Eq. 9 전후 audit | causal replay energy + cue coverage gate |
| 실패 처리 | 적용 후 유지 | touched row 전체 rollback |
| SGLD | 사용 | 기본적으로 사용하지 않음 |
| opacity sparsity | 기본 사용 | 기본 0; source 선택에도 opacity 미사용 |

## 동작 순서

1. 도착한 view마다 각 slot의 visibility와 positive-cue overlap을 누적한다.
2. 다음 조건을 모두 만족하는 slot만 source 후보가 된다.
   - `observation_count >= conservative_min_observations`
   - `cue_support_count == 0`
   - 이전 state에서 support되지 않음
   - tentative relocation 상태가 아님
3. causal replay buffer의 각 view에서
   `max(cue - clamp(render, 0, 1), 0)` residual을 계산한다.
4. residual 상위 pixel ray를 view pair 사이에서 triangulation한다.
5. 다른 causal view로 다시 projection하여 최소 view 수 이상에서 residual support를
   받는 3D 후보만 남긴다.
6. source를 거의 0인 transport opacity로 만든 뒤 위치를 옮긴다.
7. 작은 tentative opacity로 활성화한다.
8. replay SSF energy와 cue-positive coverage가 허용 범위 안이면 유지하고,
   아니면 parameter와 evidence metadata를 즉시 정확히 rollback한다.

완료 state는 기존 `StateArchive`에 저장되므로 이후 state의 relocation이 과거
state parameter를 수정하지 않는다. Oracle-stream에서는 새 mode에 한해 같은
state의 causal replay buffer를 사용해 current-frame-only forgetting도 완화한다.

## 중요한 해석 제한

이 mode는 공식 3DGS-MCMC Eq. 9 relocation이 아니다. Residual-guided proposal에
MH correction을 적용하지 않으므로 exact MCMC 또는 stationary-distribution
preservation을 주장하지 않는다. 정확한 표현은 다음과 같다.

> Causal multi-view residual-guided conservative fixed-capacity relocation

Gaussian slot 이동 로그는 물체 trajectory가 아니라 representation-capacity
lineage이다.

## 실행 예시

```bash
conda run -n oscd python experiments/train_oracle_boundary_mcmc_rchange.py \
  --source-path data/Instance_1/scene_change1_2_3 \
  --output-dir outputs/oracle_boundary_mcmc/oracle_stream/geo_conservative_reloc \
  --resolution 4 \
  --protocol oracle_stream \
  --mode geo_conservative_reloc \
  --allow-full-stream \
  --updates-per-frame 16 \
  --state-init previous \
  --seed 0
```

기본값은 source 관측 3회, destination support 2 views, replay buffer 8 views,
relocation당 최대 256 slots이다. Acceptance는 **energy 비증가**와 최소
`1e-6`의 cue coverage 증가를 동시에 요구한다. 논문 비교 전에는 이 값들을
manifest와 함께 고정해야 한다.

실패한 proposal은 즉시 rollback한다. 일반 `post_*` 필드는 rollback 후 실제
state를 기록하고, 거절된 tentative 상태의 변화량은 `attempted_*` 필드에만
별도로 남긴다.

## Engineering smoke

검증 artifact:

- `outputs/oracle_boundary_mcmc/smoke/geo_conservative_reloc_strict`
- `outputs/oracle_boundary_mcmc/smoke/geo_conservative_reloc_acceptance_path`
- `outputs/oracle_boundary_mcmc/smoke/geo_conservative_reloc_rollback_strict`

50k fixed-capacity smoke에서 확인된 항목:

- 모든 step에서 `N_t = 50,000`
- densification/pruning 없음
- training GT pixel load 0
- causal replay 사용
- strict 기본값: 7 audits 중 3 proposal을 모두 rollback
- acceptance-path 수치 오차 허용(`energy <= 1e-6`, `coverage gain >= 1e-7`):
  1 transaction, 4 sources 적용
- pure transport render L1 최대 0
- proposal의 최대 absolute relative energy change 약 `6.48e-7`
- archive parameter/render replay drift 0
- strict/강제 coverage-failure smoke의 모든 source rollback이 exact row restoration
- 거절 event의 `post_*`는 실제 원복 상태, `attempted_*`는 tentative 상태로 분리됨

이 smoke는 engineering 검증이며 성능 결론으로 사용하지 않는다.

## Full oracle-stream pilot 결과

동일한 full-N(`1,283,501`), 304 frames, 16 updates/frame, seed 0 조건에서
실행했다. Relocation의 순수 효과를 분리하기 위해 같은 causal replay schedule과
optimizer를 사용하고 `conservative_max_relocations=0`만 적용한 control도 함께
평가했다.

| 설정 | posthoc mIoU | posthoc mF1 | seen-prefix mIoU |
|---|---:|---:|---:|
| DC-only A0 | 0.6571 | 0.7717 | 0.6333 |
| 기존 official-style A4 | 0.2296 | 0.3130 | 0.5017 |
| conservative replay, relocation 없음 | **0.6286** | **0.7479** | **0.5070** |
| conservative residual relocation | 0.6250 | 0.7450 | 0.5068 |

State별 relocation/no-relocation mIoU는 각각 다음과 같다.

| State | relocation | no relocation | 차이 |
|---|---:|---:|---:|
| S0 | 0.5833 | 0.5909 | -0.0076 |
| S1 | 0.6577 | 0.6589 | -0.0012 |
| S2 | 0.6304 | 0.6327 | -0.0023 |

32회 proposal 중 18회가 채택되어 총 4,608 slots가 이동했다. Gaussian 수는
항상 고정됐고 archive drift와 training GT 접근은 0이었다. 그러나 같은 replay
control보다 posthoc mIoU가 `0.0036` 낮았다. 따라서 A4의 심각한 collapse는
해결했지만, 현재 residual relocation 자체의 정확도 이득은 입증되지 않았다.

가능성이 높은 이유는 instantaneous SSF energy/coverage gate가 이후 Adam
trajectory의 GT generalization을 보장하지 않으며, cue ray pairing에 depth 또는
appearance correspondence가 없어 잘못된 3D destination도 통과할 수 있기 때문이다.
다음 refinement에는 accepted slot의 delayed multi-view 재검증과 timeout rollback이
우선 필요하다.

## Occlusion-safe row-local burn-in ablation

DC=0 Gaussian도 뒤쪽 change Gaussian을 alpha로 가리는 유효한 occluder일 수
있다는 가설을 반영해 다음을 추가했다.

1. source를 이동하기 전에 원위치에서 opacity를 낮춘다.
2. SSF energy 또는 cue-negative 영역의 rendered change mass가 증가하면 source를
   `removal_protected`로 표시하고 이동하지 않는다.
3. 안전한 source만 residual 위치로 이동한다.
4. 이동된 row만 별도 Adam으로 학습하며, 앞 절반은 DC/opacity만, 뒤 절반은
   xyz/scale/rotation까지 연다.
5. proposal/train view와 validation view를 causal replay buffer에서 분리하고,
   validation energy, positive coverage, negative mass가 모두 gate를 통과해야 commit한다.

Full-N, 304 frames, 16 base updates/frame, seed 0 결과:

| 설정 | posthoc mIoU | posthoc mF1 | seen-prefix mIoU |
|---|---:|---:|---:|
| matched replay, relocation 없음 | **0.627839** | **0.747288** | 0.506543 |
| relocation, burn-in 0 | 0.626926 | 0.746521 | 0.506093 |
| relocation, burn-in 32 | **0.627578** | **0.747081** | 0.506654 |
| relocation, burn-in 64 | 0.626871 | 0.746463 | **0.506841** |
| relocation, burn-in 120 | 0.627115 | 0.746787 | 0.506281 |

32-step burn-in은 burn-in 0보다 mIoU `+0.000652`(`+0.065pp`) 높았지만,
matched no-relocation control보다 `-0.000261`(`-0.026pp`) 낮았다. 64/120으로
학습량을 늘려도 단조롭게 좋아지지 않았다. 따라서 **relocation 후 집중 학습은
약간의 회복 효과는 있지만 relocation 자체를 이득으로 바꾸지는 못했다.**

이번 seed에서 source deactivation 단계의 occlusion/energy protection event는 0회였다.
즉 선택된 source group은 실제로 제거 가능한 저책임 capacity였고, 성능 한계는
source occluder 손상보다는 destination 정확도와 이후 generalization에 더 가깝다.
반면 final negative-mass gate는 설정별 8--11 proposal을 거절하여 relocation이
cue-negative 영역을 오염시키는 경우는 실제로 존재함을 확인했다.

비교 artifact:
`outputs/oracle_boundary_mcmc/oracle_stream/conservative_burnin_comparison.json`

## 현재 한계

- cue만으로 triangulation하므로 texture correspondence나 depth prior가 없다.
- 두 view만으로도 후보가 될 수 있어 잘못된 ray pairing 가능성이 있다.
- tentative slot은 두 view에서 cue support를 받으면 확정되지만, 장기간 support가
  없을 때 자동으로 이전 위치로 되돌리는 delayed timeout rollback은 아직 없다.
- full-capacity oracle-stream 성능 평가는 별도 pilot이 필요하다.
