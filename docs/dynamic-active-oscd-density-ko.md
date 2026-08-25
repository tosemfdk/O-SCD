# Equal-status dynamic ACTIVE O-SCD density ablation

## 1. 실험 질문

이 ablation은 `reference/root Gaussian`과 `densified child Gaussian`을 서로
다른 종류로 취급하지 않는다. 현재 mutable `R_change` bank의 모든 row는
동등하며, 유일한 표현 상태 차이는 다음뿐이다.

```text
ACTIVE   : 현재 change를 표현하고 학습/density-control 가능
INACTIVE : 현재 render에서 숨기고 parameter/Adam state를 보존
```

Detector는 row를 ACTIVE/INACTIVE로 분류한다. Representation은 ACTIVE row만
사용해 candidate cue를 표현한다.

## 2. 매 frame causal 순서

```text
cached current candidate cue
  -> current mutable bank의 raw xyz/opacity/scale/rotation alpha-T evidence
  -> direct binary Bayesian filter + K3/BF3 lifecycle controller
  -> OPEN/CLOSE/KEEP/NONE
  -> already-processed view만 사용하는 16회 O-SCD loss update
  -> ACTIVE AND selected-view visible row만 Adam update
  -> local update 4에서 ACTIVE-only density event
```

Future view, GT mask, manual boundary는 inference/training/density decision에
사용하지 않았다. GT는 전체 frame 처리가 끝난 뒤 평가에만 읽었다.

## 3. 이전 immutable-detector 실험과 달라진 점

이번 구조에서는 초기 Gaussian과 densified Gaussian 모두 같은 detector row다.
따라서 detector evidence도 현재 mutable bank의 raw geometry/opacity footprint에서
계산한다. Lifespan opacity gate는 무시하므로 INACTIVE row도 계속 관측될 수 있다.

이 선택은 명시적인 trade-off다.

```text
장점: 새 GS도 다음 frame부터 독립적인 Bayesian evidence를 직접 받음
단점: geometry/opacity 학습과 topology 변화가 detector observability에 feedback됨
```

따라서 이 ablation은 기존의 엄격한 `immutable detector / mutable
representation` 분리와 동일한 확률 실험이 아니다.

## 4. Densification과 pruning

### 4.1 Densification

`oscd.py`의 online loop와 같은 local schedule을 사용했다.

```text
updates/frame = 16
density local update = 4
gradient threshold = 0.0002 * 5 = 0.001
percent_dense = 0.01
```

후보는 반드시 ACTIVE이며 첫 다섯 update 중 실제 visible gradient를 받은 row다.

```text
clone = ACTIVE AND observed-gradient AND grad >= 0.001 AND small
split = ACTIVE AND observed-gradient AND grad >= 0.001 AND large
```

Clone은 raw DC/xyz/SH-rest/opacity/scaling/rotation을 복사한다. Split은 기존
O-SCD/FastGS port의 위치 perturbation과 scale 축소를 재사용하고 나머지 raw
attribute를 복사한다.

### 4.2 새 row의 Bayesian/lifecycle 상태

생성 순간 다음 상태를 source에서 한 번 복사한다.

```text
p_active
visible_observations
last_timestamp
open/close confirmation counter
lifespan interval/status/current slot
```

Optimizer moment는 0에서 시작한다. 생성 이후 parameter, Bayesian posterior,
lifespan은 source와 연결되지 않고 독립적으로 갱신된다. `parent_stable_id`는
분석용 로그일 뿐 broadcast나 root protection에 사용하지 않는다.

### 4.3 Pruning

Pruning은 ACTIVE row에만 허용한다. 초기 bank row도 ACTIVE라면 제거될 수 있고,
densified row라는 이유로 별도 보호하거나 차별하지 않는다. 제거할 때는 raw
parameter, Adam state, Bayesian state, controller state, lifespan buffer를 같은
row mask로 함께 제거한다.

주의할 점은 원본 `oscd.py`의 16-step online loop는 clone/split만 하고 prune은
하지 않는다는 것이다. 이번 `opacity < 0.4`/size pruning은 사용자가 요청한
명시적 확장이고, O-SCD online 원형 그대로인 조건은 `min_opacity=0`인
densify-only 조건이다.

FastGS cue VCD/VCP와 K=10 importance window는 사용하지 않았다.

## 5. 비교 조건

모든 조건은 Instance 1의 독립 `ref -> SC1`, `ref -> SC2`, `ref -> SC3`, fixed
pose, 동일 cached O-SCD pixel+SAM binary cue, seed 0, frame당 16 update다.

1. `none`: topology 고정
2. `active_oscd_densify_only`: ACTIVE-only O-SCD clone/split, 추가 opacity prune 없음
3. `active_oscd`: 위 densification + ACTIVE `opacity < 0.4` prune

## 6. 결과

| 조건 | scope | mIoU | F1 | Precision | Recall | final GS | clone | split source | removed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | SC1 | 0.533042 | 0.636652 | 0.777252 | 0.865930 | 1,283,501 | 0 | 0 | 0 |
| densify-only | SC1 | 0.531757 | 0.635114 | 0.773978 | 0.867226 | 1,283,790 | 10 | 279 | 279 |
| densify+prune | SC1 | 0.526396 | 0.630783 | 0.776823 | 0.856455 | 1,265,675 | 10 | 296 | 18,428 |
| none | SC2 | 0.617550 | 0.751805 | 0.687505 | 0.893823 | 1,283,501 | 0 | 0 | 0 |
| densify-only | SC2 | 0.616295 | 0.750791 | 0.684432 | 0.895525 | 1,283,965 | 23 | 441 | 441 |
| densify+prune | SC2 | 0.616925 | 0.751458 | 0.688631 | 0.887540 | 1,243,285 | 29 | 474 | 41,193 |
| none | SC3 | 0.579160 | 0.718612 | 0.700946 | 0.826539 | 1,283,501 | 0 | 0 | 0 |
| densify-only | SC3 | 0.578554 | 0.718028 | 0.699880 | 0.827249 | 1,283,764 | 5 | 258 | 258 |
| densify+prune | SC3 | 0.567185 | 0.709300 | 0.692611 | 0.815146 | 1,246,427 | 10 | 319 | 37,722 |

304개 독립 frame의 frame-weighted 결과:

| 조건 | mIoU | F1 | 총 runtime | max peak CUDA |
|---|---:|---:|---:|---:|
| none | 0.577881 | 0.704355 | 237.7 s | 4.15 GiB |
| densify-only | 0.576841 | 0.703326 | 252.7 s | 4.96 GiB |
| densify+prune(0.4) | 0.571455 | 0.699186 | 252.1 s | 4.84 GiB |

## 7. 해석

### 7.1 Densification은 거의 중립

Densify-only는 topology를 scope당 263--464 row 순증시켰지만 weighted mIoU는
`-0.001040`이었다. 현재 독립 ref-to-scene protocol에서는 reference에서 시작한
약 128만 Gaussian topology가 이미 충분하고, gradient-only 추가 capacity가
주요 병목은 아니었다.

### 7.2 `opacity < 0.4` prune은 과도함

Clone/split 후보는 수백 개인 반면 prune은 SC1/SC2/SC3에서 각각
18,428/41,193/37,722 row를 제거했다. 결과는 특히 recall이 감소했고 weighted
mIoU가 fixed topology보다 `-0.006426` 낮아졌다.

새로 OPEN된 많은 row는 base opacity가 0.4보다 낮거나 update 4까지 충분히
학습되지 않은 상태다. 이를 즉시 제거하면 실제 cue-support row까지 사라진다.
또 topology가 detector hypothesis space 자체이므로 pruning 뒤 alpha-T 책임이
다른 row로 재분배되어 lifecycle OPEN/CLOSE count도 fixed-topology 조건과 달라졌다.

따라서 이 결과는 `densified child만 제거해야 한다`는 뜻이 아니다. 모든 row를
동등하게 처리한 상태에서도 **prune criterion과 timing이 부적절했다**는 뜻이다.

### 7.3 Detector와 representation failure를 분리할 수 없음

이 ablation에서는 mutable geometry/opacity/topology가 다음 frame detector
evidence에 영향을 준다. 따라서 mIoU 차이는 representation capacity만의 효과가
아니며 detector observability 변화까지 포함한다. 이는 equal-status design의
직접적인 결과다.

## 8. Integrity 결과

모든 9개 full run에서 다음을 확인했다.

```text
future training view access = 0
inactive/off-view gradient violation = 0
CLOSED parameter/Adam drift = 0
active->active false lifespan split = 0
reused lifespan slot violation = 0
parameter/optimizer/filter/controller/lifecycle topology alignment = valid
features_rest max gradient = 0 (SH degree 0)
```

## 9. 실행 명령

```bash
# fixed topology matched baseline
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_dynamic_active_oscd_density \
  --scope scene_change1 \
  --density-policy none \
  --output-dir outputs/.../none/scene_change1 \
  --skip-checkpoint

# source-faithful online O-SCD clone/split, no extra pruning
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_dynamic_active_oscd_density \
  --scope scene_change1 \
  --density-policy active_oscd \
  --min-opacity 0 \
  --output-dir outputs/.../active_oscd_densify_only/scene_change1 \
  --skip-checkpoint

# requested active-only opacity pruning extension
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_dynamic_active_oscd_density \
  --scope scene_change1 \
  --density-policy active_oscd \
  --min-opacity 0.4 \
  --output-dir outputs/.../active_oscd/scene_change1 \
  --skip-checkpoint
```

결과 root:

```text
outputs/escd_dynamic_active_oscd_density_u16_seed0_20260825/
  comparison.json
  comparison.csv
  comparison.md
  none/{scene_change1,scene_change2,scene_change3}/
  active_oscd_densify_only/{scene_change1,scene_change2,scene_change3}/
  active_oscd/{scene_change1,scene_change2,scene_change3}/
```

## 10. 현재 판단

Equal-status dynamic topology와 Bayesian state inheritance는 정상 동작한다.
하지만 이 protocol에서는 densification gain이 없었고, `opacity < 0.4`를 frame
내 update 4에서 반복 적용하는 pruning은 버려야 한다. 다음 pruning 실험을 한다면
root/child 보호가 아니라 ACTIVE row 공통으로 다음 중 하나를 독립 ablation해야 한다.

```text
충분한 active-visible update 이후에만 prune
낮은 opacity가 여러 causal observation 동안 지속된 경우만 prune
prune threshold를 0.4보다 낮추기
```

이 셋을 동시에 넣어 원인을 섞지 않는다.
