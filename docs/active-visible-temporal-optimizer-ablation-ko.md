# Active-visible temporal optimizer ablation

## 질문

현재 lifespan이 OPEN이라는 이유만으로, 현재 카메라에 보이지 않는 Gaussian의 temporal parameter와 Adam moment까지 업데이트해야 하는가?

결론은 아니다. Representation 최적화 대상은 다음 교집합이어야 한다.

```text
current lifespan OPEN
AND
current temporal render에서 radius > 0
```

Detector는 계속 immutable reference alpha-T evidence를 사용한다. 이 visibility gate는 representation optimizer에만 적용하며 detector evidence에는 영향을 주지 않는다.

## 이전 구현의 문제

이전 runner는 매 frame 모든 OPEN row-slot을 optimizer mask로 전달했다.

```python
active_pairs = current_pair_mask(model)
optimizer.step(active_pairs)
```

현재 view 밖의 row는 보통 gradient가 0이지만, Adam에 과거 momentum이 남아 있으면 zero-gradient step에서도 parameter와 moment가 변할 수 있다. 따라서 CLOSED pair의 exact freeze는 보장됐지만, OPEN이면서 off-view인 pair의 exact freeze는 보장되지 않았다.

## 수정 계약

`temporal/masked_optimizer.py`에 `active_visible_pair_mask()`를 추가했다.

```python
update_pairs = active_pairs & (render_package["radii"] > 0)[:, None]
optimizer.step(update_pairs)
```

`render_change_temporal()` 결과의 `radii`는 현재 state-local xyz/opacity/scale/rotation으로 계산된다. Geometry update 뒤 visibility가 달라질 수 있으므로 120번의 각 update 직전에 mask를 다시 계산한다.

다음 row-slot은 optimizer-owned `step`, `exp_avg`, `exp_avg_sq`까지 그대로 유지한다.

- CLOSED 또는 inactive pair
- OPEN이지만 current view에서 radius가 0인 pair

## 회귀 테스트

새 synthetic test는 다음 순서를 검증한다.

1. OPEN row 0이 보일 때 momentum을 만든다.
2. row 0은 OPEN 상태를 유지하지만 다음 view에서 보이지 않게 한다.
3. visible row 1만 여러 번 업데이트한다.
4. row 0의 parameter, step, first/second moment가 bitwise 동일한지 확인한다.
5. row 0이 다시 보이면 의도적으로 최적화 가능한지 확인한다.

실행 결과:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  conda run -n oscd pytest -q tests

365 passed
```

CUDA 5-frame smoke에서도 timestamp 3/4의 off-view OPEN row `96/238`개가 optimizer에서 제외됐고, base/closed drift와 unselected gradient violation은 모두 0이었다.

## ESCD 304-frame 비교

공통 조건:

- Instance 1 `ref -> SC1 -> SC2 -> SC3`
- binary O-SCD pixel + SAM cue
- direct binary filter, K3/BF3 controller
- capped immutable-reference alpha-T evidence
- 120 updates/frame, seed 0
- fixed topology, densification/pruning OFF
- 같은 lifecycle decision sequence

### 결과

| Condition | Optimizer mask | mean mIoU | mean F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| B0 DC-only 이전 | all OPEN | 0.5981 | 0.7201 | 0.7273 | 0.8673 |
| B0 DC-only 수정 | OPEN + visible | 0.5922 | 0.7156 | 0.7185 | 0.8655 |
| B1 all-geometry 이전 | all OPEN | 0.5996 | 0.7204 | 0.6847 | 0.9251 |
| B1 all-geometry 수정 | OPEN + visible | 0.6021 | 0.7215 | 0.6864 | 0.9256 |

Lifecycle signature는 네 조건 모두 동일하다.

```text
OPEN   123,476
CLOSE   64,619
REOPEN  22,212
final active GS 58,857
```

따라서 성능 변화는 detector/lifecycle 변화가 아니라 optimizer visibility contract의 영향이다.

### 실제 선택량

304 frame 동안의 active row-frame 관측 합은 `12,100,115`였다.

| Condition | optimizer selected | off-view excluded | selected fraction |
|---|---:|---:|---:|
| B0 DC-only | 4,490,765 | 7,609,350 | 37.1% |
| B1 all-geometry | 3,792,492 | 8,307,623 | 31.3% |

B1의 state-local geometry가 바뀌므로 temporal render visibility도 B0와 달라질 수 있다.

## 해석

- B0 하락은 과거 view의 DC momentum을 off-view frame에서도 계속 적용하던 효과가 제거되면서 나타났다. 이 효과는 성능에 일부 유리했지만 현재 이미지로 정당화되지 않는 update였다.
- B1은 off-view geometry drift를 막자 mIoU가 `+0.00245` 상승했다. Geometry가 현재 관측과 무관하게 계속 움직이는 현상을 제거한 것이 소폭 유리했다.
- active-visible isolation은 detector를 개선하지 않는다. OPEN/CLOSE/REOPEN과 change-point failure는 동일하다.
- `radii > 0`는 renderer의 in-view visibility 정의다. 완전 occlusion까지 엄밀히 거르는 alpha-T contribution gate는 별도 ablation이 필요하다.

## Beam-2 동일 ablation

동일한 active-visible optimizer contract를 protected reset-candidate Beam-2에도 적용했다. Detector 설정, cue, camera, seed, 120-update schedule은 기존 Beam-2 결과와 동일하다.

| Condition | Optimizer mask | mean mIoU | mean F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| Beam-2 DC-only 이전 | all OPEN | 0.6047 | 0.7281 | 0.7278 | 0.8589 |
| Beam-2 DC-only 수정 | OPEN + visible | 0.5991 | 0.7237 | 0.7195 | 0.8572 |
| Beam-2 all-geometry 이전 | all OPEN | 0.6140 | 0.7323 | 0.6856 | 0.9297 |
| Beam-2 all-geometry 수정 | OPEN + visible | 0.6082 | 0.7275 | 0.6823 | 0.9285 |

Beam-2 lifecycle signature도 수정 전후 동일하다.

```text
OPEN   153,458
CLOSE   39,852
REOPEN  10,041
```

실제 optimizer 선택량:

| Condition | optimizer selected | off-view excluded | selected fraction |
|---|---:|---:|---:|
| Beam-2 DC-only | 8,488,703 | 14,992,094 | 36.2% |
| Beam-2 all-geometry | 7,040,226 | 16,440,571 | 30.0% |

Beam-2에서는 DC와 all-geometry가 모두 약 `0.0055~0.0058` mIoU 하락했다. 이는 이전 score의 일부가 현재 view에서 관측되지 않은 OPEN row의 retained Adam momentum에 의존했음을 보여준다. 다만 올바른 isolation 이후에도 all-geometry `0.6082`가 DC-only `0.5991`보다 `+0.0091` 높고, K3/BF3 all-geometry `0.6021`보다도 `+0.0061` 높다. 따라서 Beam-2의 상대적 detector/representation 이점은 남아 있다.

## 출력

```text
outputs/escd_view_consistent_k3bf3_active_visible_b0_dc_u120_20260824/
outputs/escd_view_consistent_k3bf3_active_visible_b1_all_geometry_u120_20260824/
outputs/escd_beam2_active_visible_dc_u120_20260824/
outputs/escd_beam2_active_visible_all_geometry_u120_20260824/
```

생성된 output/checkpoint는 git에 포함하지 않는다.
