# Persistent direct R_change lifespan ablation

## 질문

기존 `TemporalGeometryChangeModel`은 Gaussian마다 lifespan slot별로 다음
학습값을 분리했다.

```text
state_change_dc[N,S]
state_xyz_delta[N,S]
state_opacity_delta[N,S]
state_scaling_delta[N,S]
state_rotation_delta[N,S]
```

이번 ablation은 lifespan을 **표현 파라미터 저장소가 아니라 가시성
interval로만** 사용한다. Gaussian마다 하나의 mutable `R_change` parameter
set만 두고 모든 episode가 이를 공유한다.

## 표현 계약

각 Gaussian의 직접 학습값은 다음과 같다.

```text
change DC       _features_dc [N,1,3]
xyz             _xyz         [N,3]
SH rest         _features_rest[N,K,3]
opacity raw     _opacity     [N,1]
scaling raw     _scaling     [N,3]
rotation raw    _rotation    [N,4]
```

별도의 state-local DC나 geometry delta는 없다. Lifespan slot에는 다음
interval metadata만 남는다.

```text
state_start, state_end, state_valid, state_status
num_states, current_state_index
```

렌더 시 OPEN row만 보이며 직접 parameter를 사용한다.

```text
DC       = change_dc
xyz      = xyz
opacity  = sigmoid(opacity_raw) * is_open
scaling  = exp(scaling_raw)
rotation = normalize(rotation_raw)
```

## CLOSE와 REOPEN

예를 들어 change DC가 `0.97`까지 학습된 row가 닫히면:

```text
CLOSE:
    current opacity gate = 0
    change DC remains 0.97
    geometry remains unchanged
    Adam moments remain unchanged
```

나중에 다시 열릴 때에는 interval slot만 새로 할당한다.

```text
REOPEN:
    new interval slot is allocated
    change DC resumes from 0.97
    direct geometry and Adam moments also resume
```

따라서 historical interval은 보존되지만 historical representation snapshot은
보존하지 않는다. REOPEN 이후 direct parameters가 다시 변하면 과거 CLOSED
episode의 당시 geometry를 재생할 수 없다. 이것은 이번 ablation의 의도적인
trade-off다.

## Causal separation

Detector는 별도의 immutable reference bank로 alpha-T evidence를 계산한다.
Mutable `R_change`의 DC/geometry/lifespan은 detector 입력에 사용되지 않는다.

```text
immutable R_ref + current cue
        -> alpha-T evidence
        -> direct binary filter / K3-BF3 controller
        -> interval OPEN/CLOSE
        -> mutable persistent R_change render/optimization
```

Immutable detector reference의 여섯 tensor drift는 모든 실험에서 정확히
0이었다.

## Optimizer isolation

`MaskedRowAdam`은 각 update에서 다음 row만 갱신한다.

```text
currently OPEN AND radius(current training view) > 0
```

선택되지 않은 row는 parameter뿐 아니라 Adam `step`, `exp_avg`,
`exp_avg_sq`도 갱신하지 않는다. CLOSE부터 REOPEN 직전까지의 값과 optimizer
state를 전수 snapshot/검증했으며 최대 drift는 0이었다.

`features_rest`도 direct optimizer parameter로 보존하지만 현재
`render_change`는 source O-SCD와 동일하게 SH degree 0을 사용한다. 따라서
이번 change-mask loss에서는 해당 gradient가 정확히 0이다. Directional SH
change score나 RGB reconstruction objective는 이번 실험에 추가하지 않았다.

## 공정 비교

비교 대상은 density-off state-local all-geometry baseline이다.

```text
Dataset       Instance_1 independent ref->SC1, ref->SC2, ref->SC3
Cue           identical cached O-SCD pixel+SAM binary cue
Detector      identical direct binary filter
Controller    identical K=3, BF>=3
Updates/frame 120
Seed          0
Pose          identical fixed cameras
Topology      fixed
GT            inference 종료 후 평가에만 사용
```

따라서 scope별 OPEN/CLOSE/REOPEN 수는 두 표현에서 정확히 동일하다.

## 결과

| Scope | state-local mIoU | persistent mIoU | Δ | state-local F1 | persistent F1 | Δ | OPEN/CLOSE/REOPEN |
|---|---:|---:|---:|---:|---:|---:|---:|
| SC1 | 0.555488 | 0.555712 | +0.000224 | 0.653767 | 0.654060 | +0.000293 | 33,246 / 5,078 / 1,403 |
| SC2 | 0.614383 | 0.617399 | +0.003016 | 0.743325 | 0.745691 | +0.002366 | 66,995 / 13,767 / 2,748 |
| SC3 | 0.591943 | 0.592957 | +0.001014 | 0.725348 | 0.725964 | +0.000616 | 67,948 / 20,762 / 3,937 |

Frame-count weighted result:

```text
mIoU 0.588227 -> 0.589679  (+0.001452)
F1   0.709129 -> 0.710243  (+0.001114)
```

Persistent reuse는 세 scope 모두 소폭 상승했지만 detector chattering으로
인한 많은 CLOSE/REOPEN을 해결하지는 않는다. 즉 zero-initialized state slot의
재학습 비용은 일부 제거했지만 현재 성능의 주 병목은 여전히 lifecycle
transition 품질이다.

## 메모리와 시간

| Scope | state-local runtime | persistent runtime | 변화 | state-local peak | persistent peak | 변화 |
|---|---:|---:|---:|---:|---:|---:|
| SC1 | 277.1 s | 170.4 s | -38.5% | 7.72 GiB | 2.92 GiB | -62.2% |
| SC2 | 304.7 s | 196.8 s | -35.4% | 7.81 GiB | 2.95 GiB | -62.3% |
| SC3 | 315.0 s | 202.9 s | -35.6% | 7.84 GiB | 2.93 GiB | -62.6% |

큰 감소는 `[N,S,...]` geometry/DC parameter 및 optimizer state를
`[N,...]` direct bank로 바꾼 결과다.

## 무결성 결과

모든 scope에서:

```text
immutable detector reference drift = 0
CLOSED-duration direct parameter drift = 0
CLOSED-duration Adam-state drift = 0
inactive/current-view-invisible gradient violation = 0
active->active false split = 0
lifespan slot reuse violation = 0
features_rest gradient = 0 (SH degree 0 limitation)
```

## 실행

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_persistent_gaussian_lifespan \
  --scope scene_change1 \
  --updates-per-frame 120 \
  --output-dir outputs/persistent_direct/scene_change1 \
  --skip-checkpoint
```

`scene_change2`, `scene_change3`도 같은 명령에서 `--scope`만 바꾼다.

검증:

```bash
PYTHONPATH=. conda run -n oscd python -m compileall -q \
  experiments poses temporal tests gaussian_renderer scene utils

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  conda run -n oscd pytest -q tests
```

실험 출력은 다음 경로에 있으며 저장소에는 commit하지 않는다.

```text
outputs/escd_independent_persistent_direct_rchange_u120_seed0_20260825/
```

이번 runner는 representation 차이만 분리하기 위해 fixed topology를 사용한다.
기존 active-residual FastGS density integration은 동시에 적용하지 않았다.
