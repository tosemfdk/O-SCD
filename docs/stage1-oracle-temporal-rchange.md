# Stage 1 — Oracle Temporal `R_change` Vertical Slice

> 상태: **E3 구현·검증 완료 — lifespan DC, corrected cue, state-specific geometry**
> 목표: changepoint를 정확히 안다고 가정했을 때, timestamp-conditioned temporal `R_change`가 서로 다른 시간대의 change cue 충돌을 분리할 수 있는지 검증한다.

현재 checkpoint는 `state_change_dc`, `state_start`, `state_end`, `state_valid`와 state별 `xyz`/`opacity`/`scale`/`rotation` delta를 포함한다. 아래 전체 Stage 1 계약의 `state_status`, `num_states`, `transition_state()`는 automatic lifecycle transition을 구현하는 다음 단계까지 명시적으로 보류한다. 따라서 이후 절에서 이 metadata를 “필수”라고 표현한 부분은 **완성된 Stage 1의 목표 계약**을 뜻하며 현재 oracle-boundary 구현 계약은 아니다. Oracle boundary는 manual global segment를 제공할 뿐이며 BOCD 또는 event detector는 구현하지 않는다.

- 현재 결과와 시각화: [`temporal-lifespan-smoke.md`](temporal-lifespan-smoke.md)
- E3 결과: [`instance1-state-geometry-lifespan-comparison-ko.md`](instance1-state-geometry-lifespan-comparison-ko.md)
- 현재 구현 범위: checked selector, temporal DC/geometry sidecar, renderer attribute override와 opacity gating, CUDA gradient isolation, exact-120 runner와 checkpoint
- 정리된 계약: `-1` local state는 tensor에 남기되 renderer에서 zero effective opacity로 숨긴다.
- 아직 미구현: automatic state transition, online-loop integration, BOCD

## 1. 이번 단계가 답해야 하는 질문

Stage 1의 핵심 질문은 하나다.

> 동일한 base Gaussian geometry를 공유하면서 Gaussian별 change DC를 시간 state로 분리하고, 각 저장 프레임을 자신의 timestamp에 해당하는 state로 replay하면 기존 single-state O-SCD의 cue averaging을 제거할 수 있는가?

이번 단계에서는 changepoint detector의 성능을 평가하지 않는다. Oracle boundary를 사용하여 **temporal representation과 timestamp-conditioned optimization만 독립적으로 검증**한다.

성공의 최소 증거는 synthetic A -> B -> A sequence에서 다음이 모두 성립하는 것이다.

1. temporal model의 전체 replay loss가 single-state model보다 낮다.
2. B segment가 두 A segment와 별도 local lifespan slot으로 유지된다.
3. 과거 frame replay가 당시 state만 업데이트한다.
4. 마지막 A state가 과거 B evidence에 오염되지 않는다.
5. `--temporal_mode off`에서는 기존 O-SCD 경로가 유지된다.

---

## 2. 현재 코드에서 확인한 문제 구조

아래 내용은 추정이 아니라 현재 저장소 구현을 직접 확인한 결과다.

| 현재 위치 | 확인한 동작 | Stage 1 설계에 미치는 영향 |
|---|---|---|
| `oscd.py::main()` | 새 view의 `candidate_map`을 만든 뒤 즉시 `viewpoints.append(view)`를 수행한다. | boundary 처리는 replay buffer 삽입과 fusion보다 먼저 실행해야 한다. |
| `oscd.py::main()` | frame마다 16회, 현재 view 또는 과거 view를 뽑아 하나의 `gaussians_change`를 업데이트한다. | replay frame의 timestamp로 state를 선택하지 않으면 B와 C가 다시 같은 parameter에서 충돌한다. |
| `oscd.py::main()` | online loop와 refine loop에 SSF loss가 인라인으로 중복되어 있다. | temporal renderer와 baseline이 같은 목적함수를 쓰는지 비교할 수 있도록 loss를 먼저 분리해야 한다. |
| `scene/gaussian_model.py::training_setup_change()` | xyz, DC/rest feature, opacity, scale, rotation을 모두 optimizer에 등록한다. | temporal MVP에서는 base topology와 geometry를 freeze하고 temporal DC만 학습해야 한다. |
| `oscd.py::main()` | online fusion 중 clone/split을 호출하고 refine에서는 densify/prune을 호출한다. | Gaussian identity가 바뀌므로 Stage 1 temporal mode에서는 densification을 금지한다. |
| `scene/gaussian_model.py::load_ply_change()` | `_features_dc`를 `[N, 1, 3]` zero tensor로 초기화한다. | temporal slot 0은 이 tensor를 그대로 복제하여 baseline과 같은 초기 조건을 가져야 한다. |
| `gaussian_renderer::render_change()` | rasterizer의 `dc` 입력으로 `pc._features_dc`를 직접 전달한다. | renderer 복제 없이 `override_dc`와 effective opacity mask를 추가하면 timestamp별 local state를 주입할 수 있다. |
| `gaussian_renderer::render_change()` | `override_color` 또는 특정 SH 변환 경로에서 `dc`가 명시적으로 초기화되지 않을 가능성이 있다. | `override_dc` 추가 시 `dc`, `shs`, `colors_precomp`의 상호배타적 경로를 명시적으로 검증해야 한다. |
| `ImageDataset` | `info`에 `name`, `is_test`만 기본 제공한다. | offline sequence index를 `frame_id`와 fallback timestamp로 제공해야 한다. |
| `StreamDataset` | queue에는 frame만 저장하며 capture time과 source frame id가 없다. | 소비 시각이 아닌 capture 시각을 queue item에 함께 저장해야 한다. |
| `Camera` | `timestamp`, `frame_id`, `segment_id`가 없다. | replay가 원래 frame의 시간 state를 선택할 수 있도록 backward-compatible 필드를 추가해야 한다. |

### 현재 replay sampling의 정확한 의미

현재 코드는 약 0.33 확률로 최신 frame을 강제로 선택하고, 나머지 경우 전체 `viewpoints`에서 균일 표본을 뽑는다. 전체 표본에도 최신 frame이 포함되므로 최신 frame의 실제 선택 확률은 정확히 1/3이 아니라 다음과 같다.

```text
0.33 + 0.67 / len(viewpoints)
```

Stage 1 regression에서는 이 **분기 정책 자체**를 유지해야 한다. 단순히 최신 frame 확률을 정확히 1/3로 재정의하면 baseline behavior change가 된다.

---

## 3. Stage 1의 범위

### 3.1 포함하는 것

- SSF loss의 순수 함수화
- frame timestamp와 segment metadata 전달
- 고정 크기 temporal change state slot
- 반열린 lifespan interval `[start, end)`
- optional logical unborn/absent/outdated lifespan gating (`-1` active state)
- manual oracle global boundary와 segment id
- timestamp-conditioned change rendering
- timestamp-conditioned replay와 fusion
- temporal parameter 전용 optimizer
- synthetic A -> B -> A conflict separation test
- temporal `.pt` checkpoint와 manifest
- `temporal_mode=off` regression gate

### 3.2 의도적으로 포함하지 않는 것

- BOCD와 run-length posterior
- scalar 또는 vector pre-fusion detector
- signed RGB residual
- SAM feature vector 보존
- primitive responsibility CUDA accumulation
- KNN graph와 spatial confirmation
- tentative / commit / rollback
- smooth lifespan gate
- physical Gaussian tensor 생성/삭제 및 lineage
- densification, clone, split, prune
- velocity 또는 deformation field
- 새 geometry reconstruction
- reference map overwrite
- `update.py` 또는 scene update pipeline 변경

이 항목들은 Stage 1 결과가 실패했을 때 원인을 흐리지 않기 위해 금지한다.

---

## 4. Stage 1 데이터 모델

### 4.1 Base Gaussian과 temporal state의 분리

Base change Gaussian은 다음 값만 제공한다.

```text
base geometry = xyz, opacity, scaling, rotation, feature_rest
```

Stage 1에서 학습 가능한 값은 temporal state별 DC 하나뿐이다.

```text
state_change_dc: [N, S_max, 1, 3]
```

Oracle boundary는 전체 stream을 global segment로 나눈다. 예를 들어 boundaries `[5, 10]`은 global segment `0/1/2`를 만든다. 그러나 segment id와 Gaussian별 local state slot은 같은 개념이 아니다. 각 Gaussian은 어떤 timestamp에서 local active state가 **0개 또는 1개**일 수 있다.

```text
active_state_index[g_i, t] = local_slot_index 또는 -1
```

`-1`은 해당 Gaussian이 그 timestamp에서 unborn, absent, 또는 outdated라는 뜻이다. 이 Gaussian은 tensor와 index identity에는 남아 있지만 rendering에서는 zero effective opacity로 처리한다.

필수 metadata는 다음과 같다.

```text
state_start:  [N, S_max] float
state_end:    [N, S_max] float
state_valid:  [N, S_max] bool
state_status: [N, S_max] int8
num_states:   [N]        long
```

Stage 1 status는 세 값이면 충분하다.

```text
EMPTY  = 0
OPEN   = 1
CLOSED = 2
```

Oracle state는 생성 즉시 확정된 것으로 취급한다. `TENTATIVE`와 `COMMITTED`는 이후 detector 단계에서 추가한다.

### 4.2 Interval 규칙과 global segment

모든 lifespan은 반열린 구간을 사용한다.

```text
[start, end)
```

boundary가 frame 5라면:

```text
segment 0: t < 5
segment 1: t >= 5
```

A -> B -> A, boundaries `[5, 10]`의 global segment interval은 다음과 같다.

```text
segment 0: [0, 5)
segment 1: [5, 10)
segment 2: [10, inf)
```

Gaussian별 local state는 이 segment 안에서 활성화되거나 비활성화될 수 있다.

```text
g_i local active slots by segment: [0, -1, 2]
```

segment 0과 segment 2의 DC 값이 우연히 같아도 두 lifespan을 같은 slot으로 collapse하지 않는다. Stage 1은 "값이 같은지"가 아니라 "서로 다른 시간 lifespan이 분리되는지"를 검증한다.

### 4.3 반드시 유지할 invariant

각 Gaussian에 대해:

1. valid local state는 slot index 순서대로 연속 배치된다.
2. 각 valid local state는 `start < end`를 만족한다.
3. valid local state끼리는 overlap하지 않는다.
4. gap은 허용하며, gap에서 query 결과는 `-1`이다.
5. 어떤 유효 query timestamp에서도 active local state는 0개 또는 1개다.
6. active local state가 2개 이상이면 invariant 위반이다.
7. `num_states`가 구현된 경우 valid slot 수와 같다.
8. OPEN state가 구현된 경우 많아야 하나이며, 존재하면 마지막 valid slot이다.
9. CLOSED state의 `end`는 finite 값이다.
10. `state_change_dc` Parameter 객체는 transition 전후 동일하다.

---

## 5. 전체 실행 흐름

### 5.1 초기화 흐름

```text
load fixed reference RGB Gaussians
    -> load change Gaussian scaffold
    -> parse temporal arguments
    -> if temporal_mode == off:
           기존 training_setup_change()와 기존 경로 유지
       else if temporal_mode == oracle_global:
           validate oracle boundaries and capacity
           TemporalChangeModel.from_gaussians()
           freeze base change Gaussian parameters
           create temporal-only optimizer
           initialize replay buffer and frame manifest
```

`temporal_mode=off`에서는 `TemporalChangeModel`을 만들지 않는다. 이 분리는 regression을 위한 필수 조건이다.

### 5.2 frame별 흐름

```text
1. image, info 수신
2. frame_id와 timestamp 결정
3. timestamp monotonicity 검증
4. pose 추정
5. Camera 생성 후 frame_id/timestamp 설정
6. 고정 R_ref에서 RGB render
7. 기존 generate_candidate_map()으로 candidate cue 생성
8. manual oracle boundary인지 확인하고 global segment 갱신
9. boundary이면 fusion 전에 Gaussian별 local lifespan을 닫거나 연다
10. segment_id 설정
11. view를 replay buffer에 추가
12. 16회 timestamp-conditioned replay/fusion
13. 현재 timestamp의 active local state와 zero-opacity inactive mask로 change mask render
14. frame metadata를 manifest에 기록
```

Stage 1에는 학습 기반 changepoint detector가 없다. Manual boundary를 consume하는 순서만 향후 구조와 동일하게 유지한다.

```text
boundary 결정 first -> state assignment -> fusion second
```

### 5.3 timestamp-conditioned replay

replay frame을 선택한 뒤 절대 현재 frame의 timestamp를 덮어쓰지 않는다.

```text
frame 8  replay -> timestamp 8의 state
frame 25 replay -> timestamp 25의 state
```

이 규칙이 깨지면 과거 B frame이 현재 C state를 업데이트하여 temporal separation이 다시 무너진다.

### 5.4 refine 흐름

`--refine`을 temporal mode와 함께 지원한다면, refine도 반드시 각 stored view의 timestamp를 사용해 `temporal_fusion_step()`을 호출해야 한다. temporal refine에서는 densify/prune과 opacity reset을 호출하지 않는다.

구현 복잡도를 줄이기 위해 Stage 1에서 temporal refine을 지원하지 않기로 결정한다면 silent fallback을 금지하고 명시적 `ValueError`를 발생시켜야 한다. 구현 시 둘 중 하나를 선택하고 test로 고정한다.

### 5.5 출력 흐름

현재 O-SCD의 mask 후처리를 Stage 1에서 임의로 바꾸지 않는다.

```text
rendered_change_rgb
    -> RGB channel mean
    -> threshold 0.5
    -> current frame change mask
```

loss 내부 sigmoid와 최종 출력 threshold의 기존 차이는 Stage 1에서 별도 알고리즘 변경 대상으로 삼지 않는다.

---

## 6. 함수별 구현 계약

아래 순서는 의존성 순서이자 실제 구현 순서다. 앞 단계 test가 통과하기 전에는 다음 단계로 넘어가지 않는다.

### 6.1 `compute_ssf_loss()`

**권장 위치:** `utils/loss_utils.py`

```python
def compute_ssf_loss(
    candidate_map: torch.Tensor,
    rendered_change_rgb: torch.Tensor,
    *,
    regularizer_offset: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
```

#### 책임

- renderer 종류와 무관한 SSF 목적함수를 한 곳에서 계산한다.
- 기존 online loop의 수식을 기본 동작으로 정확히 유지한다.
- temporal renderer와 baseline renderer의 loss 비교 기준이 된다.

#### 계산 계약

```text
change_mask = sigmoid(mean(rendered_change_rgb, RGB channel))
d_loss      = mean(candidate_map * (1 - change_mask))
d_reg       = log(mean(change_mask)^2 + regularizer_offset)
loss        = d_loss + d_reg
```

반환 dict의 최소 key:

```text
loss
d_loss
d_reg
change_mask
```

#### 이 함수가 먼저 필요한 이유

temporal change의 효과와 loss 식 변경의 효과가 섞이지 않게 한다. 기존 식을 먼저 고정해야 renderer와 state representation만 비교할 수 있다.

#### 현재 코드에서 주의할 점

online loop는 regularizer에 `+1.0`, refine loop는 `+1.000000001`을 사용한다. 두 경로를 하나의 고정 signature로 치환하면 둘 중 하나의 수치가 바뀐다. 따라서 optional `regularizer_offset`으로 기존 값을 각각 전달하거나, Stage 1에서는 refine 치환을 보류해야 한다. 이 차이를 숨기지 않는다.

#### 테스트

- 기존 인라인 online 식과 forward 값 비교
- 기존 인라인 refine 식과 해당 offset forward 값 비교
- `rendered_change_rgb` gradient 비교
- CPU `atol=1e-7`
- CUDA `atol=1e-6`
- shape mismatch에 명시적 오류
- NaN/Inf 입력 또는 결과 정책을 명확히 고정

#### 완료 조건

새 함수 적용 전후 동일 입력의 loss와 gradient가 tolerance 내 동일하다.

---

### 6.2 `resolve_frame_timestamp()`

**권장 위치:** `utils/time_utils.py`

```python
def resolve_frame_timestamp(
    frame_id: int,
    info: dict,
) -> float:
    ...
```

#### 책임

frame에 사용할 canonical timestamp를 다음 우선순위로 결정한다.

```text
info["timestamp"]
-> info["capture_timestamp"]
-> float(frame_id)
```

#### 검증 계약

- 결과는 finite `float`여야 한다.
- bool, NaN, Inf, 문자열 timestamp는 명시적으로 거부한다.
- timestamp의 단위는 source metadata가 정의하며 Stage 1 offline oracle은 frame index를 사용한다.

#### 필요한 이유

lifespan query, boundary 처리, replay state 선택, checkpoint 재현성의 공통 시간축을 만든다.

#### 테스트

- `timestamp`가 `capture_timestamp`보다 우선
- `capture_timestamp` fallback
- metadata가 없으면 frame id fallback
- invalid numeric value 오류
- input dict를 mutation하지 않음

---

### 6.3 `validate_monotonic_timestamp()`

**권장 위치:** `utils/time_utils.py`

```python
def validate_monotonic_timestamp(
    timestamp: float,
    previous_timestamp: float | None,
    *,
    allow_equal: bool = False,
) -> None:
    ...
```

#### 책임

`resolve_frame_timestamp()`와 별도로 stream 순서 invariant를 검증한다. Stateless resolver 하나만으로는 이전 timestamp를 알 수 없으므로 두 책임을 분리한다.

#### 규칙

- 첫 frame은 항상 허용한다.
- 기본값에서는 `timestamp > previous_timestamp`여야 한다.
- duplicate timestamp를 허용해야 하는 dataset이라면 `allow_equal=True`를 명시한다.
- 역행은 silent sorting하지 않고 즉시 오류로 처리한다.

#### 필요한 이유

역행 timestamp는 interval transition을 되돌리고 과거 state를 다시 열 수 있으므로 temporal state를 손상시킨다.

#### 테스트

- 첫 timestamp 허용
- 증가 허용
- 동일 값의 옵션별 동작
- 감소 오류

---

### 6.4 `ImageDataset.__getitem__()` metadata 확장

**수정 위치:** `dataloaders/image_dataset.py`

#### 변경 계약

offline dataset의 각 `info`에 다음을 제공한다.

```text
frame_id: int
timestamp: float(frame_id)
```

실제 capture timestamp metadata를 읽는 기능이 추후 생기면 그 값을 우선하되, Stage 1에서는 정렬된 image sequence index가 canonical time이다.

#### 필요한 이유

prefetch thread와 main loop의 local loop counter에만 의존하지 않고, frame 자체가 자신의 시간 identity를 가져야 한다.

#### 테스트

- 정렬된 image name 순서와 frame id 일치
- prefetch 후에도 반환 순서와 timestamp 일치
- 기존 `name`, `is_test`, pose metadata 유지

---

### 6.5 `StreamDataset._capture_frames()`와 `getnext()` metadata 확장

**수정 위치:** `dataloaders/stream_dataset.py`

#### queue item 계약

```text
(frame, capture_timestamp, source_frame_id)
```

`capture_timestamp`는 `cap.read()` 성공 직후 `time.monotonic()`으로 기록한다. `source_frame_id`는 capture thread에서 성공한 frame마다 증가시킨다. 소비 횟수인 기존 `num_frames`를 source frame identity로 재사용하지 않는다.

#### 필요한 이유

queue size가 1이라 오래된 frame을 버릴 수 있다. 따라서 소비 index는 실제 source frame 순서를 나타내지 못하며, 처리 지연도 capture time과 다르다.

#### 테스트

- capture time과 consume time이 달라도 capture time 반환
- queue overwrite 후 source frame id가 건너뛸 수 있음을 허용
- source frame id는 역행하지 않음
- `get_image_size()` 호출이 metadata 의미를 손상시키지 않는지 검증

---

### 6.6 `Camera` temporal metadata

**수정 위치:** `scene/cameras.py`

#### backward-compatible optional field

```python
frame_id: int = -1
timestamp: float = 0.0
segment_id: int | None = None
```

#### 책임

replay buffer에 저장된 view가 원래 시간과 oracle segment를 유지하게 한다.

#### 필요한 이유

timestamp를 외부 dictionary에만 두면 view sampling 이후 원래 frame과 state association이 깨지기 쉽다.

#### 테스트

- 기존 positional/keyword constructor 호출 호환
- 새 metadata 보존
- replay 후 값 불변

#### CPU test 주의

현재 `Camera` 구현은 transform 생성에서 `.cuda()`를 직접 호출한다. 따라서 pure CPU temporal test는 실제 `Camera` 대신 필요한 attribute만 가진 mock view를 사용한다. Camera 전체의 CPU 지원은 Stage 1 목표가 아니다.

---

### 6.7 `TemporalChangeModel.from_gaussians()`

**새 파일:** `scene/temporal_change_model.py`

```python
class TemporalChangeModel(torch.nn.Module):
    @classmethod
    def from_gaussians(
        cls,
        base_change_gaussians: GaussianModel,
        max_states: int = 4,
        initial_time: float = 0.0,
    ) -> "TemporalChangeModel":
        ...
```

#### 책임

- fixed base Gaussian scaffold를 참조한다.
- Gaussian별 고정 크기 temporal slot을 한 번만 할당한다.
- slot 0을 기존 `_features_dc`와 동일한 값으로 초기화한다.
- metadata tensor를 buffer로 등록한다.

#### 초기 상태 계약

```text
state_change_dc[:, 0] = base._features_dc.detach().clone()
state_start[:, 0]     = initial_time
state_end[:, 0]       = +inf
state_valid[:, 0]     = True
state_status[:, 0]    = OPEN
num_states[:]         = 1
나머지 slot           = EMPTY / invalid
```

#### device 계약

- `device="cuda"`를 hardcode하지 않는다.
- device와 dtype은 `base._features_dc`에서 가져온다.
- pure state logic은 CPU dummy base에서도 생성 가능해야 한다.

#### 필요한 이유

changepoint마다 새 `nn.Parameter`를 만들면 optimizer reference와 Adam state가 끊어진다. 전체 slot을 먼저 할당하면 parameter identity를 유지하면서 state만 열 수 있다.

#### 테스트

- `N=5`, `S=4` shape
- slot 0 값이 base DC와 동일
- 나머지 slot invalid
- `state_change_dc.requires_grad=True`
- metadata는 gradient 없음
- state dict에 parameter와 buffers 포함
- CPU dummy base에서 생성 가능

---

### 6.8 `freeze_base_parameters()`

**위치:** `TemporalChangeModel`

```python
def freeze_base_parameters(self) -> None:
    ...
```

#### freeze 대상

```text
base._xyz
base._features_dc
base._features_rest
base._opacity
base._scaling
base._rotation
```

#### 책임

temporal mode에서 renderer가 사용하는 base Gaussian tensor가 optimizer 또는 accidental backward로 변경되지 않게 한다.

#### 필요한 이유

geometry가 frame마다 움직이거나 topology가 바뀌면 Gaussian index가 동일 surface identity를 나타내지 못한다. 또한 pose error가 geometry update로 흡수되면 temporal state 효과를 해석하기 어렵다.

#### 테스트

- 모든 base parameter의 `requires_grad=False`
- temporal DC만 `requires_grad=True`
- fusion step 전후 base tensor bitwise 또는 tolerance 내 동일

---

### 6.9 `get_active_state_indices()`

**위치:** `TemporalChangeModel`

```python
def get_active_state_indices(
    self,
    timestamp: float,
) -> torch.Tensor:
    ...
```

#### 출력 계약

```text
LongTensor[N]
```

각 Gaussian에서 `start <= timestamp < end`를 만족하는 local state index를 반환한다. 해당 timestamp에 active local state가 없으면 `-1`을 반환한다.

#### 오류 조건

- active state 2개 이상
- non-finite timestamp
- `initial_time`보다 이른 query

#### 필요한 이유

render, replay, checkpoint validation이 모두 같은 interval 선택 규칙을 사용하게 한다.

#### 테스트

```text
t=0       -> state 0
t=4.999   -> state 0
t=5       -> state 1
t=9.999   -> state 1
t=10      -> state 2
t=100     -> state 2
inactive gap -> -1
```

---

### 6.10 `get_active_change()`

**위치:** `TemporalChangeModel`

```python
def get_active_change(
    self,
    timestamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

#### 출력 계약

```text
Tensor[N, 1, 3]
BoolTensor[N] active_mask
```

#### 책임

`get_active_state_indices()` 결과로 active local state의 `state_change_dc`를 gather한다. `-1` entry는 마지막 slot을 accidental gather하지 않고 zero placeholder와 `active_mask=False`로 처리한다. Active 반환 tensor는 autograd graph를 유지해야 하며 `.detach()`하지 않는다.

`get_active_change_dc(timestamp)`가 별도로 남는다면 DC tensor만 반환하는 convenience API로 둔다. Tuple을 반환하는 API 이름은 `get_active_change()`로 고정한다.

#### 필요한 이유

현재 timestamp에 해당하는 slot만 renderer와 loss에 연결하여 inactive slot gradient를 zero로 만든다. Inactive Gaussian은 tensor에는 남지만 temporal render에서 zero effective opacity가 되어야 한다.

#### 테스트

- timestamp별 올바른 DC 선택
- inactive Gaussian `active_mask=False`
- 출력 shape/dtype/device
- backward 시 active slot만 non-zero gradient
- inactive slot gradient exactly zero

---

### 6.11 `transition_state()`

**위치:** `TemporalChangeModel`

이 helper는 active local state를 닫고 같은 Gaussian의 새 active local state를 여는 **active -> active split**만 담당한다. `-1 -> active` birth와 `active -> -1` close-only는 oracle lifespan application에서 별도 처리해야 하며, inactive Gaussian을 억지로 transition하지 않는다.

```python
def transition_state(
    self,
    gaussian_mask: torch.Tensor,
    timestamp: float,
    init_mode: str = "copy_active",
) -> torch.Tensor:
    ...
```

#### 입력 계약

- `gaussian_mask`: `[N]` bool, model metadata와 같은 device
- `timestamp`: 모든 선택 Gaussian의 현재 open interval 내부이며 start보다 커야 함
- `init_mode`: `copy_active` 또는 `zeros`

#### 반환 계약

`LongTensor[N]`을 반환하며 transition하지 않은 Gaussian은 `-1`, transition한 Gaussian은 새 slot index를 가진다.

#### atomic 처리 순서

mutation 전에 모든 선택 Gaussian에 대해 다음을 먼저 검증한다.

1. selected Gaussian에는 active local state가 정확히 하나 있는지
2. timestamp가 active start보다 큰지
3. free slot이 있는지
4. init mode가 유효한지

모든 검증이 통과한 뒤 `torch.no_grad()`에서 한 번에:

1. old state `end = timestamp`
2. old state `status = CLOSED`
3. new slot `start = timestamp`
4. new slot `end = inf`
5. new slot `valid = True`
6. new slot `status = OPEN`
7. DC를 copy 또는 zero로 초기화
8. `num_states += 1`

중간 실패로 일부 Gaussian만 transition된 상태를 남기면 안 된다.

#### 필요한 이유

oracle boundary에서 이전 evidence의 lifespan을 닫고 새 parameter lifespan을 여는 핵심 연산이다.

#### 테스트

- Gaussian subset transition
- unselected Gaussian 불변
- `copy_active` 값 복사
- `zeros` 초기화
- boundary가 새 state에 포함
- timestamp 역행 또는 동일 start 오류
- mask shape/dtype 오류
- state capacity 부족 오류
- 실패 시 전체 metadata 불변
- `id(state_change_dc)` 전후 동일
- optimizer 생성 후 transition해도 parameter reference 동일

---

### 6.12 `validate_invariants()`

**위치:** `TemporalChangeModel`

```python
def validate_invariants(self) -> None:
    ...
```

#### 책임

Section 4.3의 invariant를 전체 Gaussian에 대해 검증한다. 오류가 있으면 Gaussian index, slot index, 위반 종류를 포함한 명시적 예외를 발생시킨다.

#### 호출 시점

- model 생성 직후
- transition 직후 debug/test mode
- checkpoint 저장 전
- checkpoint load 직후

#### 테스트

- 정상 chain 통과
- overlap 검출
- gap은 허용하고 gap timestamp query가 `-1`인지 검증
- `end <= start` 검출
- OPEN state 여러 개 검출
- `num_states` mismatch 검출
- valid slot 사이 EMPTY slot 검출

---

### 6.13 `render_change(..., override_dc=None, override_opacity=None)`

**수정 위치:** `gaussian_renderer/__init__.py`

```python
def render_change(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    mult=0.5,
    scaling_modifier=1.0,
    override_color=None,
    get_flag=None,
    metric_map=None,
    override_dc=None,
    override_opacity=None,
):
    ...
```

#### 책임

기존 renderer를 복제하지 않고 DC 입력과 effective opacity만 외부에서 선택할 수 있게 한다.

#### 경로 규칙

- `override_dc is None`: 기존 `pc._features_dc` 경로 유지
- `override_dc` 제공: rasterizer의 `dc`에 해당 tensor 전달
- `override_opacity` 제공: rasterizer의 opacity에 해당 activated tensor 전달
- `override_color`와 `override_dc` 동시 제공: `ValueError`
- `dc`, `shs`, `colors_precomp`를 분기 전에 `None`으로 초기화
- override tensor는 `[N,1,3]`, pc와 같은 dtype/device여야 함
- `override_opacity` tensor는 `pc.get_opacity`와 같은 shape/dtype/device여야 함
- 모든 override가 `None`인 결과는 수정 전 renderer와 수치적으로 동일해야 함

#### 필요한 이유

geometry, covariance, rasterization을 그대로 재사용하면서 timestamp별 change representation만 교체할 수 있다. Inactive Gaussian은 pruning하지 않고 `override_opacity`에서 zero effective opacity로 숨긴다.

#### 테스트

- 기존 renderer equivalence
- invalid shape/dtype/device 오류
- override conflict 오류
- `convert_SHs_python` on/off 경로 smoke test
- active DC까지 gradient 전달
- inactive Gaussian opacity가 0으로 전달되는지 검증

---

### 6.14 `render_change_temporal()`

**권장 위치:** `gaussian_renderer/__init__.py` 또는 작은 temporal renderer module

```python
def render_change_temporal(
    view: Camera,
    temporal_model: TemporalChangeModel,
    pipe,
    background: torch.Tensor,
    timestamp: float | None = None,
):
    ...
```

#### 처리 순서

```text
timestamp = explicit timestamp or view.timestamp
active_dc, active_mask = temporal_model.get_active_change(timestamp)
effective_opacity = temporal_model.base.get_opacity * active_mask[:, None]
return render_change(
    ...,
    pc=temporal_model.base,
    override_dc=active_dc,
    override_opacity=effective_opacity,
)
```

#### 필요한 이유

모든 call site가 timestamp selection을 동일하게 수행하게 하며, 과거 replay가 현재 state를 잘못 업데이트하는 것을 막는다.

#### 테스트

- slot 0 equivalence `max_abs_error <= 1e-6`
- state 0/1의 active mask, effective opacity, 또는 DC가 다를 때 timestamp별 render 차이
- inactive state gradient zero
- inactive Gaussian zero effective opacity
- timestamp가 view에 없고 explicit 값도 없으면 오류
- CUDA/scene fixture가 없으면 skip reason 명시

---

### 6.15 `training_setup_temporal_change()`

**권장 새 파일:** `temporal/fusion.py`

```python
def training_setup_temporal_change(
    temporal_model: TemporalChangeModel,
    args,
) -> torch.optim.Optimizer:
    ...
```

#### optimizer 계약

등록 parameter group은 하나뿐이다.

```text
name   = temporal_change_dc
params = [temporal_model.state_change_dc]
lr     = args.feature_lr
```

#### 필요한 이유

기존 `training_setup_change()`는 geometry와 opacity까지 업데이트하므로 Stage 1 identity 가정을 깨뜨린다.

#### 테스트

- optimizer parameter object가 `state_change_dc` 하나뿐
- base parameter가 optimizer state에 없음
- transition 전후 optimizer parameter identity 유지
- step 후 active state DC만 변경 가능

---

### 6.16 `temporal_fusion_step()`

**권장 위치:** `temporal/fusion.py`

```python
def temporal_fusion_step(
    view,
    temporal_model: TemporalChangeModel,
    optimizer: torch.optim.Optimizer,
    pipe,
    background: torch.Tensor,
    render_fn=render_change_temporal,
) -> dict[str, float]:
    ...
```

#### 처리 순서

1. `view.timestamp`의 active state render
2. `compute_ssf_loss(view.candidate_map, render_pkg["render"])`
3. `optimizer.zero_grad(set_to_none=True)`
4. `loss.backward()`
5. `optimizer.step()`
6. scalar log dictionary 반환

#### 주의

- base model learning-rate scheduler를 호출하지 않는다. 현재 scheduler는 xyz group만 대상으로 한다.
- view timestamp를 current global timestamp로 대체하지 않는다.
- densification statistics를 축적하지 않는다.
- `render_fn` injection을 허용해 CPU mock renderer test가 가능해야 한다.

#### 필요한 이유

timestamp selection과 optimization 순서를 한 함수로 고정하여 online loop와 refine/replay의 의미가 달라지지 않게 한다.

#### 테스트

- base parameter 불변
- state B frame step에서 state A slot 불변
- 과거 frame replay가 과거 state만 update
- mock CPU renderer에서 loss 감소
- 실제 CUDA renderer smoke test
- 반환 scalar에 NaN/Inf 없음

---

### 6.17 `sample_temporal_replay()`

**권장 위치:** `temporal/fusion.py`

```python
def sample_temporal_replay(
    replay_buffer: list,
    current_view,
    recent_probability: float = 0.33,
):
    ...
```

#### 책임

기존 sampling branch를 유지하되 반환 view의 timestamp를 그대로 보존한다.

#### 규칙

```text
random <= recent_probability -> current_view
otherwise                    -> replay_buffer 전체에서 균일 표본
```

전체 buffer 표본에는 current view도 포함된다.

#### 테스트

- empty buffer 오류
- current view가 buffer에 포함되어 있다는 precondition 검증 또는 문서화
- 선택한 object identity와 timestamp 불변
- seeded RNG에서 기존 branch와 동일한 선택 sequence
- replay buffer shuffle 후에도 view-timestamp association 유지

---

### 6.18 `load_oracle_changepoints()`

**새 파일:** `temporal/oracle.py`

```python
def load_oracle_changepoints(path: str) -> list[int]:
    ...
```

#### JSON 계약

```json
{
  "time_unit": "frame",
  "boundaries": [5, 10]
}
```

#### 검증

- file 존재
- JSON object 형식
- `time_unit == "frame"`
- boundary는 bool이 아닌 non-negative int
- strict ascending
- duplicate 없음

#### 필요한 이유

Oracle은 GT mask나 event posterior가 아니라 manual boundary timestamp만 제공해야 한다. 파일 계약을 고정하면 temporal representation 검증과 BOCD/event detection 성능이 섞이지 않는다.

#### 테스트

- 정상 parse
- unsorted, duplicate, negative, float boundary 오류
- unsupported time unit 오류
- missing key 오류

---

### 6.19 `validate_oracle_capacity()`

**위치:** `temporal/oracle.py`

```python
def validate_oracle_capacity(
    boundaries: list[int],
    max_states: int,
) -> None:
    ...
```

#### 규칙

oracle global mode에서는 manual boundary가 만들 수 있는 최대 global segment 수를 수용해야 한다.

```text
len(boundaries) + 1 <= max_states
```

#### 필요한 이유

동일 DC 값의 segment도 collapse하지 않으므로 실행 중간에 slot exhaustion으로 partial output이 생성되는 것을 시작 전에 막는다.

#### 테스트

- exact capacity 허용
- capacity 초과 오류
- `max_states < 1` 오류

---

### 6.20 `get_oracle_segment_id()`

**위치:** `temporal/oracle.py`

```python
def get_oracle_segment_id(
    frame_id: int,
    boundaries: list[int],
) -> int:
    ...
```

#### 규칙

boundary frame은 새 segment에 속한다. 개념적으로 `bisect_right(boundaries, frame_id)`와 같다.

```text
boundaries [5, 10]
frame 4  -> segment 0
frame 5  -> segment 1
frame 9  -> segment 1
frame 10 -> segment 2
```

#### 필요한 이유

checkpoint manifest, replay 검증, output 분석이 같은 segment convention을 사용하게 한다.

#### 테스트

- boundary 직전/정확히 boundary/직후
- boundary 없는 sequence
- invalid frame id 오류

---

### 6.21 `apply_oracle_global_transition()`

**권장 위치:** `temporal/oracle.py`

```python
def apply_oracle_global_transition(
    temporal_model: TemporalChangeModel,
    frame_id: int,
    timestamp: float,
    boundaries: list[int],
    init_mode: str,
    local_lifespan_update: dict | None = None,
) -> bool:
    ...
```

#### 책임

- 현재 frame이 boundary가 아니면 아무것도 변경하지 않고 `False` 반환
- boundary이면 manual global segment를 consume하고 `local_lifespan_update` 또는 local lifespan policy를 적용한 뒤 `True` 반환
- active -> active Gaussian은 `transition_state()`로 split한다
- active -> `-1` Gaussian은 현재 state를 boundary에서 close하고 새 state를 열지 않는다
- `-1` -> active Gaussian은 새 local slot을 boundary에서 open한다
- 반드시 replay buffer append와 fusion보다 먼저 호출

#### 필요한 이유

main loop에 boundary membership, mask 생성, transition 순서가 흩어지지 않게 한다. 이 함수는 manual boundary consumer이며 BOCD/event detector가 아니다.

#### 테스트

- non-boundary no-op
- boundary에서 선택된 Gaussian만 transition
- 선택되지 않거나 inactive인 Gaussian은 query에서 `-1` 유지 가능
- active -> `-1` close-only와 `-1` -> active open-only
- transition 후 boundary timestamp query가 새 state 선택
- 같은 boundary 중복 적용 오류

---

### 6.22 `validate_temporal_args()`

**권장 위치:** `temporal/config.py` 또는 `arguments/config_args.py` 근처

```python
def validate_temporal_args(args) -> None:
    ...
```

#### CLI 계약

```text
--temporal_mode {off,oracle_global}
--oracle_changepoints PATH
--temporal_max_states INT
--temporal_init_mode {copy_active,zeros}
--max_frames INT
```

#### mode별 검증

`off`:

- oracle path가 없어도 됨
- temporal model을 만들지 않음
- 기존 optimizer와 densification 유지

`oracle_global`:

- oracle path 필수
- max states 양수
- boundaries capacity 검증
- manual boundaries만 consume하며 BOCD/event detector는 생성하지 않음
- temporal optimizer만 사용
- densification 금지

#### 필요한 이유

잘못된 조합을 실행 중간의 silent fallback으로 처리하면 baseline과 temporal 결과의 의미가 불명확해진다.

#### 테스트

- mode별 valid/invalid 조합
- 기존 CLI 인자만으로 off mode parse 가능
- unsupported mode argparse 오류

---

### 6.23 `save_temporal_checkpoint()`

**권장 새 파일:** `temporal/checkpoint.py`

```python
def save_temporal_checkpoint(
    path: str,
    temporal_model: TemporalChangeModel,
    metadata: dict,
) -> None:
    ...
```

#### 저장 항목

```text
temporal model state_dict
max_states
interval convention: [start,end)
temporal mode
oracle boundaries
frame_id/timestamp/segment_id 목록
base Gaussian count
base PLY path
가능하면 base PLY checksum
trainable parameter 목록
densification_enabled=false
```

#### 필요한 이유

PLY 하나는 여러 temporal slot과 lifespan metadata를 표현할 수 없다.

#### 테스트

- parent directory 생성 정책
- invariant validation 후 저장
- metadata 직렬화 가능성 검증
- partial write를 피하기 위한 temp file + atomic replace 권장

---

### 6.24 `load_temporal_checkpoint()`

**권장 위치:** `temporal/checkpoint.py`

```python
def load_temporal_checkpoint(
    path: str,
    base_change_gaussians: GaussianModel,
) -> tuple[TemporalChangeModel, dict]:
    ...
```

#### 검증

- checkpoint schema/version
- base Gaussian count
- max state shape
- interval convention
- base PLY path/checksum이 있으면 일치 여부
- load 후 invariant

#### 필요한 이유

잘못된 base scaffold에 temporal state를 붙이면 같은 index가 다른 Gaussian을 가리키게 된다.

#### 테스트

- save/load 모든 tensor 동일
- timestamp별 active state index 동일
- mock render 동일
- base count mismatch 오류
- corrupted metadata 오류

---

### 6.25 `build_temporal_manifest()`

**권장 위치:** `temporal/checkpoint.py`

```python
def build_temporal_manifest(
    *,
    args,
    boundaries: list[int],
    frames: list[dict],
    base_gaussian_count: int,
) -> dict:
    ...
```

#### 최소 manifest

```json
{
  "temporal_mode": "oracle_global",
  "interval_convention": "[start,end)",
  "boundaries": [5, 10],
  "max_states": 4,
  "densification_enabled": false,
  "trainable_parameters": ["state_change_dc"],
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "segment_id": 0
    }
  ]
}
```

#### 필요한 이유

실험 결과만 보고 어떤 interval, boundary, trainable parameter를 사용했는지 재구성할 수 있게 한다.

#### 테스트

- required key 존재
- frame timestamp monotonic
- segment id가 oracle 함수 결과와 일치
- JSON round trip

---

## 7. `oscd.py` 통합 책임

Stage 1에서 `oscd.py`는 새 알고리즘을 직접 구현하는 곳이 아니라, 위 함수들을 올바른 순서로 연결하는 orchestration layer여야 한다.

### 7.1 시작 시 mode branch

```python
if args.temporal_mode == "off":
    # 기존 gaussians_change.training_setup_change(opt)
    # 기존 경로 유지
else:
    # oracle load/capacity validation
    # TemporalChangeModel.from_gaussians(...)
    # freeze_base_parameters()
    # training_setup_temporal_change(...)
```

### 7.2 oracle frame branch의 순서

```python
image, info = dataset.getnext()
frame_id = info["frame_id"]
timestamp = resolve_frame_timestamp(frame_id, info)
validate_monotonic_timestamp(timestamp, previous_timestamp)

view = build_camera(...)
view.frame_id = frame_id
view.timestamp = timestamp
view.segment_id = get_oracle_segment_id(frame_id, boundaries)

candidate_map = generate_candidate_map(...)
view.candidate_map = candidate_map.detach().clone()

local_lifespan_update = resolve_stage1_lifespan_update(...)

apply_oracle_global_transition(
    temporal_model,
    frame_id,
    timestamp,
    boundaries,
    args.temporal_init_mode,
    local_lifespan_update=local_lifespan_update,
)

replay_buffer.append(view)

for _ in range(16):
    replay_view = sample_temporal_replay(replay_buffer, view)
    temporal_fusion_step(
        replay_view,
        temporal_model,
        temporal_optimizer,
        pipe,
        background,
    )

current_render = render_change_temporal(
    view,
    temporal_model,
    pipe,
    background,
)
```

`resolve_stage1_lifespan_update(...)`는 detector가 아니다. Stage 1에서는 synthetic/oracle setup이 제공하는 local open/close/split mask 또는 all-true split smoke mask를 사용한다.

### 7.3 금지되는 호출

oracle temporal branch에서는 다음을 호출하지 않는다.

```text
gaussians_change.training_setup_change()
gaussians_change.update_learning_rate()
gaussians_change.add_densification_stats()
gaussians_change.densify_and_clone()
gaussians_change.densify_and_split()
gaussians_change.densify_and_prune()
gaussians_change.reset_opacity()
```

### 7.4 off mode regression

off branch는 기존 순서를 유지한다.

```text
candidate 생성
-> viewpoints append
-> 기존 16회 fusion
-> 기존 densification
-> 기존 mask output
```

temporal helper가 off path에 끼어들지 않게 한다.

---

## 8. 파일별 구현 단위

| 순서 | 파일 | 구현 내용 | 선행 조건 |
|---:|---|---|---|
| 1 | `utils/loss_utils.py` | `compute_ssf_loss()` | 없음 |
| 2 | `utils/time_utils.py` | timestamp resolve/validation | 없음 |
| 3 | `dataloaders/image_dataset.py` | frame id/timestamp metadata | 2 |
| 4 | `dataloaders/stream_dataset.py` | capture timestamp/source id | 2 |
| 5 | `scene/cameras.py` | optional temporal metadata | 2 |
| 6 | `scene/temporal_change_model.py` | slot model, query, transition, invariant | 없음 |
| 7 | `gaussian_renderer/__init__.py` | `override_dc`, inactive opacity mask, temporal wrapper | 6 |
| 8 | `temporal/fusion.py` | optimizer, fusion, replay sampling | 1, 6, 7 |
| 9 | `temporal/oracle.py` | parser, capacity, segment, transition | 6 |
| 10 | `arguments/config_args.py` | temporal CLI | 9 |
| 11 | `temporal/checkpoint.py` | save/load/manifest | 6, 9 |
| 12 | `oscd.py` | mode branch와 전체 orchestration | 1–11 |
| 13 | `tests/temporal/` | unit, synthetic, CUDA, integration tests | 각 단계와 병행 |

`update.py`와 CUDA rasterizer submodule은 수정 대상이 아니다.

---

## 9. 테스트 구조

```text
tests/
  temporal/
    test_ssf_loss.py
    test_timestamps.py
    test_temporal_state_model.py
    test_temporal_transition.py
    test_temporal_renderer.py
    test_temporal_fusion.py
    test_oracle_parser.py
    test_oracle_conflict_separation.py
    test_temporal_checkpoint.py
```

pytest marker:

```text
cuda: requires CUDA rasterizer and GPU
integration: requires PASLCD data
```

기본 CPU test는 실제 scene이나 CUDA extension 없이 통과해야 한다.

---

## 10. 테스트 레이어별 목적

### 10.1 Layer A — Pure logic

검증 대상:

- SSF loss equivalence
- timestamp resolution과 monotonicity
- interval boundary
- subset transition
- invariant failure
- oracle parsing과 segment id
- checkpoint metadata

실행:

```bash
pytest -q tests/temporal -m "not cuda and not integration"
```

### 10.2 Layer B — Gradient isolation

mock renderer 또는 단순 tensor gather를 사용한다.

검증 대상:

- timestamp A backward -> A slot gradient만 non-zero
- timestamp B backward -> B slot gradient만 non-zero
- base parameters unchanged
- transition 전후 optimizer parameter identity 유지
- B frame 추가 학습 후 A slots unchanged

### 10.3 Layer C — Synthetic conflict separation

sequence:

```text
frames 0–4   target A: low change
frames 5–9   target B: high change
frames 10–14 target A: low change
```

비교:

```text
single-state baseline
vs.
oracle global segments [0,5), [5,10), [10,inf)
```

필수 assertion:

1. temporal total loss < single-state total loss
2. segment 0/1/2가 별도 local lifespan slot으로 유지됨
3. identical DC value across segments가 허용되며 slot collapse가 없음
4. frame 2 -> state 0
5. frame 7 -> state 1
6. frame 12 -> state 2
7. B-only extra training 후 두 A slot 불변
8. replay buffer shuffle 후 timestamp-state assignment 동일
9. interval이 정확히 `[0,5)`, `[5,10)`, `[10,inf)`
10. inactive Gaussian은 tensor에 남지만 render contribution은 zero effective opacity
11. opacity-driven smoke case에서도 active mask와 effective opacity가 separation을 지지

loss, active mask, effective opacity, synthetic scale, optimizer 설정을 고정하고 보고서에 기록한다. DC margin을 필수 성공 조건으로 두지 않는다.

### 10.4 Layer D — CUDA renderer

검증 대상:

- slot 0 renderer equivalence
- timestamp switch render difference
- active state gradient isolation
- inactive Gaussian zero effective opacity
- invalid override contract

실행:

```bash
pytest -q tests/temporal -m cuda
```

GPU 또는 extension이 없으면 실패로 위장하지 않고 SKIP 이유를 출력한다.

### 10.5 Layer E — Dataset integration

PASLCD scene을 자동 탐색하여 가능하면 첫 3 frame baseline과 첫 6 frame temporal smoke test를 수행한다.

검증 대상:

- off mode 기존 CLI 실행
- oracle boundary frame 전에 transition 수행
- temporal mode densification 미호출
- output manifest 생성
- checkpoint round trip

데이터가 없으면 integration test는 SKIP 처리한다.

---

## 11. Baseline regression gate

반드시 확인한다.

1. `render_change(override_dc=None, override_opacity=None)` 결과가 수정 전과 같다.
2. online SSF loss와 gradient가 기존 인라인 식과 같다.
3. refine을 치환한다면 refine offset도 기존과 같다.
4. `temporal_mode=off`에서 temporal model을 생성하지 않는다.
5. off mode에서 기존 optimizer와 densification 경로가 유지된다.
6. `update.py`의 import와 실행 경로에 영향이 없다.
7. 기존 CLI 인자만으로 parsing과 실행이 가능하다.
8. 기존 output directory 구조를 유지한다.

수정 전후 동일 3-frame baseline을 실행할 수 있으면 다음을 보고한다.

```text
raw rendered mask max absolute difference
thresholded mask pixel disagreement count
loss trace difference
```

목표:

```text
raw max difference <= 1e-6
thresholded disagreement = 0
```

CUDA 비결정성이 확인되면 tolerance를 임의로 넓히지 않고 원인과 실행 조건을 기록한다.

---

## 12. Stage 1 완료 조건

다음 조건을 모두 만족해야 Stage 2로 넘어간다.

- [ ] 모든 CPU unit test 통과
- [ ] interval invariant 통과
- [ ] transition atomicity 통과
- [ ] inactive state gradient zero
- [ ] inactive Gaussian zero effective opacity
- [ ] base Gaussian parameter 불변
- [ ] optimizer parameter identity 유지
- [ ] synthetic A -> B -> A에서 temporal loss 우세
- [ ] 과거 replay가 해당 timestamp state만 업데이트
- [ ] 동일 DC 값을 가진 segment도 lifespan slot collapse 없음
- [ ] checkpoint round trip 통과
- [ ] manifest 생성
- [ ] temporal mode에서 densification 미실행
- [ ] off mode regression 통과
- [ ] CUDA 환경이 있으면 renderer equivalence 통과
- [ ] PASLCD 데이터가 있으면 smoke test 통과

하나라도 실패하면 BOCD 구현으로 넘어가지 않는다.

---

## 13. 구현 중 특히 경계해야 할 위험

### 13.1 Parameter를 transition 때 교체하는 문제

`state_change_dc = nn.Parameter(...)`를 transition마다 다시 할당하면 optimizer가 이전 object를 계속 참조한다. slot 값만 `torch.no_grad()`에서 수정해야 한다.

### 13.2 transition의 partial mutation

일부 Gaussian을 닫은 뒤 다른 Gaussian에서 capacity 오류가 나면 model이 손상된다. 모든 validation을 먼저 끝내고 한 번에 mutation한다.

### 13.3 replay timestamp overwrite

과거 view에 현재 timestamp나 current segment id를 덮어쓰면 temporal separation이 사라진다. view metadata는 immutable하게 취급한다.

### 13.4 hidden densification

online loop뿐 아니라 refine loop에도 densify/prune 경로가 있다. temporal mode의 모든 학습 경로에서 topology 변경 호출이 없어야 한다.

### 13.5 renderer 분기에서 uninitialized `dc`

현재 `render_change()`는 일부 color override 경로에서 `dc`가 초기화되지 않을 수 있다. `override_dc` 구현 시 기존 경로까지 명시적으로 초기화하고 equivalence test를 둔다. Temporal inactive Gaussian은 DC를 0으로 두는 것만으로 충분하지 않으며 opacity contribution도 0이어야 한다.

### 13.6 timestamp resolver와 monotonic validation 혼합

resolver는 이전 frame을 모르므로 단독으로 역행을 검출할 수 없다. 별도 monotonic validator 또는 tracker가 필요하다.

### 13.7 Stream source id와 consume id 혼동

queue가 frame을 버리므로 consume count는 source frame id가 아니다. capture thread에서 source id를 부여해야 한다.

### 13.8 off mode에 temporal refactor가 침투하는 문제

Stage 1은 새 feature보다 regression 위험이 크다. mode branch를 초기화 시점부터 분리하고 off path에서 temporal object를 만들지 않는다.

### 13.9 Oracle이 GT mask를 사용하는 문제

Oracle input은 boundary timestamp뿐이다. GT change mask를 DC 학습이나 candidate cue 생성에 사용하면 representation 검증이 아니라 supervised upper bound가 된다.

### 13.10 A -> B -> A에서 state를 재사용하는 문제

마지막 A가 첫 A와 외형상 같거나 learned DC가 거의 같더라도 Stage 1에서는 새 state slot을 연다. 두 interval을 같은 slot로 collapse하면 시간별 lifespan 검증이 불가능하다.

### 13.11 manual oracle boundary와 detector를 혼동하는 문제

Stage 1은 global segment boundary를 파일에서 읽어 consume한다. BOCD, event posterior, learned boundary detection은 Stage 2 이후로 남기며 이 문서의 oracle path에 끼워 넣지 않는다.

---

## 14. Stage 1 이후의 함수 순서

Stage 1 gate가 통과한 뒤에만 다음을 순서대로 추가한다.

### Stage 2 — Scalar pre-fusion evidence

```python
def compute_prefusion_change_evidence(...):
    ...
```

현재 frame을 fusion하기 전에 active state가 요구하는 scalar update를 계산한다. model parameter는 수정하지 않으며 invisible primitive는 no-observation으로 둔다.

### Stage 3 — Scalar BOCD

```python
class GaussianBOCD:
    def update(self, observation, observed_mask) -> BOCDOutput:
        ...
```

appearance/disappearance와 reference 복귀 A -> B -> A를 먼저 검출한다.

### Stage 4 — Tentative lifecycle

```python
class TemporalStateManager:
    def step(...):
        ...
```

```text
stable -> tentative -> committed
                    -> rollback
```

한 frame의 noise가 permanent state가 되는 것을 막는다.

### Stage 5 — Vector evidence

```python
def generate_candidate_evidence(...):
    ...

def project_evidence_to_gaussians(...):
    ...
```

```text
z_i,t = [q_i,t, signed RGB residual, projected signed feature residual]
```

scalar가 놓치는 changed B -> changed C 전이를 처리한다.

### Stage 6 — Spatial confirmation

```python
def spatial_confirm_changepoints(...):
    ...
```

primitive별 posterior는 유지하되 local 3D region 단위로 transition을 확인한다.

---

## 15. Stage 1 구현 보고 형식

실제 구현이 끝나면 다음 순서로 결과를 기록한다.

1. 수정 파일 목록
2. 새 함수와 실제 signature
3. 함수별 책임과 설계 변경점
4. 실행한 test 명령
5. test별 PASS / FAIL / SKIP
6. off mode baseline regression 수치
7. synthetic single-state vs temporal loss
8. state별 learned DC / active mask / effective opacity 통계
9. CUDA renderer equivalence max error
10. checkpoint/manifest 검증 결과
11. 확인된 문제와 남은 위험
12. Stage 2 진행 가능 여부

FAIL과 SKIP을 숨기지 않으며, test를 통과시키기 위해 assertion을 약화하거나 baseline behavior를 변경하지 않는다.
