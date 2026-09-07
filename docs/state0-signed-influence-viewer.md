# State 0 Signed-Influence Gaussian Viewer

## 목적

State 0의 마지막 frame `t=94`까지의 95개 view를 사용해 현재 change mask에
영향을 주는 Gaussian을 offline으로 선별한다. 밝은 change를 직접 만드는
Gaussian뿐 아니라, 뒤쪽의 밝은 change를 가리는 DC 0 Gaussian도 보존한다.

## 판정 방식

기존 `state_valid[:, 0]` render를 baseline으로 사용한다.

- 기존 valid row: opacity 제거 효과를 측정한다.
- 기존 invalid row: candidate opacity를 하나만 추가했을 때의 효과를 측정한다.

모든 invalid Gaussian을 동시에 켜지 않는다. 그렇게 하면 reference scene의 많은
검은 Gaussian이 한꺼번에 occluder가 되는 문제가 생기기 때문이다.

Hard binary threshold는 미분할 수 없으므로 다음 soft approximation을 사용한다.

```text
gray      = mean(rendered_change_rgb)
soft_mask = sigmoid((gray - 0.5) / 0.05)
influence = candidate_opacity * d(sum(soft_mask)) / d(effective_opacity)
```

```text
influence > 0  -> additive GS
influence < 0  -> occluding GS
```

95개 view의 평균 absolute influence가 `1e-4` 이상이면 valid로 저장한다.

## 현재 State 0 결과

```text
전체 Gaussian:    1,283,501
valid:               29,451
additive:            10,451
occluding:           20,147
both roles:           1,147
```

산출물:

```text
outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120/
  signed_influence/
    state0_signed_influence.pt
    state0_signed_influence_summary.json
    state0_signed_influence_preview.png
```

## 재생성

```bash
PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.export_state_signed_influence
```

## Viewer 실행

```bash
conda run --no-capture-output -n oscd python viewer.py \
  --ref_ply data/Instance_1/scene_change1_2_3/reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply \
  --temporal_checkpoint outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120/temporal_rchange_checkpoint.pt \
  --influence_artifact outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120/signed_influence/state0_signed_influence.pt \
  --resolution 4 \
  --skip_inference_images \
  --port 8093
```

Viewer는 `State Influence GS`로 시작한다.

- 주황: additive
- 파랑: occluding
- 자주: 두 역할 모두

GUI의 `Influence Role`에서 전체, additive, occluding, both만 따로 볼 수 있다.

## Fixed-view multiview GIF

State 0의 fixed view `t=0..94` 전체에서 동일한 `valid_mask`를 렌더한 결과:

```text
outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120/
  signed_influence/state0_multiview_valid_render/
    state0_influence_valid_multiview.gif
    state0_influence_valid_contact_sheet.png
    state0_influence_valid_multiview.json
    frames/*.png
```

각 GIF frame은 `RGB / all valid / additive / occluding` 네 panel을 포함한다.

```bash
PYTHONPATH=. conda run --no-capture-output -n oscd python \
  -m experiments.render_state_signed_influence_multiview
```

## Gaussian 클릭 검사

`State Influence GS` 화면에서 색칠된 splat을 클릭하면 `Gaussian Inspector`에
다음 값이 표시된다.

- Gaussian index와 additive/occluding role
- State 0 signed-influence 통계와 기여 view 수
- 모든 temporal slot의 `state_valid`, `[state_start, state_end)`, change DC
- frozen base의 xyz, opacity, scale, rotation, base DC
- 클릭 ray와 선택된 Gaussian 사이의 진단 값

클릭 선택은 point center가 아니라, 각 GS를
`Pick support (sigma) × scale` 크기의 회전 타원체로 근사한 뒤 ray가 처음 만나는
현재 role-filter 대상 GS를 고른다. 기본값은 `3σ`이다. 클릭이 잘 잡히지 않으면
`Pick support (sigma)`를 높일 수 있다. 선택된 GS의 타원체는 노란색 계열
wireframe으로 표시되며 `Clear selection`으로 제거할 수 있다.

## 제한

이 값은 Gaussian을 실제로 하나씩 제거해 hard binary mask를 다시 만드는 exact
leave-one-out 결과가 아니다. FastGS opacity backward와 soft threshold를 사용한
효율적인 first-order approximation이다. 현재 baseline validity와 manual State 0
boundary에도 의존한다.
