# Bayesian detector step viewer

> O-SCD-evolving 전체 목표부터 현재 viewer의 cue, BF30, lifespan, DA3 birth,
> historical replay, `2Q` DC/geometry loss와 output 수식까지 한 문서로 연결한 설명은
> [`oscd-evolving-current-viewer-full-pipeline-20260906-ko.md`](oscd-evolving-current-viewer-full-pipeline-20260906-ko.md)에
> 정리한다.

> 2026-09-04 checkpoint는 timestamp mismatch가 있던 이전 비교 기준이다. 현재
> canonical 계약과 full-run 결과는
> [`bayesian-da3-historical-lifespan-replay-20260906-ko.md`](bayesian-da3-historical-lifespan-replay-20260906-ko.md)에
> 보존한다.

`experiments/view_bayesian_detector_steps.py`는 현재 기본 detector 경로와
`R_change` representation optimization을 한 프레임씩 함께 재생하는 Viser GUI다.
한 번의 `Next cue →`는 현재 raw cue로 detector를 정확히 한 번 갱신한 뒤,
이미 관측한 frame만 사용해 launcher 기준 120회의 representation update를 수행한다.

## 실행

기본 ESCD 304-frame 경로와 BF30 설정을 그대로 사용한다. 실행 스크립트의 현재
viewer preset은 L1 항에만 power를 적용하는 soft product cue다.

```bash
./run_bayesian_detector_viewer.sh
```

NEVER_OPEN geometry를 완전히 고정하고 OPEN geometry에 unclipped `2Q`를 전달하는
viewer variant는 다음과 같이 실행한다.

```bash
./run_bayesian_detector_viewer.sh \
  --no-train-never-open-geometry \
  --geometry-cue-amplitude 2
```

이 variant에서도 base geometry는 `base_geometry_scope=frozen`이므로 고정이다. DA3
OPEN row의 xyz/opacity/scale/rotation만 historical lifespan replay로 학습된다.

현재 preset은 다음과 같다.

```text
P_l1pow = norm(0.8 · L1^0.3 + 0.2 · (1-SSIM))
C = 2 · P_l1pow · S
detector observation = clamp(C / 2, 0, 1) = P_l1pow · S
```

현재 launcher는 이 observation을 바로 쓰지 않고 causal prequential 방식으로
학습되는 frame별 sigmoid를 적용한다.

```text
q = P_l1pow · S
Q_t = sigmoid(logit(0.95) · (q - tau_t) / width_t)
detector / geometry target = Q_t
representation DC target  = 2Q_t
```

첫 frame은 고정 `tau=0.25`, `width=0.10`을 쓴다. Frame `t`에서는 현재 cue
histogram과 frame `t-1`까지 업데이트된 network로 `tau_t,width_t`를 먼저 예측해
`Q_t`를 고정한다. 그 뒤에야 현재 frame의 saved online-at-arrival teacher를 공개해
network를 한 번 업데이트하며, 이 업데이트는 frame `t+1` 이후에만 영향을 준다.
미래 frame이나 현재 frame의 teacher가 현재 `Q_t` 계산에 들어가지 않는다.

```text
predict: theta_(t-1), histogram_t -> tau_t, width_t -> Q_t
update : theta_(t-1), teacher_t   -> theta_t
```

기본 artifact는 다음과 같다.

`outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/learned_boundaries_causal.json`

주의할 점은 saved teacher가 현재 frame의 representation update 후 생성된
online-at-arrival mask라는 것이다. Replay의 시간 순서는 causal하지만 실제 배포에서도
동등한 causal teacher 또는 self-supervised update signal을 얻을 수 있어야 한다.

고정 pose cache에는 `P+S`만 저장되어 있으므로, viewer는 immutable reference RGB를
렌더링하여 원래 O-SCD pixel/SSIM cue `P`를 다시 계산하고 `S=(P+S)-P`로 기존 SAM
cue를 복원한다. 이 cue 복원 자체는 SAM2.1을 다시 실행하지 않는다. 다만 현재
launcher는 별도의 signed feature-difference 관찰 패널을 위해 SAM2.1 image
embedding을 현재 frame에서 한 번 계산한다. 이 signed feature는 DA3 seed proposal
위치를 정할 때만 사용하며 representation loss를 NEW/REMOVE로 나누지 않는다.

브라우저에서 `http://localhost:8090`을 열고 `Next cue →`를 누른다. 포트나
프레임 수를 바꾸려면 다음처럼 실행한다.

```bash
./run_bayesian_detector_viewer.sh --port 8091 --max-frames 20
```

Soft-cue 비교 실행은 cached O-SCD cue의 원래 범위 `0..2`를 `0..1`로
정규화한 fractional observation을 사용한다.

```bash
./run_bayesian_detector_viewer.sh --cue-fusion sum --cue-mode soft --cue-scale 2
```

Binary 모드는 `raw cue > 0.5`, soft 모드는 `clamp(raw cue / 2, 0, 1)`이다.
Soft 모드는 현재 production binary observation을 바꾸지 않는 viewer ablation이다.

메인 캔버스는 1--7번을 첫 행, 8--14번을 둘째 행에 배치한다. 6번은 5번 raw
R_change에 `>=0.5`를 적용한 binary prediction, 7번은 GT union, 8번은
`SAM feature diff × Q`, 9번은 frame별 scale-aligned `GS depth - DA3 depth`다.
각 frame에서 `Q < 0.2`이고 reference alpha가 0.5 이상인 unchanged pixel로 depth
scale을 먼저 맞춘다. 9번은 그 뒤 `|8번 signed SAM × Q| > 0.1`과
`|rendered depth - scaled DA3 depth| > 0.03`의 교집합만 표시한다. 이 교집합의
signed residual을 매 frame 절댓값 q95로 정규화하며, DA3가 GS보다 앞이면 빨강,
뒤이면 파랑, 나머지는 검정이다.
`GT Change`는 ADD와 REMOVE의
흰색 union mask이며 평가·관찰 전용이고 detector evidence에는 절대 들어가지 않는다.

8번 패널은 `SAM(reference render) - SAM(online RGB)`의 signed Delta-PCA score를
64x64에서 최대 절댓값으로 정규화하고 bilinear upsampling한 뒤 learned Q를 곱한다.
양수는 빨강, 음수는 파랑, 0은 검정이다. 별도의 hard Q나 magnitude 상위 40% gate는
적용하지 않는다. Depth overlap에는 절댓값이 0.1보다 큰 support만 쓰고, DA3 처리
해상도로 옮길 때 이 boolean support를 nearest resize한다. Scene별 저장 PC1은 SC1→SC2→SC3 전체
스트림에서 직전 축과 내적이 음수이면 축을 뒤집어 연속 정렬한다.

현재 frame의 alpha-T projection은 detector로 들어가는 `delta_a,delta_b`
pseudo-count 관측이다. 이것 자체는 누적 확률이 아니다. 네 번째 BF 패널은 live
candidate에 대해 `clamp(log BF / log 30, 0, 1)`을 표시한다. Candidate가 없으면
0이다. 즉 fresh-reset 가설이 stable 가설보다 얼마나 우세해져 commit threshold에
접근했는지를 Gaussian별로 보여준다.
순수한 현재-frame alpha-T projection은 첫 패널의 `Main Gaussian layer`를
`Current cue projected to Gaussians`로 선택하면 볼 수 있다.

`--da3-seed-checkpoint`를 주면 두 번째 Online RGB 패널에 causal DA3 seed center를
함께 표시한다. 이전 frame까지 accept된 center는 초록, 현재 frame에서 새로 태어난
center는 흰 테두리의 빨강이다. 표시 여부는 checkpoint의 `frame_global` birth
metadata로만 결정하며 미래 birth를 앞선 frame에 보여주지 않는다. 이 overlay는
seed geometry 관찰용이며 GT를 사용하지 않는다.

현재 launcher는 causal prequential sigmoid cue와 positive depth residual 판정으로
SC1, SC2, SC3의 seed birth를 순서대로 재생한 artifact를 기본으로 사용한다.

`outputs/causal_da3metric_scene123_panel7pos010_depthpos003_dynamiccoverage_20260904/da3_seed_replay.pt`

이 seed replay는 최대 8개의 현재/과거 view만 사용했고 future-view/GT birth access는
모두 0이다. 각 frame에서 learned `Q < 0.2`이고 reference alpha가 0.5 이상인
unchanged 영역으로 DA3Metric-Large depth를 immutable rendered GS camera-z에 positive
scale-only 정렬한다. Proposal은 reference alpha가 0.5 이상이고 `8번 > +0.1`이며 signed residual
`rendered depth - scaled DA3 depth > 0.03`인 pixel만 허용한다. 9번 시각화는
SAM 양·음 support를 모두 보이지만 실제 seed는 SAM 양수 support에서만 생성한다.
별도의 `Q >= 0.5`, top-40%, 고정 2 cm voxel gate는 두지 않는다.

DA3 8-view window는 scene 경계를 가로질러 직전 frame을 그대로 사용한다. Offline
artifact는 동적 중복 판정을 하기 전의 causal proposal bank다. 장면별 proposal 수는
`16,842 / 24,206 / 24,631`, 총 `65,679`개다. Viewer는 각 birth timestamp에 도착했을 때만 proposal을 검사하고
accepted row만 `NEVER_OPEN`으로 materialize한다. 따라서 artifact의 `accepted` 표기는
offline proposal acceptance이며 실제 viewer acceptance와는 구분한다.

각 브라우저 client의 실제 canvas 종횡비에
맞춘 dashboard를 만들고, 각 패널 내부에서는 원본 이미지 종횡비를 유지한 채
letterbox한다. 따라서 창 너비가 바뀌어도 카메라 이미지가 가로로 늘어나지 않는다.
사이드바에는 제어와 수치만 남기므로 cue/evidence 확인을 위해 아래로 스크롤할
필요가 없다.

## 한 번 클릭할 때 수행하는 일

1. 신규 online RGB와 저장된 O-SCD pixel+SAM sum cue를 한 장 읽는다.
2. product preset이면 reference RGB로 원래 `P`를 재계산해 cached sum에서 `S`를
   복원한다. 현재 preset은 L1에만 `0.3` power를 적용해 `2·P_l1pow·S`를 만든다.
3. 현재 frame의 cue histogram과 과거까지만 학습된 boundary network가 미리 산출한
   `tau_t,width_t`로 연속값 `Q_t`를 만든다.
4. Immutable reference geometry/opacity에 alpha-transmittance VJP를 한 번
   수행하여 Gaussian별 positive/negative pseudo-count를 얻는다.
5. BF30 single-candidate Beta filter에 관측을 한 번만 추가한다.
6. Commit된 전이만 OPEN 또는 CLOSE한다.
7. 현재 frame보다 먼저 태어난 DA3 seed만, birth geometry를 고정한 별도 detector
   probe에서 Part19와 동일하게 저장된 원본 O-SCD `P+S > 0.5` binary cue의
   alpha-T evidence를 계산한다. 각 seed는 `start=inf`, `end=inf`, stable
   `Beta(1,10)`인 `NEVER_OPEN`으로 시작하고 같은 BF30 규칙으로
   OPEN/CLOSE/REOPEN한다. Birth frame은 geometry proposal에만 쓰고 detector
   evidence는 다음 신규 frame부터 받는다. Learned sigmoid Q는 이 seed detector에
   들어가지 않으며, 최적화된 seed geometry/DC도 detector 입력에서 제외한다.
8. 현재 timestamp의 DA3 proposal을 공개하고 learned Gaussian coverage로 중복을
   제거한 뒤, accepted row를 `NEVER_OPEN`으로 materialize한다. Birth frame은 seed
   detector evidence로 재사용하지 않는다.
9. 현재까지 관측한 frame bank에 현재 view와 causal soft target을 추가한다.
10. 120회 각각에서 확률 0.33으로 가장 최근 frame을 선택하고, 나머지 0.67에서는
    `[0,t]`의 이미 관측한 frame `k`를 균등 랜덤 선택한다. 균등 branch가 우연히 현재
    frame을 다시 뽑을 수도 있으며 미래 frame은 선택할 수 없다.
11. Sampled frame `k`를 골랐으면 camera=`I_k`, cue=`Q_k`, base
    lifespan=`L_base(k)`, seed lifespan=`L_seed(k)`를 함께 사용한다. `k`에서 OPEN인
    base/seed만 learned DC로 렌더하고, `k`까지 materialize됐지만 아직 OPEN되지 않은
    row는 black occluder로 남긴다. `k`에서 CLOSED이거나 `k` 이후 태어난 seed는 렌더와
    optimizer에서 제외한다. DC target만 원본 O-SCD 범위인 `2Q_k`이고 geometry target은
    unit `Q_k`다. FastGS degree-zero SH에서 DC 0은 RGB 0.5이므로 black에는
    `RGB2SH(0)`을 명시한다.
12. Sampled `k`에서 OPEN인 DA3 row는 DC, xyz, opacity, scale, rotation을 학습한다.
    `k`에서 `NEVER_OPEN`인 row도 별도 pending branch에서 xyz, scale, rotation을 전체
    unsigned `Q_k` coverage로 학습하지만 DC와 opacity는 고정한다. Detector용 birth
    geometry probe는 별도 tensor로 유지하므로 pending geometry 학습이 BF30 evidence로
    역류하지 않는다. NEW/REMOVE sign, 9-point BCE, Part19 seed-only projected-coverage
    DC loss는 사용하지 않는다.
13. 학습 후 current valid base와 DA3 sidecar를 합쳐 raw change prediction을
    렌더하고, 최종 mask threshold는 기존대로 0.5를 유지한다.
14. 마지막으로 evaluation-only `ADD ∪ REMOVE` GT를 표시한다.

Representation replay, post-optimization render, learned DC/geometry는 detector
evidence에 다시 들어가지 않는다. Detector는 항상 해당 frame 도착 직후의 raw cue를
한 번만 소비하므로 representation 학습에서 detector로 역류하는 경로가 없다.

DA3 seed는 미래 geometry를 미리 GPU model에 넣지 않는다. 각 row의
`frame_global == t`일 때만 sidecar와 고정 detector probe에 append한다. Representation
append는 그 frame의 seed detector update 뒤에 수행하므로 proposal을 만든 관측이
동시에 proposal을 검증하는 selection bias가 없다. 새 row의 tracker posterior는
첫 후속 관측 전까지 정확히 stable `Beta(1,10)`으로 유지된다. Representation
sidecar에서는 **sampled timestamp `k` 기준** `NEVER_OPEN`과 OPEN만 렌더 support에
포함하되 NEVER_OPEN은 black occluder이고 `k`에서 CLOSED인 row는 완전히 제외한다.
`k` 이후 태어난 row 역시 존재하지 않는 것으로 처리한다. Reference
xyz/RGB SH/opacity/scale/rotation은 optimizer에 포함하지 않는다. Base optimizer는
`k`에서 OPEN이고 sampled view에 보이는 row의 DC만 갱신한다. DA3 optimizer는 `k`에서
visible OPEN인 row의 모든 attribute와 `--train-never-open-geometry`가 선택한 `k`의
visible NEVER_OPEN row의 xyz/scale/rotation만 갱신한다.
Off-view row의 parameter와 Adam moment는 보존한다.

현재 viewer의 learnable DC/geometry는 Gaussian row마다 하나씩 유지하고 모든 causal
replay view가 그 parameter를 공동 최적화한다. 이것이 의도한 multiview optimization이다.
Lifespan은 parameter를 복제하는 축이 아니라 sampled timestamp에서 해당 row가
render/optimizer에 참여하는지를 정하는 gate다. 따라서 과거 parameter snapshot을
되감거나 CLOSE→REOPEN마다 parameter를 새로 만들지 않는다.

고정 2 cm voxel occupancy 대신 학습된 Gaussian coverage를 사용한다. 이미 accepted된
seed가 실제 geometry optimizer update를 최소 4회 받은 뒤부터만 새 proposal을 막을
수 있다. 판정 중심은 birth xyz가 아니라 현재 학습된 xyz이며, 반경은 현재 학습된
세 축 scale 중 최댓값의 2배다. `NEVER_OPEN`과 OPEN row는 coverage에 참여하지만
`CLOSED` row는 제외한다. 같은 frame의 아직 학습되지 않은 proposal끼리는 서로를
막지 않는다. Viewer sidebar에는 frame별 proposal/accepted/coverage-rejected 수를
따로 표시한다.

DA3 geometry의 단안 depth 오차가 온라인 update에서 다시 폭주하지 않도록 birth
scale 대비 xyz displacement 4배, scale `0.25..4배`, opacity `0.01..0.99`의
trust region을 사용한다. 이는 geometry를 고정하는 것이 아니라 제한된 범위 안에서
실제로 최적화하는 안전장치다.

## 색 의미

- 검정: reference-consistent `NEVER_OPEN`
- 초록: committed `OPEN`
- 빨강: 과거에 OPEN된 뒤 committed `CLOSED`

Lifecycle layer는 committed 상태만 표시한다. Live candidate는 lifecycle 색을
덮어쓰지 않는다. 따라서 OPEN의 candidate는 계속 초록이고 NEVER_OPEN/CLOSED의
candidate도 각각 검정/빨강으로 남는다. Candidate 여부와 BF 진행도는 네 번째
detector-score panel에서 확인한다.

1번 lifecycle render의 모집단은 base Gaussian과 현재까지 causally born된 DA3
sidecar 전체다. 따라서 5번 learned R_change에 들어가는 DA3 seed도 1번에서
`NEVER_OPEN=검정`, `OPEN=초록`, `CLOSED=빨강`으로 확인할 수 있다.

3번 cue와 4번 BF는 동일한 heatmap을 사용한다.

```text
0(검정) → 초록 → 노랑 → 빨강(1)
```

각 anchor 사이는 RGB 값을 선형 보간한다. 따라서 0은 정확히 검정이고, 값이
증가할수록 끊김 없이 초록, 노랑, 빨강 순서로 변한다. 큰 cue 또는 commit
threshold에 가까운 reset BF일수록 빨강에 가까워진다.

`Main Gaussian layer`에서 다음 네 레이어를 바꿔 볼 수 있다.

- committed lifecycle black/green/red
- 현재 2D cue가 reference Gaussian에 투영된 positive evidence
- 누적 stable Beta flip probability와 live candidate BF 진행도
- 현재 OPEN base DC와 active DA3 DC/geometry를 합친 learned raw R_change prediction

`Reset to reference initialization`은 cue를 하나도 소비하지 않은 검정 초기
상태로 되돌리고 base DC, DA3 sidecar, 두 optimizer의 Adam state, causal replay
bank도 함께 초기화한다. `--capture-dir PATH`를 주면 클릭 시 main/detector/learned
render를 PNG로 저장한다.
