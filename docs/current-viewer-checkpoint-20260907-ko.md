# Evolving O-SCD current viewer checkpoint — 2026-09-07

사용자가 승인한 이 지점의 코드·회귀 테스트·연구 기록을 하나의 checkpoint로
보존하고 `main`에 반영한다. 기존 `develop`의 이력을 재작성하거나 다른 실험
worktree의 미커밋 변경을 가져오지 않는다.

## 이 checkpoint에 포함된 현재 경로

1. Learned-sigmoid soft Q의 NEW / 그 외 영역 분할과 base/DA3 seed DC gradient
   인터페이스 수정. NEW loss는 seed, 나머지 loss는 base를 학습한다.
2. DA3 seed를 먼저 생성한 뒤 신규 seed를 포함한 base+seed joint raw-cue detector를
   프레임당 한 번 실행한다. Learned DC나 replay를 detector evidence로 재사용하지 않는다.
3. 최초 OPEN BF와 이후 전이 BF를 분리하고, Gaussian별 evidence를 모은 **다음**
   이진화하는 비교 옵션. 학습과 seed target은 여전히 soft Q다.
4. DA3 root birth의 3D 반경 + 현재 뷰 2σ projected occupancy 검사.
   NEVER_OPEN도 공간을 예약하고, CLOSED/retired/future/base는 2D 예약에서 제외한다.
   같은 프레임에 선택한 root도 즉시 자리를 예약한다.
5. Timestamp별 hybrid RLE/2-bit lifespan snapshot과 GPU mask cache.
   시간축 run length가 아니며, 기존 interval 조회와의 일치 검증을 유지한다.
6. NEW/base를 RGB 채널에 나눠 scene 하나·render 한 번으로 학습하는 joint 옵션.
   공유 occlusion은 seed gradient 경로를 바꾸므로 split과 출력의 bitwise 동일성을
   주장하지 않는다.
7. 현재 10-panel viewer와 저장 결과 viewer, panel 재배치 및 lifecycle timeline
   시각화 도구, PASLCD 평가/입력 준비 도구와 관련 ablation 기록.

이전에 작성했지만 아직 커밋되지 않았던 DA3/XFeat/geometry/cue 학습 비교 코드와
문서도 포함한다. 이들을 전부 기본 방법으로 채택한다는 뜻은 아니다.

## 검증된 통합 옵션

기존 데이터/카메라/cue/DA3/SAM 인자에 다음 옵션을 사용한 조건을 검증했다.

```bash
--training-partition panel10_new \
--da3-birth-coverage 3d_plus_2d \
--da3-coverage-2d-sigma 2 \
--da3-max-rows 0 \
--first-open-bayes-factor-threshold 10 \
--bayes-factor-threshold 30 \
--detector-pixel-cue unchanged \
--detector-gaussian-cue binary_q05 \
--lifespan-state-cache snapshot \
--panel10-render-mode joint_channels
```

이것은 검증된 **명시적 preset**이다. Cache off / split / 3D-only 등 기존 CLI
기본값을 이 checkpoint를 만들면서 바꾸지 않았다.

## 최신 성능 근거

PASLCD Instance_1의 Cantina/Garden 각 25프레임, 조건당 50프레임, seed 0:

| 조건 | Mean-frame IoU | Mean-frame F1 | 학습 + seal | 프로세스 wall 합계 |
| --- | ---: | ---: | ---: | ---: |
| 3D+2D + off/split | 0.467943 | 0.617647 | 76.487초 | 112.375초 |
| 3D+2D + snapshot/joint | 0.469255 | 0.618736 | 36.711초 | 72.235초 |

단회·두 scene에서 평균 정확도 저하 없이 학습 52.0%, 프로세스 wall 35.7% 감소를
관측했다. 전체 PASLCD/ESCD 또는 통계적 동등성 검증은 아니다. 사전 cue/DA3 cache
생성 시간은 제외했다. 저장된 100개 mask 재평가와 100 bank/timestamp × 3 mask
snapshot/interval bitwise 검증, detector/optimizer/causality audit를 통과했다.

- [상세 비교](paslcd-two-scene-3d2d-compact-joint-comparison-20260907-ko.md)
- [Git에 보존한 작은 수치·runner 기록](checkpoints/20260907-paslcd-compact-joint/README.md)
- [통합 검증](compact-lifespan-joint-render-develop-integration-20260907-ko.md)
- [3D+2D seed ablation](escd-da3-3d2d-seed-occupancy-ablation-20260907-ko.md)

## 출판 전 검증 / 제외 범위

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. OMP_NUM_THREADS=4 \
/home/rvl/miniforge3/envs/oscd/bin/python -m pytest -q tests
```

현재 저장소의 테스트 **857개 통과**. 실험 산출물 아래에 복사된 과거 테스트가
있으므로 수집 범위를 `tests/`로 지정한다. 범위 없는 `pytest -q`는 해당 복사본과
동일 이름 테스트를 함께 수집해 import mismatch가 발생했다. 코드 검증 결과와
분리해 기록하며 산출물을 삭제하거나 테스트 이름을 바꾸지는 않았다.

변경 Python AST/JSON 파싱, shell syntax, diff whitespace, 파일 크기와 알려진
credential 형식 검사를 수행한다. Ruff/mypy는 환경에 없어 실행하지 않았다.
`outputs/`, 데이터/모델, MP4/PNG, `.omx/`, `.secrets/`, 캐시는 로컬에 그대로 두고
푸시하지 않는다. 따라서 clone만으로 대용량 실험 입력을 복원할 수는 없다.
