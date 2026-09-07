# PASLCD 2-scene: 3D+2D seed + compact lifespan + joint render 비교

## 결론

2026-09-07, PASLCD `Instance_1/Cantina`와 `Instance_1/Garden`의 각 25프레임을
두 조건으로 새로 실행했다. **이번 두 scene / seed-0 측정에서는 평균 정확도가
유지되면서 학습시간이 약 절반으로 줄었다.**

- Mean-frame IoU: **0.467943 → 0.469255** (+0.001312, +0.1312 percentage point).
- Mean-frame F1: **0.617647 → 0.618736** (+0.001089).
- 학습 + snapshot seal: **76.487 → 36.711초** (52.00% 감소, 2.083배).
- 프로세스 실행시간 합계: **112.375 → 72.235초** (35.72% 감소, 1.556배).

이는 통계적 동등성이나 전체 PASLCD / ESCD 성능 보존의 증명은 아니다.
Garden F1은 0.000023 낮아졌고, 두 scene 평균은 소폭 높아졌다.
작은 정확도 차이를 유의한 개선으로 해석하지 않는다.

## 비교 조건

| 조건 | DA3 root birth | Lifespan cache | 학습 render |
| --- | --- | --- | --- |
| baseline | 3D+2D, 2σ | off | split, update당 2회 |
| combined | 3D+2D, 2σ | snapshot | joint_channels, update당 1회 |

여기서 baseline은 **현재 구현의 3D+2D + split 조건**이며 원본 O-SCD가 아니다.
이 비교는 compact와 joint의 **결합 효과**를 측정한다. 각 기능의 기여도를
독립적으로 분리한 factorial ablation은 아니다.

공통 설정:

- Dataset: `/home/rvl/workspace/github/O-SCD/data/PASLCD`.
- 사전 선택한 `Instance_1/Cantina`, `Instance_1/Garden`, scene마다 전체 25프레임.
- scene와 조건마다 독립 프로세스에서 상태 초기화. 조건당 50프레임, 총 100프레임 처리.
- 최초 OPEN BF10, CLOSE/REOPEN BF30.
- Detector: Gaussian별 alpha-T evidence를 모은 뒤 `binary_q05`; 픽셀은 이진화하지 않음.
- 학습/생성 target: 기존 learned-sigmoid soft Q. 이미지당 120 updates, seed 0,
  causal sampled replay, current-view probability 0.33.
- DA3 root 총량 제한 없음, 프레임당 최대 1,024개, 기존 3D 검사 + 2σ projected occupancy.
- Seed를 먼저 생성한 뒤 base+seed joint detector를 프레임당 한 번 실행.
- NEVER_OPEN frozen-black 정책, 기존 density/pruning 기준 유지.
- GPU: NVIDIA RTX A6000. 기존 사용자 viewer는 중단하지 않음.

실행 순서는 `Cantina baseline → Cantina combined → Garden combined → Garden baseline`으로
scene별 순서를 반대로 배치했다. 학습 job끼리는 GPU에서 겹치지 않았다.
전체 batch는 결과 재검증/그래프 생성을 포함해 187.305초였다.
이전 ESCD 304-frame 작업은 사용자 범위 변경에 따라 baseline 21프레임에서 중단했고,
그 미완료 결과는 이 표에 섞지 않았다.

## 정확도

아래 mIoU/F1은 **프레임별 foreground IoU/F1의 산술평균**이다.
PASLCD binary GT만 사용했으며 NEW/REMOVE별 GT metric은 계산하지 않았다.

| Scene | 조건 | mIoU | F1 | Root births | Density children |
| --- | --- | ---: | ---: | ---: | ---: |
| Cantina | baseline | 0.439286 | 0.582057 | 2,543 | 55 |
| Cantina | combined | 0.441899 | 0.584258 | 2,564 | 62 |
| Garden | baseline | 0.496601 | 0.653238 | 2,232 | 20 |
| Garden | combined | 0.496611 | 0.653215 | 2,224 | 18 |
| **50-frame 평균/개수 합계** | **baseline** | **0.467943** | **0.617647** | **4,775** | **75** |
| **50-frame 평균/개수 합계** | **combined** | **0.469255** | **0.618736** | **4,788** | **80** |

전체 pixel count를 합친 별도 metric:

| 조건 | Pooled IoU | Pooled F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.457633 | 0.627913 | 0.796023 | 0.518427 |
| combined | 0.459727 | 0.629881 | 0.795459 | 0.521358 |

Joint는 단순 캐시 최적화와 달리 공유 occlusion 때문에 seed geometry/opacity의
gradient 경로가 바뀐다. 이후 geometry에 의존하는 seed 중복 검사, density,
lifecycle도 달라질 수 있으므로 root 수/출력이 bitwise 같아야 하는 실험은 아니다.

## 실행시간

| Scene | 조건 | 학습 + seal (초) | Scene runner (초) | 프로세스 wall (초) |
| --- | --- | ---: | ---: | ---: |
| Cantina | baseline | 35.666 | 52.204 | 54.162 |
| Cantina | combined | 17.845 | 34.232 | 36.150 |
| Garden | baseline | 40.821 | 56.309 | 58.213 |
| Garden | combined | 18.866 | 34.162 | 36.085 |
| **합계** | **baseline** | **76.487** | **108.513** | **112.375** |
| **합계** | **combined** | **36.711** | **68.394** | **72.235** |

- 학습시간은 `_train_representation` 전체 호출의 CUDA-synchronized wall time이다.
  프레임 종료 snapshot sealing을 별도로 측정한 뒤 합산했다.
- 내부 mask/assembly/render별 반복 synchronize는 꺼서 계측의 영향을 줄였다.
  따라서 `timing.json`의 비활성 component timer 0은 실제 비용이 0이라는 뜻이 아니다.
- Scene runner에는 입력 로딩, detector/birth/training, frozen-state audit,
  panel/mask PNG 저장과 저장 mask 재검증이 포함된다.
- 프로세스 wall에는 import/setup과 마지막 snapshot-vs-interval audit까지 포함된다.
  마지막 audit은 combined Cantina 0.029초, Garden 0.036초였다.
- **SAM/DA3/cue cache 생성 시간은 포함하지 않는다.** MP4 encoding도 수행하지 않았다.
- Combined 학습 render 호출은 총 **12,000 → 6,000회**로 절반이었다.
- 기존 viewer 및 desktop 부하가 있어 timing은 단회 관측값이다. GPU telemetry와
  각 프로세스 시작/종료의 GPU process 목록을 보존했다.

## 검증 및 provenance

- 네 프로세스 모두 exit 0. 원본 GT와 **저장된 100개 prediction mask**에서
  TP/TN/FP/FN/IoU/F1을 다시 계산해 frame CSV와 일치함을 확인했다.
- Scene별 두 조건의 source/input hash, 설정(출력 경로/cache/render 옵션 제외),
  매 timestamp의 soft Q/NEW target hash와 sampled replay index가 정확히 같았다.
- 모든 프레임의 joint detector evidence pass는 1회였다.
- Reference/probe/frozen parameter/frozen Adam drift, future-view access,
  lifespan render violation은 모두 0이었다.
- 네 run 모두 실제 CUDA base DC와 seed DC 갱신이 확인됐다.
- Snapshot과 원래 interval 조회의 **100개 bank/timestamp 조합 × 3 mask**가
  bitwise 일치했다. Combined 각 bank에 25개 snapshot, interval fallback 0.
- `develop` HEAD는 `a5ecdb49fb071a890a6abafb5b972d1aedac63a0`.
  미커밋 통합 구현을 포함한 main의 151개 Python 파일과 frozen source hash가
  일치한다. Frozen runtime에는 실험 runner 2개를 더한 153개 파일을 기록했다.
- 이번 작업은 실험 runner/분석 및 보고서만 추가했으며 production 구현,
  기본값, git HEAD/index, 기존 viewer는 변경하지 않았다.
- Runner/분석 script AST 검증과 독립 read-only runner review를 통과했다.
  새 production 변경이 없어 기존 통합의 253-test 검증을 재실행하지는 않았다.
  Ruff/mypy/static-analysis 도구는 이 환경에 없어 실행하지 않았다.

## 산출물 / 재현

`outputs/`가 제외되는 Git checkout에서도 읽을 수 있도록 작은 수치·timing·runner
기록을 [checkpoint evidence](checkpoints/20260907-paslcd-compact-joint/README.md)에 복사했다.

Root: [`outputs/paslcd_2scene_3d2d_compact_joint_20260907/`](../outputs/paslcd_2scene_3d2d_compact_joint_20260907/)

- [`comparison.json`](../outputs/paslcd_2scene_3d2d_compact_joint_20260907/comparison.json): 전체 수치/검증.
- [`comparison.csv`](../outputs/paslcd_2scene_3d2d_compact_joint_20260907/comparison.csv): 요약 표.
- [`comparison_timeline.png`](../outputs/paslcd_2scene_3d2d_compact_joint_20260907/comparison_timeline.png): 프레임별 IoU/학습시간.
- `plan.json`, `runtime_source_sha256.json`, `main_source_match.json`: 비교 계획/소스.
- `<Scene>_<condition>/timing.json`, `process_time.json`: 시간 및 snapshot 통계.
- `<Scene>_<condition>/Instance_1/<Scene>/`: 설정/input hash/CSV/audit/저장 panel·mask.
- `run_batch.py`: 네 조건 순차 실행. 기존 결과 보호를 위해 출력 폴더가 있으면 중단한다.
- `compare_results.py`: 저장 결과 재검증 및 표/그래프 재생성.

완료된 학습을 다시 실행하지 않고 분석만 재현:

```bash
ROOT=outputs/paslcd_2scene_3d2d_compact_joint_20260907
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
PYTHONPATH="$ROOT/runtime_source:." \
/home/rvl/miniforge3/envs/oscd/bin/python "$ROOT/compare_results.py"
```

관련 구현: [develop 통합 기록](compact-lifespan-joint-render-develop-integration-20260907-ko.md).
이 두 scene 결과를 ESCD 연속 304프레임 또는 PASLCD 전체 20 scene 결과로 대체하지 않는다.
