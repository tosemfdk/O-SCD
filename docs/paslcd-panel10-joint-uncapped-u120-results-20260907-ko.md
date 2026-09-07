# PASLCD: 현재 Panel10 + 공동 BF30 + 무제한 DA3 seed 평가

2026-09-07. 사용자 지정 데이터 `/home/rvl/workspace/github/O-SCD/data/PASLCD`의
**20 scenes ×25 frames =500 frames**를 전부 평가했다. 각 scene마다 새 reference,
lifespan, optimizer, PCA, sigmoid calibration으로 시작한다. Scene 사이 상태는 공유하지 않는다.

## 핵심 결과

| 조건 | Updates/frame | 평균 foreground IoU | 평균 F1 |
|---|---:|---:|---:|
| **현재 Panel10 + joint BF30 + DA3, archive 무제한** | **120** | **0.4288** | **0.5576** |
| O-SCD online 저장 mask, 동일 GT 재평가 | 16 | 0.4906 | 0.6440 |
| O-SCD refined 저장 mask, 동일 GT 재평가 | 추가 정제 | 0.5584 | 0.7018 |

**현재 방법은 O-SCD online보다 mIoU −6.18%p, F1 −8.64%p로 낮다.**
Scene별 mIoU 승리는 6/20이다. ESCD 연속 stream에서의 개선이 PASLCD 전체로
일반화되었다고 결론내릴 수 없다.

현재 전체 pixel 통합 IoU/F1은 **0.4518/0.6224**, precision/recall은
**0.8367/0.4954**다. O-SCD online 통합 precision/recall은 **0.6680/0.6667**로,
현재 결과는 FP가 적지만 변화 누락이 크다. TP/FP/FN은 각각
**4,855,663 /947,352 /4,945,100**이다.

기존 baseline CSV의 online **0.4887/0.6423**, refined **0.5573/0.7009**와
이번 수치가 조금 다르다. 본 표는 현재 mask와 동일한 binary GT loader/native 해상도로
원본 저장 mask를 전부 다시 평가한 값을 사용한다. 기존 CSV와 정확히 같다고 가정하지 않는다.

## 초반 도착 프레임별 진단

각 scene 안에서 같은 도착 위치를 모아 평균했다. 각 행100 frames다.

| Scene 내 frame | 현재 mIoU | O-SCD online mIoU |
|---|---:|---:|
| 1–5 | 0.1497 | 0.4737 |
| 6–10 | 0.4170 | 0.4935 |
| 11–15 | 0.5131 | 0.4812 |
| 16–20 | 0.5265 | 0.5311 |
| 21–25 | 0.5378 | 0.4736 |

초반5 frames의 mIoU **0.1497**과 마지막5 frames의 **0.5378**이 크게 다르다.
초기 변화 누락이 전체 평균 하락의 중요한 관측 패턴이다. 다만 이것만으로 BF30 OPEN
지연, seed confirmation, DC 학습, cue calibration 중 하나를 원인으로 확정할 수는 없다.
위치를 제외한 더 좋은 subset을 headline 성능으로 사용하지 않는다.

## Scene별 결과

| Scene | 현재 mIoU | 현재 F1 | O-SCD online mIoU | IoU 차이 | 최종 seed archive |
|---|---:|---:|---:|---:|---:|
| Instance_1 / Cantina | 0.2881 | 0.4112 | 0.4954 | -0.2074 | 12,095 |
| Instance_1 / Garden | 0.5040 | 0.6505 | 0.4912 | +0.0129 | 5,892 |
| Instance_1 / Lounge | 0.5775 | 0.6966 | 0.5463 | +0.0311 | 3,402 |
| Instance_1 / Lunch_room | 0.1848 | 0.2846 | 0.3901 | -0.2053 | 4,573 |
| Instance_1 / Meeting_room | 0.4938 | 0.6351 | 0.4964 | -0.0026 | 5,436 |
| Instance_1 / Playground | 0.4216 | 0.5622 | 0.3865 | +0.0351 | 9,824 |
| Instance_1 / Porch | 0.4760 | 0.6094 | 0.5721 | -0.0961 | 4,950 |
| Instance_1 / Pots | 0.4310 | 0.5660 | 0.6088 | -0.1778 | 3,140 |
| Instance_1 / Printing_area | 0.4927 | 0.6007 | 0.6104 | -0.1177 | 11,434 |
| Instance_1 / Zen | 0.4370 | 0.5796 | 0.4736 | -0.0366 | 1,929 |
| Instance_2 / Cantina | 0.3079 | 0.4284 | 0.4907 | -0.1828 | 11,351 |
| Instance_2 / Garden | 0.4794 | 0.6298 | 0.5079 | -0.0284 | 3,703 |
| Instance_2 / Lounge | 0.4327 | 0.5811 | 0.4790 | -0.0463 | 7,151 |
| Instance_2 / Lunch_room | 0.1779 | 0.2736 | 0.3817 | -0.2038 | 4,758 |
| Instance_2 / Meeting_room | 0.4376 | 0.5860 | 0.4582 | -0.0206 | 7,337 |
| Instance_2 / Playground | 0.4191 | 0.5556 | 0.3636 | +0.0554 | 10,673 |
| Instance_2 / Porch | 0.4212 | 0.5640 | 0.5318 | -0.1107 | 5,233 |
| Instance_2 / Pots | 0.6076 | 0.7217 | 0.5704 | +0.0372 | 14,248 |
| Instance_2 / Printing_area | 0.4142 | 0.5385 | 0.5087 | -0.0945 | 7,268 |
| Instance_2 / Zen | 0.5724 | 0.6778 | 0.4501 | +0.1223 | 16,215 |

모든 scene의 seed archive가20,000보다 작았다(max **16,215**).
따라서 이 PASLCD 실험에서 기존20k 상한이 실제로 성장을 막은 것은 아니며, cap 해제가
PASLCD 성능을 개선했다는 ablation 증거도 아니다. 총 archive는20개 별도 모델을 합쳐
**150,612**, 최종 OPEN seed 합은 **16,246**다.

## 유지한 방법과 PASLCD 입력 연결

- Core: `training_partition=panel10_new`, `da3_max_rows=0`, seed0,120updates,
  current-view branch0.33, raw learned-sigmoidQ, l1 exponent0.3.
- 현재-view DA3 birth 후 고정 base+seed 공동 alpha-T evidence를1회 계산하고 단일
  BF30 tracker를 갱신한다. Q_NEW는 seed DC/geometry SSF, Q−Q_NEW는 base DC SSF.
- Seed geometry 제약,1,024roots/frame,128children/event, root별32 descendants,
  generation/one-shot 제한, pruning은 그대로다.
- NEVER_OPEN은 frozen black occluder, CLOSED는 제외, final mask는 base+seed
  joint raw render meanRGB≥0.5. GT를 detector/loss/birth에 공급하지 않았다.
- 기존 PASLCD fixed pose와 raw pixel+SAM cue를 재사용하고 hash를 확인했다.
  `Instance_1/Lounge`만 기존 compatible fallback camera를 사용하며 cue cache와 일치한다.
- PASLCD에는 binary `gt_mask/{image_stem}.png`가 있으므로 평가/display용
  `--gt-format binary`를 추가했다. NEW/REMOVE GT label을 임의로 만들지 않았다.
- Signed SAM: scene-local exact prefix CausalPC1을 현재/과거 frame만으로 생성했다.
  기존 canonical 첫-axis + 방향을 그대로 사용했다. +가 모든 PASLCD scene에서
  의미적으로 NEW라는 보장은 없고, GT나 depth로 방향을 다시 정렬하지 않았다.
- DA3METRIC-LARGE/process_res504: 현재 이미지1장씩 depth를 생성하고, 같은 기존
  viewer 코드가 매 frame immutable reference GS depth에 정렬한다. 미래 view 미사용.
- Stage2 boundary: 첫 tau/width0.25/0.10에서 시작, 현재 histogram으로 Q_t를 확정한
  다음 원본 O-SCD online-at-arrival teacher_t가 다음 frame의 boundary만 갱신한다.
  GT/refined mask를 teacher로 쓰지 않는다. 현재 방법처럼 보조 causal O-SCD teacher에
  의존하므로 teacher-free 배포 성능으로 해석하지 않는다.
- 각25-frame sequence는 기존40-frame trunk warm-up 안에 있으므로 boundary head만
  학습한다. ESCD304-frame의 calibration state를 PASLCD에 재사용하지 않았다.

## 검증

-20 scenes/500 frames 정상 종료(exit0), frame 누락 없음.
- 모든500 frame의 base/seed DC 학습과 reference/fixed probe/frozen parameter/
  frozen Adam 보존 audit를 기록했다. 허용하지 않은 drift, future-view access,
  lifespan render violation은 모두0. 공동 evidence는 frame당1회.
- 저장 prediction500장과 원본 baseline 두 종류를 GT와 독립 재계산하여 모든 TP/TN/
  FP/FN, frame score, aggregate가 일치했다. Source/input artifact hash도 일치했다.
- 전체500-frame MP4: **1920×1120,10fps,50초,H.264/yuv420p**. 전체 디코딩 오류0,
  마지막 frame 이미지 비교 통과. 원래 layout의 letterboxing/reserved panel은 유지했다.
-138 targeted CPU+CUDA tests 통과. 실제 DC backward/optimizer 갱신, 공동 evidence,
  seed growth/reset, causal boundary/PCA, binary GT 및 결과 검증 포함.
- 독립 code review: blocking issue0. Prep/eval 기본 입력 경로 불일치1건은 수정하고
  회귀 테스트 후 authoritative full run을 재시작했다.
- AST/py_compile, shell syntax, diff whitespace 검사 통과. 전용 LSP/lint/typechecker는
  설치되어 있지 않았다.
- Scene 실행시간 합 **1109.7초**, 최대 scene peak allocated CUDA
  **1,932,372,992 bytes**. 입력 준비·모델 초기화·audit/렌더·GPU 공유 조건이
  달라 모델-only throughput 비교로 사용하지 않는다.

PASLCD는 scene별로 고정된 post-change 상태를 제공하므로 본 실험은 single-state
mask 품질 평가다. 실제 repeated evolution 대응의 성공/실패를 직접 검증하지 않는다.
단일 seed0이며 현재120updates와 원본online16updates는 계산량이 맞지 않는다.

## 변경 파일 및 자료

- `experiments/view_bayesian_detector_steps.py`: binary union GT와 scene-local SAM trace 입력.
- `experiments/prepare_paslcd_panel10_inputs.py`: causal PASLCD 입력 준비/검증.
- `experiments/evaluate_paslcd_panel10.py`: per-scene 실행, matched metric, invariant audit, MP4.
- 각 대응 regression test; 기존 detector/loss/topology 동작은 변경하지 않았다.

[전체500-frame viewer MP4](../outputs/paslcd_panel10_uncapped_u120_20260907/viewer_dashboard_all_frames.mp4)
· [전체/scene summary](../outputs/paslcd_panel10_uncapped_u120_20260907/summary.json)
· [500-frame CSV](../outputs/paslcd_panel10_uncapped_u120_20260907/frame_metrics.csv)
· [독립 검증](../outputs/paslcd_panel10_uncapped_u120_20260907/posthoc_verification.json)
· [초반/후반 진단](../outputs/paslcd_panel10_uncapped_u120_20260907/analysis_notes.json)
· [재현 방법](../outputs/paslcd_panel10_uncapped_u120_20260907/README.md)
