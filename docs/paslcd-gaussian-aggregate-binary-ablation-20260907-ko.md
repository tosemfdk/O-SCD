# PASLCD: Gaussian 합산 후 이진화 수정 실험 (2026-09-07)

## 수정 의도와 결론

이전 실험은 사용자의 의도를 잘못 해석해 **pixel Q를 먼저 이진화**했다.
이번에는 **soft pixel Q를 Gaussian별 alpha-T로 합산한 다음, 그 비율을 이진화**한다.
이전 pixel ablation은 비교용으로 보존하며 두 방법을 같은 것으로 취급하지 않는다.

PASLCD20scenes/500frames/u120/seed0에서 수정 조건의 mean-frame mIoU/F1은
**BF30: 0.4494/0.5817**, **첫 OPEN BF10: 0.5114/0.6535**다.
같은 첫 OPEN BF10의 pixel-binary 결과0.5048/0.6450보다 **+0.66%p/+0.85%p**였다.
그러나 soft 대비 FP와 반복 전이가 크게 늘어, 평균 mask 점수 개선과 detector 안정성을 구분해야 한다.

## 정확한 계산 위치

각 신규 frame에서 base와 seed를 포함한 고정 joint probe의 기여도 uᵢₚ=αᵢₚTᵢₚ를 사용한다.

```text
E⁺ᵢ = Σₚ uᵢₚ Qₚ
E⁻ᵢ = Σₚ uᵢₚ (1 − Qₚ)
mᵢ = E⁺ᵢ + E⁻ᵢ

qᵢ = E⁺ᵢ / mᵢ                    (mᵢ > 0)
zᵢ = 1[qᵢ > 0.5] = 1[E⁺ᵢ > E⁻ᵢ]
wᵢ = min(mᵢ / mass_saturation, 1)

Δpositiveᵢ = wᵢ × zᵢ
Δnegativeᵢ = wᵢ × (1 − zᵢ)
```

- z는 0/1이고, **현재 관측의 두 증분 중 한쪽만 양수**가 된다.
- 기존 관측량 가중치 w는 보존한다. w=1이면 (1,0)/(0,1), w=0.2이면 (0.2,0)/(0,0.2)다.
- 동률은 z=0이다. mass0 또는 min_evidence_mass 미만은 기존처럼 (0,0)이므로 음성 관측을 만들어내지 않는다.
- 과거 stable/candidate Beta의 a,b를 합쳐서 이진화하지 않는다. **이번 frame의 공간 합산 결과만** 이진화하고 시간 방향으로 기존 BF 누적을 한다.
- 내부 lifespan Beta는 FLIP/KEEP 좌표다. Inactive에서는 positive/negative, active에서는 negative/positive 순으로 입력한다.
- Pixel Q, 학습 loss, Panel3/Panel10, seed birth는 soft로 유지한다.
- Gaussian ratio와 pixel majority는 다르다. 같은 가중치에서 Q=[0.49,0.49,1.0]이면
  pixel 이진화 후 평균은1/3이지만 soft 합산 평균은0.66이므로 수정 방식의 z는1이다.

## 전체 결과

모든 조건은 동일500frames/u120/seed0다. 기존 네 조건은 이전 완료 run을 재사용했고
새 Gaussian-binary 두 조건을 각각500frames 실행했다. Mean-frame mIoU/F1이며 precision/recall은 pixel 통합이다.

| 조건 | mIoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| BF30 / soft | 0.4288 | 0.5576 | 0.8367 | 0.4954 |
| BF30 / pixel binary (이전) | 0.4358 | 0.5650 | 0.8305 | 0.5116 |
| BF30 / Gaussian binary (수정) | 0.4494 | 0.5817 | 0.7998 | 0.5503 |
| 첫 OPEN BF10 / soft | 0.5019 | 0.6420 | 0.8010 | 0.6167 |
| 첫 OPEN BF10 / pixel binary (이전) | 0.5048 | 0.6450 | 0.7937 | 0.6268 |
| 첫 OPEN BF10 / Gaussian binary (수정) | 0.5114 | 0.6535 | 0.7615 | 0.6640 |

Gaussian binary는 같은 BF의 soft 대비 BF30에서 **+2.06%p**, 첫 OPEN BF10에서
**+0.95%p** mIoU다. Pixel binary 대비 각각 **+1.35%p/+0.66%p**다.
첫 OPEN BF10+Gaussian binary의 pixel 통합 IoU/F1은 **0.5497/0.7094**로 mean-frame 점수와 구분한다.

## 초반 동작과 반복 전이

| 조건 | 첫5-frame mIoU | base CLOSE | seed CLOSE | detector REOPEN |
|---|---:|---:|---:|---:|
| BF30 / soft | 0.1497 | 2,833 | 21 | 이전 미분리 |
| BF30 / pixel binary (이전) | 0.1550 | 5,327 | 99 | 849 |
| BF30 / Gaussian binary (수정) | 0.1730 | 13,852 | 508 | 2877 |
| 첫 OPEN BF10 / soft | 0.3145 | 3,225 | 20 | 266 |
| 첫 OPEN BF10 / pixel binary (이전) | 0.3202 | 5,930 | 87 | 878 |
| 첫 OPEN BF10 / Gaussian binary (수정) | 0.3460 | 15,748 | 482 | 2915 |

첫 OPEN BF10+Gaussian binary는 모든20scene에서 두 번째 frame부터 출력한다.
첫5-frame mIoU는 soft0.3145, pixel binary0.3202, Gaussian binary**0.3460**이다.
BF30+Gaussian binary는 첫 두 frame이 여전히 비며 세 번째부터 출력한다.

첫 OPEN BF10의 soft→Gaussian binary에서 FP는 **1,501,858→2,037,852**,
FN은 **3,757,002→3,293,017**, detector REOPEN은 **266→2,915**로 변했다.
Base CLOSE도 **3,225→15,748**로 증가했다. Seed opacity pruning은 이 CLOSE 집계와 별도다.
0.5를 조금 넘는 약한 양성도 강한 방향 관측으로 바뀌어 recall을 높이지만, confidence 크기를
버리는 만큼 경계/view 변화에 더 민감해질 수 있다. 원인 해석은 추론이고, 전이/FP 증가 자체는 측정 사실이다.

첫 OPEN BF10+Gaussian binary는 같은 BF10 soft 대비 **13/20scene**에서 mIoU가 높고 7개에서는 낮다.
특히 Instance_2/Pots는 -0.0485다. 평균 점수만으로 일관된 개선이나 안정성 개선을 주장하지 않는다.

## Scene별 mIoU

| Scene | BF30 soft | BF30 Gaussian | 첫 BF10 soft | 첫 BF10 Gaussian |
|---|---:|---:|---:|---:|
| Instance_1/Cantina | 0.2881 | 0.3561 | 0.3925 | 0.4382 |
| Instance_1/Garden | 0.5040 | 0.4912 | 0.5127 | 0.4944 |
| Instance_1/Lounge | 0.5775 | 0.5858 | 0.6285 | 0.6370 |
| Instance_1/Lunch_room | 0.1848 | 0.2254 | 0.3038 | 0.3578 |
| Instance_1/Meeting_room | 0.4938 | 0.5067 | 0.5448 | 0.5540 |
| Instance_1/Playground | 0.4216 | 0.4454 | 0.5117 | 0.4947 |
| Instance_1/Porch | 0.4760 | 0.5132 | 0.5460 | 0.5612 |
| Instance_1/Pots | 0.4310 | 0.4660 | 0.4875 | 0.5187 |
| Instance_1/Printing_area | 0.4927 | 0.5339 | 0.6333 | 0.6678 |
| Instance_1/Zen | 0.4370 | 0.4502 | 0.4984 | 0.4817 |
| Instance_2/Cantina | 0.3079 | 0.3531 | 0.4085 | 0.4320 |
| Instance_2/Garden | 0.4794 | 0.4608 | 0.4882 | 0.4681 |
| Instance_2/Lounge | 0.4327 | 0.4342 | 0.4831 | 0.4886 |
| Instance_2/Lunch_room | 0.1779 | 0.2180 | 0.3056 | 0.3519 |
| Instance_2/Meeting_room | 0.4376 | 0.4581 | 0.4892 | 0.5202 |
| Instance_2/Playground | 0.4191 | 0.4327 | 0.5064 | 0.5003 |
| Instance_2/Porch | 0.4212 | 0.4550 | 0.4879 | 0.4934 |
| Instance_2/Pots | 0.6076 | 0.5807 | 0.6297 | 0.5812 |
| Instance_2/Printing_area | 0.4142 | 0.4424 | 0.5191 | 0.5374 |
| Instance_2/Zen | 0.5724 | 0.5786 | 0.6612 | 0.6495 |

## 구현 / 유지한 계약

- `temporal/change_evidence.py`: 기존 alpha-T VJP 결과를 사용해 `count_mode=capped_binary`로 변환한다. 별도 렌더, 새 tracker, dependency는 추가하지 않았다.
- `experiments/view_bayesian_detector_steps.py`: `--detector-gaussian-cue binary_q05`를 제공한다. Soft pixel 및 shared cue를 요구하고 pixel binary와 동시 사용은 거부한다.
- `experiments/evaluate_paslcd_panel10.py`: subprocess 옵션 전달 및 한쪽 증분/가중치/cue 일치 audit를 추가했다.
- 관련 `tests/temporal/test_change_evidence*.py`, joint detector/viewer/evaluator 회귀 테스트를 확장했다.
- 기본 raw/capped 동작과 기본 CLI 설정은 유지한다. 수정 Gaussian-binary 조건은 명시적 실험 옵션이다.
- Seed를 먼저 넣고 공동 detector가 현재 frame을 한 번 처리한 다음 학습한다. Learned DC, post-opt, replay를 detector evidence로 다시 사용하지 않는다.
- First-OPEN BF10 옵션도 그대로이며 CLOSE/REOPEN은 BF30이다. Base DC / seed DC+geometry+density+pruning 학습 계약은 바꾸지 않았다.
- NEVER_OPEN은 frozen black occluder, CLOSED는 최종 렌더에서 숨긴다.

## 검증과 한계

- CPU/CUDA/계층 통합 **178 tests passed**. Gaussian 동률·zero/low mass·한쪽 증분·pixel/Gaussian 차이·공동 detector 순서를 확인했다.
- 새 두 run의 모든1,000frames에서 **현재 관측의 mixed positive/negative row 수는0**이다.
- Saved mask, 두 O-SCD baseline, 전체 confusion count/aggregate를 재검산했다. 이전 네 조건까지 비교 script에서 같은 GT/순서를 확인했다.
- Reference/fixed probe/frozen parameter/Adam drift, future-view access, lifespan render violation **0**. Joint evidence는 frame당1회다.
- Source/input hash가 실행 중 변하지 않았음을 확인했다. AST/compile/whitespace/shell 검증 통과.
- 독립 코드 리뷰 0 concrete issues. 전용 LSP 도구는 없어 formal approval은 보류되고 compile/tests로 대체 검증했다.
- 두 MP4 모두500frames/10fps/50초/1920×1120. 전체 디코딩, 마지막 frame 비교를 통과했고 실제 마지막 화면도 확인했다. 기존 letterbox/제목 잘림/reserved panel은 유지한다.
- Single seed0/CUDA 비결정성, 이전 baseline 재사용의 한계가 있다. 다중 seed 유의성은 검증하지 않았다.
- PASLCD는 각 scene 단일 post-change 상태다. ESCD 연속 상태 전환은 수정 옵션으로 아직 재실행하지 않았다.
- 원본 O-SCD online은 u16, 현재 실험은 u120 및 보조 causal O-SCD teacher 사용이므로 비용 기준 우월성을 주장하지 않는다.

## 산출물 / 재현

```bash
bash outputs/paslcd_panel10_gaussian_binary_20260907/start.sh
```

`start.sh` 완료 후 `finish.sh`를 실행하면 saved mask/MP4/비교 검증을 재수행한다.

- [전체 비교 JSON](../outputs/paslcd_panel10_gaussian_binary_20260907/comparison.json)
- [계산 정의·실험 계획](../outputs/paslcd_panel10_gaussian_binary_20260907/plan.json)
- [테스트](../outputs/paslcd_panel10_gaussian_binary_20260907/tests.txt)
- BF30 Gaussian binary: [MP4](../outputs/paslcd_panel10_gaussian_binary_20260907/bf30_gaussian_binary/viewer_dashboard_all_frames.mp4), [frame metrics](../outputs/paslcd_panel10_gaussian_binary_20260907/bf30_gaussian_binary/frame_metrics.csv), [검증](../outputs/paslcd_panel10_gaussian_binary_20260907/bf30_gaussian_binary/posthoc_verification.json)
- 첫 OPEN BF10 Gaussian binary: [MP4](../outputs/paslcd_panel10_gaussian_binary_20260907/first10_gaussian_binary/viewer_dashboard_all_frames.mp4), [frame metrics](../outputs/paslcd_panel10_gaussian_binary_20260907/first10_gaussian_binary/frame_metrics.csv), [검증](../outputs/paslcd_panel10_gaussian_binary_20260907/first10_gaussian_binary/posthoc_verification.json)
