# PASLCD: 첫 OPEN BF 완화와 pixel-cue 이진화 ablation (2026-09-07)

> 후속 정정: 사용자의 의도는 pixel 이진화가 아니라 **Gaussian 합산 후 이진화**였다.
> 이 문서는 이전 pixel ablation 기록이며, 수정 실험은
> [Gaussian aggregate binary 보고서](paslcd-gaussian-aggregate-binary-ablation-20260907-ko.md)에 별도로 기록한다.

## 결론

PASLCD **20 scenes / 500 frames / seed0 / u120**에서 첫 OPEN만 BF30→10으로 낮추면
mean-frame mIoU/F1이 **0.4288/0.5576 → 0.5019/0.6420**으로 증가했다.
Pixel Q를 0.5에서 이진화까지 하면 **0.5048/0.6450**이었다. BF10의 효과가 크고,
이진화의 추가 효과는 **+0.00285 mIoU (+0.29%p)**로 작다.
두 BF10 조건 모두 기존 대비 **20/20 scene**에서 mIoU가 높았다. 기본 설정은 변경하지 않았다.

## 비교 조건

- BF 완화는 Gaussian별 `NEVER_OPEN → 첫 OPEN`에만 적용한다. Base와 뒤늦게 태어난 seed에 동일 적용한다.
- CLOSE와 REOPEN은 모든 조건에서 **BF30**이다. 이미 OPEN인 parent에서 생성된 density child는 첫 OPEN 예외를 받지 않는다.
- Binary는 **정규화된 각 pixel Q > 0.5**를 1, 나머지를 0으로 바꾼 뒤 alpha-T로 합산한다. Q=0.5는 0이다.
- 합산된 Gaussian 평균을 0/1로 만드는 실험은 아니다. 한 Gaussian에 a,b가 함께 쌓이는 것을 금지하지 않는다.
- **Loss, Panel3/Panel10, seed birth에는 기존 soft Q를 유지**한다. Detector 입력만 이진화한다.
- 이전 BF30+soft 평가를 기준으로 재사용하고 나머지 세 조건을 각각 500 frames 새로 실행했다.
- Raw cue/DA3/cache, priors, u120, uncapped archive, replay, density/pruning 조건은 동일하다.
- 각 scene은 독립 초기화하며, seed를 먼저 삽입하고 단일 joint base+seed evidence pass 후 학습한다.

## 전체 결과

아래 mIoU/F1은 foreground frame별 점수의 평균이다. Precision/recall은 전체 pixel 통합이다.

| 조건 | mIoU | F1 | Precision | Recall | 기존 대비 mIoU |
|---|---:|---:|---:|---:|---:|
| BF30 + soft Q (기존) | 0.4288 | 0.5576 | 0.8367 | 0.4954 | +0.00%p |
| 첫 OPEN BF10 + soft Q | 0.5019 | 0.6420 | 0.8010 | 0.6167 | +7.31%p |
| BF30 + binary Q | 0.4358 | 0.5650 | 0.8305 | 0.5116 | +0.70%p |
| 첫 OPEN BF10 + binary Q | 0.5048 | 0.6450 | 0.7937 | 0.6268 | +7.60%p |

BF10+soft 대비 BF10+binary는 precision **0.8010→0.7937**, recall **0.6167→0.6268**이다.
기존 대비 BF10+binary는 FN **1,287,667 pixel 감소**, FP **649,547 pixel 증가**다.
더 빨리/많이 열어 누락을 줄이는 대신 오검출도 늘었다.

참고: 동일 GT로 재평가한 원본 O-SCD online(u16)은 mIoU/F1 **0.4906/0.6440**이다.
BF10+binary(u120)는 +1.41%p/+0.10%p이지만 compute가 다르고 보조 causal O-SCD teacher를
사용하므로 공정한 비용 기준 우월성으로 주장하지 않는다.

## 초반 OPEN 속도

| 조건 | 첫 5-frame mIoU | 두 번째 frame mIoU | 두 번째 frame nonempty scene |
|---|---:|---:|---:|
| BF30 + soft Q (기존) | 0.1497 | 0.0000 | 0/20 |
| 첫 OPEN BF10 + soft Q | 0.3145 | 0.3072 | 20/20 |
| BF30 + binary Q | 0.1550 | 0.0000 | 0/20 |
| 첫 OPEN BF10 + binary Q | 0.3202 | 0.3157 | 20/20 |

모든 조건에서 첫 frame은 빈 mask다. BF30 조건 둘은 두 번째 frame도 비어 있고,
BF10 조건 둘은 모든 scene에서 두 번째 frame부터 출력한다.
첫 5-frame 평균은 **0.1497→0.3145→0.3202**(기존→BF10 soft→BF10 binary)로 변한다.

| Frame 구간 | BF30 soft | 첫 BF10 soft | BF30 binary | 첫 BF10 binary |
|---|---:|---:|---:|---:|
| 1-5 | 0.1497 | 0.3145 | 0.1550 | 0.3202 |
| 6-10 | 0.4170 | 0.5153 | 0.4318 | 0.5194 |
| 11-15 | 0.5131 | 0.5599 | 0.5202 | 0.5605 |
| 16-20 | 0.5265 | 0.5544 | 0.5304 | 0.5553 |
| 21-25 | 0.5378 | 0.5655 | 0.5416 | 0.5685 |

## 이진화해도 a,b가 함께 쌓이는 이유

Gaussian i가 pixel p에 기여하는 고정 probe alpha-transmittance를 uᵢₚ라 하면:

```text
soft:    cₚ = Qₚ
binary:  cₚ = 1[Qₚ > 0.5]

E⁺ᵢ = Σₚ uᵢₚ cₚ
E⁻ᵢ = Σₚ uᵢₚ (1 − cₚ)
mᵢ  = E⁺ᵢ + E⁻ᵢ
q̄ᵢ  = E⁺ᵢ / (mᵢ + ε)
wᵢ  = min(mᵢ / mass_saturation, 1)

Δpositiveᵢ = wᵢ q̄ᵢ
Δnegativeᵢ = wᵢ (1 − q̄ᵢ)
```

따라서 binary pixel이더라도 Gaussian footprint가 0/1 pixel을 함께 덮으면 두 evidence가
모두 양수다. 공간적 가중 평균은 유지되며, 단지 Q=0.6처럼 약한 양성 pixel이 1로,
Q=0.4처럼 약한 음성 pixel이 0으로 강화된다. 한 Gaussian의 관측당 총 pseudo-count는
여전히 최대 1이다. Pixel 이진화가 정수/독립 Bernoulli count를 보장하는 것은 아니다.

`lifespan_gate_beta` 내부 a,b는 FLIP/KEEP 좌표다. 현재 inactive면 positive/negative가
각각 FLIP/KEEP이고, 현재 active면 반대다. Learned DC는 detector에 입력하지 않는다.

## 반복 전이 / topology

| 조건 | base OPEN | base CLOSE | seed OPEN | seed CLOSE |
|---|---:|---:|---:|---:|
| BF30 + soft Q (기존) | 88,327 | 2,833 | 29,915 | 21 |
| 첫 OPEN BF10 + soft Q | 135,302 | 3,225 | 41,876 | 20 |
| BF30 + binary Q | 97,474 | 5,327 | 32,093 | 99 |
| 첫 OPEN BF10 + binary Q | 154,528 | 5,930 | 45,293 | 87 |

OPEN 표에는 REOPEN도 포함한다. Seed opacity pruning은 위 detector CLOSE와 별도다.
BF10 soft→binary에서 detector REOPEN 합은 **266→878**, base CLOSE는 **3,225→5,930**으로
증가했다. PASLCD는 scene별 단일 post-change 상태이므로 추가 이진화가 더 안정적인
detector라는 증거는 아니다. BF10+soft를 우선 후보로 보고, binary 추가 채택은 반복 실행과
연속 evolving-scene CLOSE/REOPEN 검증 뒤 판단하는 것이 적절하다.

## Scene별 mIoU

| Scene | BF30 soft | 첫 BF10 soft | BF30 binary | 첫 BF10 binary |
|---|---:|---:|---:|---:|
| Instance_1/Cantina | 0.2881 | 0.3925 | 0.3050 | 0.4045 |
| Instance_1/Garden | 0.5040 | 0.5127 | 0.5058 | 0.5115 |
| Instance_1/Lounge | 0.5775 | 0.6285 | 0.5835 | 0.6311 |
| Instance_1/Lunch_room | 0.1848 | 0.3038 | 0.1914 | 0.3150 |
| Instance_1/Meeting_room | 0.4938 | 0.5448 | 0.5008 | 0.5475 |
| Instance_1/Playground | 0.4216 | 0.5117 | 0.4315 | 0.5151 |
| Instance_1/Porch | 0.4760 | 0.5460 | 0.4989 | 0.5575 |
| Instance_1/Pots | 0.4310 | 0.4875 | 0.4311 | 0.4858 |
| Instance_1/Printing_area | 0.4927 | 0.6333 | 0.5011 | 0.6419 |
| Instance_1/Zen | 0.4370 | 0.4984 | 0.4454 | 0.4930 |
| Instance_2/Cantina | 0.3079 | 0.4085 | 0.3145 | 0.4132 |
| Instance_2/Garden | 0.4794 | 0.4882 | 0.4756 | 0.4846 |
| Instance_2/Lounge | 0.4327 | 0.4831 | 0.4352 | 0.4802 |
| Instance_2/Lunch_room | 0.1779 | 0.3056 | 0.1842 | 0.3127 |
| Instance_2/Meeting_room | 0.4376 | 0.4892 | 0.4463 | 0.5028 |
| Instance_2/Playground | 0.4191 | 0.5064 | 0.4164 | 0.5052 |
| Instance_2/Porch | 0.4212 | 0.4879 | 0.4464 | 0.4971 |
| Instance_2/Pots | 0.6076 | 0.6297 | 0.5987 | 0.6134 |
| Instance_2/Printing_area | 0.4142 | 0.5191 | 0.4217 | 0.5232 |
| Instance_2/Zen | 0.5724 | 0.6612 | 0.5831 | 0.6597 |

## 구현 / 검증 / 한계

- `temporal/single_candidate_beta.py`: 선택 row별 log-BF commit threshold override; 후보 posterior commit도 같은 threshold를 사용한다.
- `temporal/lifespan_gate_beta.py`: lifecycle의 first-open mask로 override를 제한한다. Default/기존 capacity 확장은 유지한다.
- `experiments/view_bayesian_detector_steps.py`: `--first-open-bayes-factor-threshold 10`,
  `--detector-pixel-cue binary_q05`; normalized pixel cue만 detector로 분기한다.
- `experiments/evaluate_paslcd_panel10.py`: subprocess 옵션 전달, evidence cue/cap 검증, first/reopen 이벤트 기록.
- 기존 evidence converter를 재사용했다. 새 detector bank나 dependency는 추가하지 않았다.
- 회귀/CUDA **161 tests passed**, 독립 code review 0 issues; AST/compile/whitespace/shell syntax 통과. 전용 LSP/typechecker는 사용 불가였다.
- 새 1,500개와 기준 500개 saved prediction mask를 GT와 재검산했다. 새 run의 두 baseline도 재검산했고 frame 순서와 비교 조건의 일치를 확인했다.
- 세 run 모두 reference/fixed probe/frozen parameter/Adam drift, future-view access, lifespan render violation **0**; evidence pass는 frame당 **1회**다.
- 세 MP4 모두 **500 frames / 10fps / 50초 / 1920×1120**, 전체 디코딩 및 마지막 frame 비교 통과.
- 첫 양성 frame과 마지막 decoded frame을 직접 확인했다. 기존 letterbox, 일부 panel 제목 잘림, reserved column은 유지한다.
- Single seed0이며 CUDA 비결정성이 있다. Baseline은 이전 run을 재사용했다. 작은 binary 추가 이득의 통계적 유의성은 검증하지 않았다.
- 이번 실행은 PASLCD만이다. ESCD 304-frame 연속 상태 전환은 이 옵션으로 재평가하지 않았다.
- 이 결과는 첫 OPEN의 민감도 ablation이지, seed birth/semantic sign calibration/연속 상태 추적 문제 전체 해결은 아니다.

## 재현 / 산출물

```bash
bash outputs/paslcd_panel10_detector_ablation_20260907/start.sh
bash outputs/paslcd_panel10_detector_ablation_20260907/finish.sh
```

- [전체 비교 JSON](../outputs/paslcd_panel10_detector_ablation_20260907/comparison.json)
- [실험 계획과 고정 조건](../outputs/paslcd_panel10_detector_ablation_20260907/plan.json)
- [테스트](../outputs/paslcd_panel10_detector_ablation_20260907/tests.txt)

- 첫 OPEN BF10 + soft Q: [MP4](../outputs/paslcd_panel10_detector_ablation_20260907/first10_soft/viewer_dashboard_all_frames.mp4), [metrics](../outputs/paslcd_panel10_detector_ablation_20260907/first10_soft/frame_metrics.csv), [검증](../outputs/paslcd_panel10_detector_ablation_20260907/first10_soft/posthoc_verification.json)
- BF30 + binary Q: [MP4](../outputs/paslcd_panel10_detector_ablation_20260907/bf30_binary/viewer_dashboard_all_frames.mp4), [metrics](../outputs/paslcd_panel10_detector_ablation_20260907/bf30_binary/frame_metrics.csv), [검증](../outputs/paslcd_panel10_detector_ablation_20260907/bf30_binary/posthoc_verification.json)
- 첫 OPEN BF10 + binary Q: [MP4](../outputs/paslcd_panel10_detector_ablation_20260907/first10_binary/viewer_dashboard_all_frames.mp4), [metrics](../outputs/paslcd_panel10_detector_ablation_20260907/first10_binary/frame_metrics.csv), [검증](../outputs/paslcd_panel10_detector_ablation_20260907/first10_binary/posthoc_verification.json)
