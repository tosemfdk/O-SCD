# Budgeted View Selection for O-SCD — 발견 정리 (2026-07-16 기준)

연구 질문의 진화, 실측 결과, 그리고 "현상"의 종합 해석.
상세 수치·재현 방법은 각 리포트 참조 (문서 끝 인덱스).

---

## 0. 질문의 진화

1. "적은 view로 25장과 동일한 mIoU가 나오나?" (예산 절약 프레임)
2. → "uniform이 이미 잘하는데 스마트 선택의 여지가 있나?" (노이즈 발견)
3. → "가능한 최적 5장 조합(oracle)은 얼마나 좋은가?" (상한 측정)
4. → **"왜 어떤 프레임은 돕고 어떤 프레임은 해치는가?"** (현재 위치)

## 1. 평가 프로토콜과 노이즈 구조 (모든 해석의 전제)

- **공정 평가 원칙**: 업데이트에 몇 장을 쓰든, 평가는 항상 최종 R_change를
  **25개 query pose 전부에서 렌더**해 GT 25장과 비교 (`query_mask/`).
- **노이즈는 3층 구조**:
  - (a) 배치 간 시프트: cuDNN benchmark의 타이밍 기반 알고리즘 선택 때문에
    같은 명령이 배치마다 다른 값 (Garden all-25: 0.444 / 0.452 / 0.488).
    **같은 배치 안에서는 거의 완전 재현** (all-25 5회 std ≈ 0.000, 9/10 씬).
  - (b) 선택 민감도: 어떤 5장이냐에 따라 씬당 ±0.05~0.11
    (uniform stride-5 offset 5종의 std: Pots 0.018 ~ Zen 0.090).
  - (c) fusion 카오스: 선택 집합이 조금 바뀌면 densify/최적화 경로가 크게
    갈라짐 — 씬 단위 단일 런 비교는 ±0.05~0.1 오차 취급.
- 결론: **비교는 반드시 같은 배치 + 다씬 집계**로. Garden에서 잡았던
  ±0.005는 심한 과소평가였다.

## 2. Uniform의 실력 (기준선의 실체)

- 포화 곡선 (Instance_1 10씬 평균, all-25 대비):
  K=2 78% → K=3 88% → **K=5 94%** → K=8 96% → K=10 98%. 95% 도달 K≈7.
- K=5에서 **사람(실제 이미지를 보고 고른 5장)도 uniform을 못 이겼다**
  (Lounge: human 0.5446 vs uniform@5 0.5812). K=5 구간의 셀렉터 간 차이는
  대부분 (c) 노이즈다. → "uniform@5 이기기"는 목표로서 부적절.
- 단, uniform은 offset 하나 차이로 0.36~0.48 (Cantina)까지 출렁인다.
  단일 uniform 런을 기준선으로 쓰면 안 됨 (5종 평균 필요).

## 3. 셀렉터 실험 (음성 결과 포함, 전부 기록)

- **nbv (Beta-EIG 커버리지)**: Garden에선 +0.106으로 이겼지만 10씬에서
  uniform에 패배 (K=5 평균 −0.039, W/T/L 2/2/6). 원인: 방향 무시 —
  "한 번 본 것 = 해결"로 취급해 재관측을 회피하지만, R_change의 3D
  국소화는 변화 영역의 **시차 다양 재관측**을 요구.
- **nbv_dopt (방향 인지 D-opt)**: 진단된 씬(Porch)은 정확히 고쳤으나
  (−0.11 → +0.12) Zen이 붕괴 (1/d² 근접 편향 의심). 여전히 uniform에
  약간 뒤짐 (K=5 −0.024). 사람의 선택 패턴(변화 지점을 각도 바꿔
  재관측)에 근접한 수준까지는 도달.
- **target 수준에서는 D-opt가 확실히 검증됨** (stage-14 pool 실험):
  단일 Gaussian의 기하 불확실성 기준, exact D-opt는 **2장으로 uniform
  5장의 정보량 도달** (10/10 target), blind 선택은 예산의 26–28%를
  target이 안 보이는 view에 낭비.

## 4. 핵심 결과: Oracle-5 지도

**잘 고른 5장은 25장 전부보다 낫다 — 10/10 씬, 평균 +10.3% (+2.4~+25.3%).**
씬당 겨우 ~30런의 hill-climbing으로 찾은 하한값.

| 씬 | best-5 | all-25 | Δ | 최적 조합 |
|---|---|---|---|---|
| Playground | 0.4499 | 0.3590 | +25.3% | (1,10,16,17,24) |
| Garden | 0.5498 | 0.4520 | +21.6% | (1,5,7,10,12) |
| Lunch_room | 0.4177 | 0.3714 | +12.5% | (1,3,5,9,22) |
| Meeting_room | 0.5661 | 0.5151 | +9.9% | (10,12,19,20,24) |
| Lounge | 0.5868 | 0.5345 | +9.8% | (0,1,3,11,20) |
| Pots | 0.6520 | 0.6116 | +6.6% | (4,6,11,13,21) |
| Zen | 0.5694 | 0.5368 | +6.1% | (2,12,15,16,23) |
| Cantina | 0.5900 | 0.5655 | +4.3% | (12,15,17,20,24) |
| Porch | 0.6292 | 0.6040 | +4.2% | (4,9,15,19,24) |
| Printing_area | 0.7031 | 0.6867 | +2.4% | (2,8,13,20,23) |

구조적 특징:
- **균등 분산이 아니다**: Cantina 최적셋은 전부 후반부(12–24),
  Garden은 전부 전반부(1–12). "골고루"는 절반의 씬에서 틀린 prior.
- **앵커 프레임**: 씬마다 2–3장이 +0.03~+0.08의 한계기여 —
  oracle 셋은 10/10 씬에서 top-3 앵커 중 2–3장을 포함.
- **독성 프레임**: −0.03~−0.11 (Porch/0 −0.101, Zen/3 −0.109).
  **all-25가 지는 이유 = 독성 프레임을 강제로 전부 섭취하기 때문.**

## 5. 독성의 정체 (부분 해명)

프레임 단독(GT-free) 특징과 한계기여도의 상관 (n=200):
- **pose 품질은 무죄** (r ≈ 0; 베이스라인 pose 실패 0/500과 일관).
- 최강 신호: cue 면적/질량 (r ≈ −0.18) — **cue가 클수록 해롭다**.
- 확인된 독성 유형: **"크고 제멋대로인 cue"** — cue 면적 상위 88–96%인데
  3D 합의 일치도 하위 4–32% (Zen/3,4, Cantina/11, Playground/19).
  기제: fusion loss는 cue 위치의 마스크를 올리기만 하므로 대형 오탐
  cue의 손상을 되돌릴 수단이 없음.
- 그러나 2-특징 필터는 최악 프레임의 6/30만 검출 (우연 3.6/30) —
  나머지 독성은 프레임 단독 특징에 없고, **셋 맥락(이미 고른 프레임과의
  시차/중복 관계)**에 있다.

## 6. 현상의 종합 그림

1. 프레임의 가치는 극도로 불균등하다: 소수의 앵커 + 소수의 독성 +
   다수의 중립. 정보가 시퀀스의 특정 구간에 뭉쳐 있는 씬이 절반.
2. **all-25는 상한이 아니라 "무필터 섭취"다.** 더 먹을수록 오탐 cue가
   누적되고, up-only 손실 구조가 이를 고착시킨다.
3. 따라서 view selection의 본질은 예산 절약이 아니라
   **앵커 포함 + 독성 거부 + (셋 수준) 시차 확보** = 품질 그 자체.
4. K=5에서 uniform과 다투는 것은 노이즈 게임이다. 의미 있는 목표는
   (i) oracle과의 갭 (현재 uniform-best 대비 oracle +0.8~+24.2%),
   (ii) K≤3 극저예산 구간, (iii) uniform이 존재할 수 없는 온라인/로봇
   세팅.
5. 검증된 부품들: target-수준 D-opt(2장=uniform 5장 정보), 방향 인지
   셋 기준(Porch 반전), cue-일치도 신호(최악 유형 검출). 미조립 상태.

## 7. 미해결 질문

- 독성의 나머지 절반(Porch/0류)의 기제는? (셋 조건부 특징 필요)
- oracle 조합의 시차 구조 — 변화 영역 기준 baseline 분포로 설명되는가?
- 1/d² 근접 편향 수정 시 nbv_dopt가 K≤3에서 uniform을 넘는가?
- Instance_2에서 oracle-5 지도가 재현되는가?
- 온라인 세팅(순차 도착, 미래 프레임 모름)에서 anchor/독성을 실시간
  판별 가능한가? — stage 13의 질문이자 이 연구의 실용 종착지.

## 8. 파일 인덱스

| 내용 | 파일 |
|---|---|
| 이 문서 | `docs/budgeted_view_findings.md` |
| target-NBV 설계 문서 | `docs/target_gaussian_nbv.md` |
| stage-14 pool 실험 (target D-opt 검증) | `experiments/pool_eval_garden/report.md` |
| Garden 셀렉터 비교 | `experiments/garden_nbv_report.md` |
| Instance_1 셀렉터 비교 + 진단 | `experiments/paslcd_nbv_report.md` |
| oracle-5 지도 | `experiments/oracle5_map_report.md` |
| 프레임 특징 분석 | `experiments/frame_features_report.md` |
| 러너/드라이버 | `subset_oscd.py`, `experiments/{oracle_search,oracle_local_search,frame_feature_analysis}.py` |
| 원시 데이터 (tracked) | `experiments/{paslcd_nbv_results,garden_nbv_results,garden_sweep_results}.csv` |
| 원시 데이터 (untracked, d0d016d 정책) | `experiments/{oracle_search_results,all25_repeats,frame_features}.csv` |

주요 커밋: 베이스라인 동결 `ab3c7e6` → Gate A–F (target-NBV MVP) →
`6464cb1`/`09270d9` (change 모드+통합) → `fd20e22`/`e6fb372` (셀렉터 음성
결과) → `c04fc85` (oracle-5 지도) → `cfd6b0d` (프레임 특징 분석).
