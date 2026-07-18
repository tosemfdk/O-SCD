# Gate S2 Report — Score Validity (2026-07-18)

**판정: NO_GO.** change-channel diagonal conditional information(G_D×pose)은
조건부 oracle marginal을 설명하지 못한다 (median ρ = −0.02). 스펙 §30에 따라
여기서 정지 — full sweep 진행 금지, 사다리 다음 단계는 사용자 승인 사안.

## 프로토콜

- 씬: Zen(워치리스트), Porch, Garden(360), Instance_1. 컨텍스트 S ∈ {∅, {12}, {6,18}}.
- oracle marginal: Δ(v|S) = mIoU₂₅(replay(S∪{v})) − mIoU₂₅(replay(S)),
  replay = 표준 manual 경로, **222런 전부 같은 머신·같은 배치**, 8×V100 병렬 (~35분).
- 스코어: b_v는 reference scoring state(c=0)에서 Hutchinson M=4, pose weight
  (reference alpha V), λ = relative-median. 전 스코어 GT 미접촉.
- 데이터: `outputs/change_nbv/s2/{oracle_marginals,score_validity}.csv`, `summary.json`.

## 결과 (Spearman ρ, score vs Δ(v|S))

| method | Zen S0/S1/S2 | Porch S0/S1/S2 | Garden S0/S1/S2 | median | top-1 regret(med) |
|---|---|---|---|---|---|
| **candidate_only** | +0.19/+0.46/+0.37 | +0.22/+0.35/+0.20 | −0.22/−0.16/−0.00 | **+0.199** | **0.040** |
| **max_alpha** | +0.33/+0.38/+0.21 | +0.02/+0.13/+0.47 | −0.01/−0.02/+0.21 | **+0.210** | 0.071 |
| dopt_pose | +0.19/−0.40/+0.04 | −0.02/−0.68/−0.02 | +0.00/+0.18/−0.12 | −0.020 | 0.111 |
| fisher_ratio | +0.19/−0.44/−0.05 | +0.22/−0.62/−0.17 | −0.22/−0.01/−0.11 | −0.106 | 0.109 |
| trace_reduction | −0.03/−0.47/+0.01 | −0.13/−0.70/−0.05 | −0.06/−0.07/−0.14 | −0.065 | 0.118 |
| max_cue_area | −0.61/−0.75/−0.56 | +0.42/+0.35/−0.15 | −0.05/+0.03/−0.10 | −0.100 | 0.182 |

보조(게이트 아님) — 기존 Instance_1 지도 marginal(씬당 ~1,018세트)과의 ρ:
max_alpha +0.24~+0.42 (3/3 양수), candidate_only Zen +0.42, dopt_pose Garden −0.39,
max_cue_area Zen −0.74. 게이트 결과와 방향 일치.

## 해석 — 실패의 구조가 발견이다

1. **prior 정규화가 부호를 뒤집는다.** S=∅에서는 dopt ≈ candidate_only(같은
   순위)지만, 컨텍스트가 생기는 순간(S1) dopt/fisher/trace가 강한 음수로 반전
   (Porch −0.68, Zen −0.40) — 반면 prior를 무시하는 candidate_only는 양수 유지
   (+0.35/+0.46). 즉 **"이미 본 Gaussian은 할인한다"는 중복 억제가 정확히
   반대 방향의 신호**다. c-채널 diagonal에서 b의 중복 = responsibility 겹침 =
   같은 영역 재관측인데, 실제 SCD marginal은 재관측을 **선호**한다.
   이것은 nbv(Beta-EIG)의 실패 진단("한 번 본 것 = 해결로 취급 → 재관측 회피")
   의 **세 번째 독립 확인**이며, 이번엔 criterion 수준에서 정량화됐다.
2. **단순 가시성 질량 신호가 약하게 유효하다**: candidate_only(+0.20)와
   max_alpha(+0.21), regret도 최소(0.040/0.071). "변화 영역을 많이·잘 보는
   프레임"까지는 GT-free로 잡히지만, 그 이상(어떤 조합이 좋은가)은 diagonal
   c-정보가 담지 못한다.
3. **독성 cue 시그니처 재현**: max_cue_area가 Zen에서 ρ −0.61~−0.75 —
   "큰 cue가 해롭다"가 조건부 marginal에서도 재확인 (Porch S0/S1에서는 +0.42/
   +0.35로 씬 의존적 — cue 크기는 단독 신호로 부적합하다는 기존 결론 유지).
4. 구조적 원인은 스펙 §25가 예고한 그대로: diagonal은 ray-방향 교차상관
   (시차가 푸는 것)을 버리고, c는 opacity처럼 합성 관측이라 POp-GS의 opacity
   실패 기제와 동형. "셋 맥락이 프레임 가치의 절반"이라는 지도 분석과 정합.

## 스펙 §30 종료 조건 적용

구현은 완료(코드·테스트·게이트 산출물 전부 존재), 연구 판정은 **"change-channel
diagonal conditional information은 SCD marginal value를 설명하지 못한다"**로
기록하고 정지한다. M3·CUDA 최적화로 진행하지 않는다.

## 사다리 다음 단계 후보 (사용자 승인 필요 — 자동 진행 안 함)

데이터가 가리키는 순서:
1. **prior 부호 반전 가설의 직접 검증** (승인 시 최우선, 반나절): criterion을
   "중복 페널티"가 아니라 "재관측 보너스"로 뒤집은 변형(예: G(v|S) =
   Σ min(b_v, h_S−λ) 류의 겹침 보상)이 S1/S2 컨텍스트에서 양의 ρ를 내는지 —
   같은 S2 데이터로 오프라인 검증 가능 (**GPU 0, 즉시**).
2. consensus weight (스펙 사다리 5단계): cue 3D-합의(+0.177 신호)를 픽셀
   가중으로 — 단, pool-based로 재명명.
3. current_map (M2): 첫 fusion 후의 R_change 상태 가중 — 재관측 신호를
   모델 상태에서 얻는 change-aware 경로.
