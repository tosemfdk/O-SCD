# ChangeNBV — Change-Channel P-Optimal View Selection 실행 플랜 (최종 병합본)

작성: 2026-07-18. 이 문서는 세 차례의 플랜 리뷰(ChangeNBV 초안 → Codex 안 → 수정안)를
병합한 **확정 실행 스펙**이다. 근거 논의는 대화 기록에 있고, 여기에는 결정과 실행
내용만 적는다. 백본은 이 레포의 O-SCD 파이프라인이며, **offline 설정**(inference
pool 전체가 주어진 상태)에서 어떤 뷰를 R_change 최적화에 투입할지 선택한다.

핵심 연구 질문: **어떤 candidate pose가 이미 선택된 뷰들과 중복되지 않으면서,
per-Gaussian change parameter c의 posterior 불확실성을 가장 크게 줄이는가?**

프레이밍(논문 서사): 본질은 "multi-view change consensus를 빠르게 만드는 view
selection". pose-only / current-map 모드는 **NBV(active)** 주장이 가능하고,
cue를 쓰는 모드는 **pool-based view subset selection**으로 정직하게 구분한다.

---

## 0. 확정 설계 결정 (변경 금지)

| # | 결정 | 근거 요약 |
|---|---|---|
| D1 | B³-Seg 계열(Beta 상태, EIG 가중) **완전 배제** | 사용자 결정. 기존 nbv 셀렉터의 음성 결과와도 부합 |
| D2 | 정보 대상 파라미터 = **change channel c만** (`_features_dc`) | O-SCD의 유일한 change 파라미터. RGB/기하 파라미터는 비범위 |
| D3 | 측정 공간 기본 = **raw logit z** (sigmoid는 ablation) | 기하 고정 시 z는 c에 선형 → Fisher 정확·캐시 가능. σ′ 할인 기능은 weight 모드로 분리 |
| D4 | **c 채널 projection**: 3채널 dc를 스칼라로 합산 (`grad_c = grad_dc.sum(ch)`) | 렌더가 채널 mean이라 3채널 Jacobian 동일. 주의: c-only diagonal에서는 **순위 불변** (점수 상수 3배) — 위생·메모리·후속 블록 확장용이지 성능 기대 없음 |
| D5 | Criterion: **D-optimality 메인**, trace-reduction/FisherRF-ratio/candidate-only는 비교군 | POp-GS 결과(D/T ≫ A/E) + 누적 prior가 중복 억제를 만드는 건 logdet 오목성 |
| D6 | Prior 처리: **batch-static은 캐시 b 누적, sequential류는 매 라운드 H 전체 재계산** `H_S(θ_t) = λ + Σ_{s∈S} b_s(θ_t)` | POp-GS 원 방식(평가 시 freeze → 선택 후 학습 → 새 구조에서 Hessian 재계산). densify 차원 변화·기하 드리프트를 한 번에 해소. append 방식 폐기 |
| D7 | **최종 학습·평가는 모든 모드에서 "선택된 셋에 대한 표준 파이프라인 재실행"** (시간순, 기존 subset 프로토콜, 마무리 조인트 최적화 없음) | 셀렉터 간 차이 = 선택된 셋뿐. 기존 동결 수치(oracle 지도, uniform 곡선)와 호환 유지 |
| D8 | 평가는 항상 **25개 query pose 전부** 렌더 vs GT (기존 `subset_oscd.py` 프로토콜) | `--test_hold`는 no-op 함정. seen/unseen 분리는 같은 렌더에서 사후 분할 |
| D9 | selection에 GT mask 절대 사용 금지. `dopt_pose` 경로는 **이미지 로더 접근 시 실패하는 no-leakage 테스트** 필수 | |
| D10 | 기존 `all` 경로(`oscd.py`, `subset_oscd.py --frames_method all`) 동작 불변, regression 테스트로 보장 | 베이스라인 동결 정책 |

---

## 1. 고정 사실 (재확인 불필요 — 인용만)

### 1.1 이미 완료된 검증 (다시 돌리지 말 것)

- **베이스라인 재현(G0 상당)**: 동결 완료. 태그 `baseline-freeze-phase0`,
  20-instance 온라인 0.4887/0.6423, 정제 0.5573/0.7009
  (`artifacts/baseline/discrepancy_report.md`).
- **Headroom(G1 상당)**: oracle-5 지도로 확정 — **10/10 씬에서 best-5 > all-25,
  평균 +12.2%**, oracle 플래토 얕음(20배 탐색에도 +3% 미만 갱신), Instance_2
  전이 r=0.89 (`experiments/oracle5_map_report.md`, `docs/budgeted_view_findings.md`).
- **프레임 가치 구조**: 앵커/독성 프레임 존재. 독성 = "크고 제멋대로인 cue"
  (cue 3D-합의 일치도 상관 +0.173 = 최강 신호, cue 면적 −0.14, pose 품질 r≈0).
- **기존 셀렉터 음성 결과**: nbv(Beta-EIG 커버리지) K=5 평균 −0.039;
  nbv_dopt(시야기하 proxy D-opt + Beta 가중) K=5 −0.024, Zen 붕괴(−0.288@K3,
  1/d² 근접 편향 의심) (`experiments/paslcd_nbv_report.md`).

### 1.2 코드 사실 (audit 선답변)

- change 파라미터 = `_features_dc (N,3,1)`, `load_ply_change`가 **0으로 초기화**
  (`scene/gaussian_model.py:427-430`). 3채널은 gradient가 동일해 학습 내내 동일
  유지 → 실질 스칼라.
- raw logit z는 `render_change`가 그대로 반환, sigmoid는 loss에서 적용
  (`subset_oscd.py:291`). raw 접근에 코드 수정 불필요.
- **`training_setup_change`는 xyz/f_dc/f_rest/opacity/scaling/rotation 전부
  옵티마이저 등록** (`scene/gaussian_model.py:264-271`) + fusion 중 densify
  clone/split. → native fusion에서 기하는 고정이 아님. "선형·캐시"는
  **batch-static(선택이 fusion보다 먼저 끝나는 경우)에서만 정확**.
- rasterizer = `submodules/diff-gaussian-rasterization_fastgs`. 함정:
  (i) 아무것도 렌더 안 된 뷰의 backward가 CUDA 컨텍스트 오염 → **radii 가드
  필수** (Hutchinson backward 직격), (ii) `max_sh_degree=0` + 빈 rest 텐서면
  dc-grad가 조용히 0 → **toy fixture/probe 모델에 더미 degree-1 rest 필요**
  (본 모델은 rest 로드되어 안전), (iii) XFeat CUDA-graph는 씬별 서브프로세스 격리.
- cue = SAM2 기반 `generate_candidate_map` (논문의 1.69ms cue 아님). runtime
  주장은 레포 실측으로.
- alpha/visibility: `visibility_filter`는 **인덱스**지 bool mask 아님 —
  `radii > 0` 사용.

### 1.3 재사용 자산

| 자산 | 용도 |
|---|---|
| `subset_oscd.py` (2-phase 러너: pose 전체 → 선택 처리) | `--frames_method` 추가로 통합. 신규 러너 금지 |
| `target_nbv/change/counts.py` (dc-adjoint responsibility) | Hutchinson VJP의 출발점 (r_vpi = dc gradient) |
| `tests/conftest.py` toy fixtures (single_pancake 등) | brute-force 레퍼런스 테스트 |
| `experiments/frame_feature_analysis.py`의 cue 3D-합의 계산 | consensus weight A의 프로토타입 |
| oracle 지도 CSV·uniform 곡선·all25_repeats | 비교선·갭 회수율 분모. 재측정 금지(같은 배치 조건 제외) |
| `experiments/paslcd_nbv_sweep.sh` (resumable) | sweep 러너 템플릿 |

---

## 2. 수학 정의

### 2.1 측정 모델과 Jacobian

raw change logit: `z_v(p) = Σ_i r_vpi · c_i` (기하·opacity 고정 시 c에 선형,
r_vpi = alpha-compositing responsibility). 채널 projection(D4) 후:

```
J_v[p,i] = ∂z_v(p)/∂c_i = r_vpi                        (raw, MVP 기본)
∂σ(z_v(p))/∂c_i = σ′(z_v(p)) · r_vpi                    (sigmoid, ablation)
```

### 2.2 후보 정보량 (weighted diagonal)

```
b_v,i = Σ_p w_v(p) · (∂y_v(p)/∂c_i)²        y_v = z_v (기본) 또는 σ(z_v)
```

### 2.3 Criterion

```
G_D(v|S) = Σ_i log(1 + b_v,i / h_S,i)                    # 메인
G_T(v|S) = Σ_i [1/h_S,i − 1/(h_S,i + b_v,i)]             # trace reduction
G_F(v|S) = Σ_i b_v,i / h_S,i                             # FisherRF-style
G_cand(v) = Σ_i b_v,i                                    # 누적 OFF 대조군
```

### 2.4 Prior H (D6)

```
batch-static:   h_S,i = λ + Σ_{s∈S} b_s,i                # b는 씬당 1회 캐시 (정확)
sequential류:   h_S,i(θ_t) = λ + Σ_{s∈S} b_s,i(θ_t)      # 매 라운드 현재 모델에서 S 전체 재계산
                                                          # (|S| ≤ K−1 렌더 — 후보 스코어링보다 저렴)
```

### 2.5 λ (relative-median)

```
λ = max(λ_abs, λ_rel · median{b_v,i > 0})     기본 λ_rel=1e-3, λ_abs=1e-8
```
계산된 λ, positive-b median/max/sparsity를 manifest에 로깅. ablation
λ_rel ∈ {1e-4, 1e-3, 1e-2}.

### 2.6 Weight 모드 사다리

공통: `V_v(p) = 1[A_ref,v(p) ≥ τ_α]` (O-SCD unseen-region alpha 필터 재사용).

| 모드 | w_v(p) | 사용 정보 | 명명 |
|---|---|---|---|
| `pose` | `V_v(p)` | pose + 현재 모델만 | **NBV** |
| `current_map` | `V_v(p)·[ε + (1−ε)·stopgrad(m_v(p))]` | + 자신의 R_change 상태 (후보 content 불사용, 1라운드는 pose로 fallback) | **NBV (change-aware)** |
| `cue` | `V_v(p)·[ε + (1−ε)·C_v^norm(p)^γ]` | + 후보 cue content (detach) | pool-based selection |
| `consensus` | `V_v(p)·[ε + C_v(p)^γ · A_v(p)^β]` | + cue **3D-합의도 A** | pool-based selection |

기본 ε=0.1, γ=β=1. **경고(데이터 근거)**: cue 크기 단독 가중(`cue` 모드)은
독성 프레임(큰 cue·낮은 합의)을 선호할 위험 — cue 면적은 한계기여와 음의 상관.
`consensus`의 A(합의도, +0.173 신호)가 데이터가 지지하는 형태다. `cue` 모드는
이 가설의 대조군으로서 유지한다.

### 2.7 diag(JᵀWJ) 추정

1. **Hutchinson VJP (기본)**: ξ ~ Rademacher(픽셀별, deterministic seed =
   (camera.uid, probe_idx)), `g = ∂⟨y·√w, ξ⟩/∂c` backward → `b ≈ mean(g²)`.
   채널 합산 projection 적용. probe **M=4 시작**, rank 불안정 시 8.
   가능하면 `torch.func.vjp`/`is_grads_batched`로 배치, 아니면 루프.
2. **brute-force 레퍼런스**: toy fixture(≤16×16, 가우시안 수십 개)에서 픽셀별
   gradient로 exact b 계산 → Hutchinson 1/2/4/8 probe와 diag·순위·greedy 순서
   비교. **영구 보존되는 correctness oracle** (target_nbv의 FD oracle과 같은 지위).
3. **CUDA exact accumulator (`Σ w·q²`, q=α′T)**: Hutchinson이 전체 병목일 때만.
   float32 누적, tiny test에서 brute-force 일치, 미사용 시 기존 렌더러 불변.

---

## 3. 선택 프로토콜 (3모드)

모든 모드 공통: Phase A(전 pool pose 등록, `subset_oscd.py` 기존 경로) → 선택 →
**D7: 선택된 셋을 표준 파이프라인으로 시간순 재처리** → D8 평가.

### M1. Batch-static (MVP 메인)

```
b[i] ← 캐시 (ref 초기 상태에서 씬당 1회; 기하 불변이므로 정확)
h ← λ;  S ← []
repeat K:  i* = argmax_i G_D(i|S)  (동점 시 작은 frame ID);  S += i*;  h += b[i*]
```
선택이 fusion과 완전히 분리 → 스코어러 검증이 fusion 카오스 노이즈에서 자유.
결정성: 같은 config → 같은 선택 (candidate 순서 셔플 불변 테스트).

### M2. Sequential-frozen (current_map/sigmoid 전용)

선택 루프 동안 **기하 freeze + densify OFF, c만 업데이트** (dc 외 param group
lr=0 + densify 스킵 플래그). 매 라운드 current-map weight(또는 σ′) 재계산.
**주의: raw+pose 조합은 M1과 수학적으로 동일하므로 그 셀은 돌리지 않는다.**

### M3. Native sequential (현실 세팅)

pick → 표준 fusion 16 iter(densify 포함) → **H·b 전부 현재 모델에서 재계산(D6)**
→ 다음 pick. PoP-GS의 sequential NBV 프로토콜에 충실한 버전.

실험 순서: **M1 → (Gate S2 통과 후) M2 → M3.**

---

## 4. 검증 게이트

### Gate S1 — 추정기 정합 (toy fixture, CPU/GPU 테스트)

- brute-force vs Hutchinson: diag 상대오차, 후보 순위(Spearman), greedy 순서 일치.
- 모든 b ≥ 0, score NaN/Inf 없음.
- **중복 뷰 감쇠**: 같은 b를 두 번 넣을 때 `G_D(b|h+b) < G_D(b|h)`.
- selector 단위 테스트: budget 준수, 중복 없음, deterministic tie-break,
  후보 셔플 불변, budget 0/초과 처리, invalid pose 제외.
- no-leakage: `pose` 모드 실행 시 이미지 로더 mock이 접근을 감지하면 실패.
- baseline regression: `--frames_method all`이 기존 출력과 동일.
- 캐시: 동일 config hit / pose·해상도·checkpoint·weight-mode 변경 시 miss /
  손상 캐시 자동 무효화.

### Gate S2 — 스코어 타당성 (조건부 oracle marginal 상관; **통과 전 전체 sweep 금지**)

2–3씬(Garden + Zen + 1개; Zen은 근접 편향 감시용 필수 포함)에서:

1. 현재 선택 셋 S(공집합 포함 2–3개 스텝)마다, 각 잔여 후보 v에 대해 동일
   checkpoint에서 S∪{v}를 짧은 고정 iter로 학습 → held-out mIoU 측정 →
   `Δ_oracle(v) = mIoU(S∪{v}) − mIoU(S)`.
2. 모든 스코어 변형(cue_area, G_cand, G_F, G_T, G_D×{pose, current_map, cue,
   consensus})과 Spearman ρ 비교. **K=1 per-frame mIoU 지도**(씬당 25런)를
   0-맥락 상관의 보조 데이터로 병행.
3. **G_D 계열이 oracle marginal과 양의 순위 상관을 보이지 않으면 전체 dataset
   실험으로 넘어가지 않는다.** 수정 순서: raw→sigmoid → pose→current_map →
   cue → consensus → probe 증가/exact accumulator.

### 노이즈 프로토콜 (모든 성능 비교에 강제)

- **같은 배치 안에서만 비교** (cuDNN benchmark 배치 간 시프트 ±0.04).
  all-25 기준선도 상수가 아니라 같은 배치 재실행값.
- uniform@K = **5-offset 평균**, random@K = seed {0,1,2} mean±std.
- 씬 단일 런 차이는 ±0.05~0.1 노이즈 취급. **판정은 10씬(최종 20 instance)
  집계로만.**
- 1차 지표: **oracle 갭 회수율** `(method − uniform_best) / (oracle − uniform_best)`
  (rev3 지도 분모) + **K≤3 극저예산 구간**. K=5 이분 판정("+2 mIoU" 류)은
  노이즈 경계라 게이트로 쓰지 않는다.

---

## 5. 실험 매트릭스

### 5.1 Budget과 베이스라인

K ∈ {2, 3, 5, 10}. (K≤3이 주 전장, K=5는 oracle 갭 회수율용, K=10은 포화 확인)

| # | method | 비고 |
|---|---|---|
| 1 | all-25 | 같은 배치 재실행 |
| 2 | random@K | 3 seeds |
| 3 | uniform@K | 5-offset 평균 |
| 4 | pose-FPS | 위치+시선방향 farthest point |
| 5 | max alpha coverage | |
| 6 | max cue area | 독성 가설 대조군 (나쁠 것으로 예측) |
| 7 | G_cand (누적 OFF) | prior 누적 효과 분리 |
| 8 | G_F, G_T | criterion 비교 |
| 9 | **G_D × pose (M1)** | 메인 NBV |
| 10 | **G_D × current_map (M2)** | change-aware NBV |
| 11 | G_D × cue / consensus | pool-based |
| 12 | (참고) 기존 nbv_dopt proxy | exact vs proxy Fisher 가치 입증 + **Zen 회복 여부로 1/d² 서사 완결** |

### 5.2 진단 (성능 표와 동급으로 저장)

- **선택 감사**: 뽑힌 뷰들의 촬영 거리 분포, cue 면적, **변화 영역 재관측
  시차 구조(ray-겹침)** — uniform 대비. 근접 편향·커버리지 편향·독성 선호를
  직접 측정.
- seen(선택)/unseen(비선택) pose 분리 mIoU (같은 25-query 렌더의 사후 분할).
- lighting split: consistent vs varied instance 별도 리포트.
- greedy step별 marginal G_D 감소 곡선, prior 분포/coverage 변화.
- Zen·Porch watchlist: 씬별 이전 실패 패턴 재발 여부.

### 5.3 Ablation (S2 통과 후, 5씬 K=5 중심)

A1 criterion (D/T/F/cand) · A2 output space (raw vs sigmoid) · A3 weight 4종 ·
A4 λ_rel · A5 probes {1,2,4,8}(+exact) · A6 M1 vs M2 vs M3 ·
A7 c-projection 유무 (순위 불변 예측의 실증 — sanity check).

### 5.4 알려진 구조적 한계 (리스크 표)

| 리스크 | 감지 | 대응 |
|---|---|---|
| diagonal이 ray-방향 교차 상관(시차가 푸는 것)을 버림 — c는 opacity처럼 합성 관측이라 POp-GS의 opacity 실패 기제와 동형 | 선택 감사에서 시차 구조 부재, S2 ρ 낮음 | 한계로 명시. 후속: ray-겹침 페널티/블록 확장 (본 플랜 비범위) |
| pose-only는 독성 프레임을 원리상 회피 불가 | 성능이 consensus 모드에서만 개선 | 서사를 "기하 정보는 필요조건" 으로 조정, consensus가 본 결론 |
| Hutchinson 노이즈 | M=4 vs 8 rank 불일치 | probe 증가 → exact accumulator |
| σ′ saturation (sigmoid ablation) | 후반 라운드 score 분산 급감 | raw 기본 유지 근거로 기록 |
| pose 실패 후보 | PnP inlier 부족 | 후보 제외 + 각주 (O-SCD 동일 가정) |
| ref 미관측 신규물체 영역 | alpha coverage 낮음 | V 필터로 일관 제외, 한계 명시 |

---

## 6. 구현 계획

### 6.1 모듈 배치

```
view_selection/
    __init__.py
    types.py          # ViewCandidate, ViewInformation, SelectionStep (Codex §8)
    information.py    # estimate_change_information (Hutchinson, 채널 projection, radii 가드)
    criteria.py       # G_D / G_T / G_F / G_cand
    weights.py        # pose / current_map / cue / consensus
    greedy.py         # batch-static + sequential (H 재계산), deterministic tie-break
    cache.py          # 캐시 키: checkpoint hash, pose/intrinsics, 해상도, τ_α,
                      # output space, weight mode, probes+seed, cue 전처리 버전
```

`subset_oscd.py`에 `--frames_method {dopt_pose, dopt_cmap, dopt_cue, dopt_cons}`
+ `--select_mode {batch, seq_frozen, seq_native}` 추가. 기존 method 불변.

### 6.2 산출물

```
outputs/view_selection/<scene>/<instance>/
    information/frame_*.pt
    selections/<method>_k{K}.json     # manifest: 후보 ID, 선택 순서, step별 score,
                                      # λ·b 통계, alpha coverage, seed, commit hash, config
    scores/<method>_steps.csv
docs/changenbv_code_map.md            # audit 문서화 (§1.2 선답변 + 잔여 확인)
tests/test_change_information_{exact,hutchinson}.py
tests/test_greedy_selection.py, test_no_leakage.py, test_selection_cache.py
results/changenbv/*.csv               # 스키마: scene, instance, method, K, seed,
                                      # mIoU, F1, runtime_select, runtime_fusion, runtime_total
```

### 6.3 커밋 순서 (검증 최단 경로)

1. `docs: ChangeNBV merged plan` (이 문서)
2. `docs: code map` — §1.2 선답변 문서화 + 잔여 확인(pose confidence 저장 여부 등)
3. `feat: view_selection types + brute-force exact reference + toy fixture`
4. `feat: Hutchinson estimator (채널 projection, radii guard)` + Gate S1 테스트
5. `feat: criteria + greedy (batch-static) + manifests` + selector 테스트
6. `feat: subset_oscd integration (dopt_pose, batch)` + no-leakage/regression 테스트
7. `exp: Gate S2 oracle-marginal correlation runner (2–3씬)` → **S2 판정 보고**
8. (S2 통과 시) `exp: K-sweep 러너 + 베이스라인 매트릭스` (5씬 → 10씬)
9. `feat: current_map weight + sequential-frozen`
10. `feat: cue/consensus weight (frame_features 합의도 재사용)`
11. `feat: native sequential (H 재계산)` 
12. `perf: batched VJP / CUDA exact accumulator` — 병목일 때만

### 6.4 Definition of Done

**코드**: 기존 all 경로 보존(regression 통과) · dopt_pose end-to-end ·
선택 셋만 fusion 투입 · manifest/score 로깅 · 전 테스트 통과 · deterministic 재현 ·
GT 미사용(no-leakage 통과).

**연구 판정 (성능 없어도 구현 완료로 간주; 후속 진행은 S2 상관 확인 시만)**:
1. G_D가 G_cand/G_F보다 다양한 pose를 고르는가 (선택 감사)
2. greedy step별 marginal G_D가 단조 감소하는가
3. G_D score가 oracle marginal mIoU와 양의 순위 상관 (Gate S2)
4. dopt_pose가 random·pose-FPS 대비 10씬 집계에서 개선 + oracle 갭 회수율 > 0
5. current_map/consensus가 pose 대비 추가 이득 (특히 K≤3, varied lighting)
6. Zen: proxy 대비 회복 여부 → 1/d² 근접 편향 서사 완결
