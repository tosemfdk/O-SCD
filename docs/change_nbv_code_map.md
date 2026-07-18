# ChangeNBV Code Map (Phase 0 확정 사실, 2026-07-18)

스펙 §18의 audit 선답변. 전부 이 세션에서 소스 확인 완료 — 재검증 불필요.

## 확정 사실

| 항목 | 위치 / 내용 |
|---|---|
| change 파라미터 | `_features_dc`, `load_ply_change`가 **(N,3,1) zeros로 초기화** (`scene/gaussian_model.py:427`). `_features_rest`는 ply에서 로드 (max_sh_degree=3 → (N,3,15), 비어있지 않음 → dc-grad 안전) |
| raw z 경로 | `render_change` (`gaussian_renderer/__init__.py:114-207`)가 raw 컴포짓 반환, **[0,1] 이미지 클램프**(:202) 존재. per-Gaussian 색 = 0.5 + SH_C0·c, CUDA에서 min-0 클램프. **c=0 스코어링 상태에서는 두 클램프 모두 비활성** → ∂z/∂c = SH_C0·r_vpi 정확 |
| sigmoid 위치 | loss에서만: `subset_oscd.py:291` `sigmoid(change_mask.mean(dim=0))` |
| full optimizer | `training_setup_change`가 xyz/dc/rest/opacity/scaling/rotation 전부 등록 (`gaussian_model.py:264-271`) + fusion 중 iteration 4에 densify clone/split (`subset_oscd.py:312-322`) |
| dc-adjoint | `target_nbv/change/counts.py` — 검증된 항등식 e1_g = Σ_p r_gp·M_p (렌더 1회+backward 1회), radii 가드, 더미 rest 프로브 |
| 25-query 평가 | `subset_oscd.py:398-405` — 최종 R_change를 all_views 전체 pose에서 렌더 → `renders/query_mask/` → `utils/evaluate.py` |
| 2-phase 러너 | `subset_oscd.py`: Phase A 전 프레임 pose 등록(:226-248) → 선택 → `process_view` 16-iter fusion(:267-333). manual/uniform/random은 시간순 처리(:388-393) = **clean replay 그 자체** |
| XFeat 격리 | 씬별 서브프로세스 필수 (CUDA-graph 캡처 충돌) |
| SAM2 cue | `generate_candidate_map` (`oscd.py`), `torch.compile(max-autotune)`은 이 서버(torch 2.5.1)에서 크래시 → 러너는 `TORCHDYNAMO_DISABLE=1` |
| 동결 기준선 | `baseline-freeze-phase0`, oracle 지도 `experiments/oracle_search_results*.csv` (~10.2k 세트), uniform 곡선, `all25_repeats*.csv` |

## 이 세션에서 실측으로 새로 확정

- **empty-rest 무음 0-gradient 함정 실재**: `_features_rest`가 (N,0,3)이면
  `render_change` backward의 dc-grad가 소리없이 전부 0. `ZeroAdjointError`
  strict 가드가 검출 (`tests/test_change_information_exact.py::test_zero_adjoint_trap_detected`).
- **Hutchinson 분산 실측** (7-gaussian 토이, 중첩 큼): 정규화 L1 오차
  M=256에서 3–8%, M=1024에서 2–5%, 1/√M 수렴. 스펙 §6.4의 "256@5%"는
  이 토이 분산보다 빡빡 → 게이트를 "수렴 + 1024@5%"로 수정 (S1 리포트 기록).
  M=4 candidate rank는 ρ≥0.9 + top-1 일치 통과.
- alpha map: `render(override_color=ones)` = per-pixel Σr (accumulated alpha).
  별도 alpha 출력 키는 rasterizer에 없음.
- `visibility_filter`는 인덱스(`(radii>0).nonzero()`) — bool mask 아님 (기존 인지 재확인).

## 잔여 확인 (해당 Phase에서)

- consensus A: `frame_feature_analysis.py`의 `cue_consistency`는 **frame-level
  스칼라** (per-Gaussian e1 lift의 cosine) — pixel-level 아님. consensus weight
  구현 시(Phase D) `agreement_granularity="frame"`으로 기록하고 broadcast.
- pose confidence: PnP inlier 수가 아티팩트로 저장되는지 미확인 — S2에는 불필요.
- topology revision hook: M2/M3 구현 시(Phase D) `gaussians_change.get_xyz.shape[0]`
  변화 감지로 충분 (densify는 fusion iteration 4에서만 발생).
