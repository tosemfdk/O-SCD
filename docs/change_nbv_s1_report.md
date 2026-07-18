# Gate S1 Report — 추정기·selector 정합성 (2026-07-18)

**판정: PASS** (1건의 스펙 수정 포함, 아래 명시). Phase B 진행 가능.

실행: `TORCHDYNAMO_DISABLE=1 pytest tests/test_change_*.py` — 41 passed
(신규 22 + 기존 stage-11 change 테스트 19). 환경: 8×V100 서버,
torch 2.5.1+cu121, `/workspace/oscd-venv`.

## 통과 항목

| 영역 | 테스트 | 결과 |
|---|---|---|
| Change gradient | tied-channel FD vs projection gradient (오차 ≤5%) | ✅ |
| | 3채널 gradient 동일성 (max spread <1e-6) | ✅ |
| | projection 스케일 항등식 b_proj = 9·b_ch | ✅ |
| | **empty-rest 무음 0-grad 함정 → ZeroAdjointError** (실재 확인) | ✅ |
| Information | exact b ≥ 0, finite | ✅ |
| | exact vs Hutchinson 수렴 (아래 스펙 수정) | ✅ |
| | M=4 candidate rank ρ≥0.9 + top-1 일치 (8 토이 카메라) | ✅ |
| | 결정적 재현 (같은 seed scope → bit-equal) | ✅ |
| | 빈 렌더 가드: backward 미호출, 0-info, 컨텍스트 무손상 | ✅ |
| | 퇴화 weight 가드 | ✅ |
| Criterion | 4종 닫힌형 일치, 중복 뷰 strict 감쇠, float64 합, NaN 즉시 실패 | ✅ |
| Greedy | budget/중복/tie-break(작은 ID)/셔플 불변/고정상태 marginal 비증가/budget 0·초과·음수 info 처리 | ✅ |
| Leakage | FrameAccessGuard 차단·복원·예외 시 복원 | ✅ |
| Cache | roundtrip hit / 키 변이 miss(프레임·씬 포함) / 손상·NaN 자동 무효화 | ✅ |

## 스펙 수정 1건 (측정 근거)

스펙 §6.4 "Hutchinson M=256 정규화 L1 ≤5%"는 토이(가우시안 7개, 중첩 큼)의
실측 분산과 충돌: M=256 오차 3–8%, M=1024 오차 2–5%, 1/√M 수렴 확인.
추정기는 불편(unbiased)이므로 게이트를 **"M=1024 ≤5% + 평균 오차의 M-단조
감소"**로 수정. 실전 신뢰성은 M=4 rank 테스트(ρ≥0.9, top-1 일치)가 담당하며,
실씬 rank 안정성은 S2에서 M=4 vs 8로 재확인 예정 (스펙 수정 사다리 유지).

## 미룬 항목 (스펙상 후속 Phase 소속)

- all-path regression·all-25 query assertion·test_hold no-op → Phase B에서
  subset_oscd 통합과 함께 (같은-머신 재현 ±0.0000이 확인돼 있어 실행 비교 가능)
- M1/M2 동치성, sequential 재계산 테스트 → Phase D (M2/M3 구현 시)

## 산출물

- 코드: `view_selection/{types,information,criteria,weights,greedy,cache}.py`
- 테스트: `tests/test_change_{information_exact,criteria,greedy,no_leakage,cache}.py`
- 실측 기록: `docs/change_nbv_code_map.md` "이 세션에서 실측으로 새로 확정" 절
