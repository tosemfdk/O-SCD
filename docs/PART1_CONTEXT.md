# SCD-NBV Part 1 — 컨텍스트 요약 (2026-07-17)

이 문서 하나로 Part 1의 상태를 이어받을 수 있게 쓴 핸드오프. 발견의 상세 서사는
`docs/budgeted_view_findings.md`, 실험별 수치는 `experiments/*_report.md` 참조.

## 1. 한 줄 결론

**Feasibility 확인 — go.** 잘 고른 5장이 25장 전부를 이긴다 (Instance_1 10/10 씬,
평균 +10.3% mIoU). 같은 K=5에서 선택이 만드는 폭(평균 0.174)이 장수 효과(0.054)보다
크다. Part 2 = 검증된 부품들의 온라인 조립.
Notion 보고서: SCD-NBV Part 1 feasibility check (39fcbb7d793780e2b14fe222272c1f44).

## 2. 저장소 / 환경

- 작업 저장소: `~/workspace/github/O-SCD-claude`, 브랜치 `develop-claude`.
  push 대상: `origin`(로컬 O-SCD)과 `github`(tosemfdk/O-SCD) 둘 다.
  `~/workspace/github/O-SCD`는 사용자 열람용 — 건드리지 않는다.
- 실행: conda env `oscd` (python 3.12, torch 2.11+cu128, CUDA toolkit이 env 안에 있음).
  재구축 함정은 루트 `CLAUDE.md` 참조.
- 테스트: `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`
  (ROS humble이 PYTHONPATH를 오염시킴). 현재 **111개 통과**, GPU 테스트는 `-m gpu`.
- 베이스라인 동결: 태그 `baseline-freeze-phase0` (온라인 20-인스턴스 0.4887/0.6423).
  `oscd.py`는 동결 — 모든 변형은 `subset_oscd.py`에서.

## 3. 평가 프로토콜 (불변 원칙)

- 업데이트에 몇 장을 쓰든 **평가는 25개 query pose 전부**에서 렌더한
  `renders/query_mask/` vs GT 25장 (`utils/evaluate.py`).
- 노이즈 3층: (a) 배치 간 시프트 — cuDNN benchmark 탓, 같은 명령도 배치마다 ±0.04
  (같은 배치 안에서는 재현됨 → **비교는 같은 배치에서**), (b) 선택 민감도 ±0.05~0.11,
  (c) fusion 카오스 — 씬 단일 런 비교는 ±0.05~0.1 오차 취급. 다씬 집계만 신뢰.

## 4. 구축된 자산

### 러너 (`subset_oscd.py`) — 2단계 구조(pose 전체 → 선택 처리)
`--frames_method {all, uniform, random, nbv, nbv_dopt, manual}` + `--budget K`
- `manual --frames_list "이름,..."` 또는 인덱스 — oracle/수동 실험용
- `nbv`: Beta-EIG 커버리지 (방향 무시 — uniform에 짐, 기록용)
- `nbv_dopt`: 방향 인지 D-opt (Porch 반전, 아직 uniform에 −0.024)

### target_nbv 패키지 (Gate A–F, 106+ 테스트)
persistent Gaussian ID/freeze → 후보 카메라 생성 → color-probe visibility →
FD Jacobian(oracle 백엔드) → 정보행렬 빌더 → proxy/exact D-opt 스코어러 →
`selector.py` + CLI(`tools/select_target_nbv.py`, `update_target_information.py`).
change 모드(Gate G): `target_nbv/change/` — Beta 상태, **렌더러 어드조인트로
per-Gaussian soft count**(핵심 트릭: dc-gradient = 합성 responsibility의 정확한 adjoint),
D-opt 프레임 상태.

### 실험 드라이버 (`experiments/`)
`paslcd_nbv_sweep.sh`(셀렉터 비교, resume 가능), `oracle_search.py`(uniform offset 기준),
`oracle_local_search.py`(hill-climb oracle 탐색), `frame_feature_analysis.py`(GT-free 특징),
`target_nbv_pool_eval.py`(target 수준 D-opt 검증), 분석 스크립트들.
**결과 CSV는 전부 tracked** (사용자 결정, 61e209b).

## 5. 핵심 수치 (Instance_1, 같은 배치)

| 항목 | 값 |
|---|---|
| oracle-5 vs all-25 | 10/10 씬 우위, 평균 +10.3% (+2.4~+25.3%) |
| uniform 포화 | K=5→94%, K=8→96%, 95% 도달 K≈7 (10씬 평균) |
| 같은 K=5 선택 폭 vs 장수 효과 | 0.174 vs 0.054 (10/10 씬에서 선택 > 장수) |
| target 수준 D-opt | 2장 = uniform 5장 정보량 (10/10 target) |
| 독성 프레임 | 씬당 −0.03~−0.11; 최악 유형 = 큰 cue + 낮은 3D 합의 (GT-free 검출 가능) |
| 사람 픽 (Lounge, K=5) | 0.545 — all-25 이김, uniform@5(0.581)엔 짐 → K=5는 노이즈 지배 구간 |

이기는 5장의 시각적 정체: **"변화 지점을, 다양한 시차로, 크게"** (몽타주: Notion 페이지).

## 6. 시스템 함정 (재발 주의)

1. fastgs: `max_sh_degree=0`(빈 rest 텐서)이면 **dc gradient가 조용히 0** — probe에 더미 degree-1 rest 필요 (`counts.py`).
2. fastgs: **아무것도 안 렌더된 view의 backward가 CUDA 컨텍스트를 오염** (size-0 커널) — radii 가드 필수.
3. XFeat Detector의 CUDA-graph 캡처는 **오염된 컨텍스트에서 실패** — 씬별 이미지 크기가 다르면 씬당 서브프로세스로 격리.
4. `--test_hold`는 플래그만 있고 실제 홀드아웃 아님 — subset_oscd가 자체 구현.
5. 렌더 산출물/데이터셋만 gitignore; CSV는 tracked.

## 7. Part 2 후보 (우선순위 제안)

1. **온라인 통합 (스펙 13단계)**: 순차 스트림에서 anchor 추구 + 독성 cue 가드(합의도 기반)
   + D-opt 시차 기준 + 중단 조건. uniform이 존재할 수 없는 세팅이 본 무대.
2. K≤3 극저예산에서 nbv_dopt 개선 (1/d² 근접 편향 수정이 첫 시도).
3. Instance_2 재현 + 노이즈 통제(같은 배치 반복) — 주장 일반화.
4. oracle 조합의 셋-조건부 특징 분석 (learning-to-rank 데이터는 이미 CSV에 있음).
5. 남은 스펙: 10단계 Schur (선택), 14단계 synthetic Test D.

## 8. 커밋 이정표

`ab3c7e6` Phase0 freeze → Gate A–F (target-NBV MVP) → `6464cb1` change 모드 →
`09270d9` nbv 통합 → `fd20e22`/`e6fb372` 셀렉터 음성결과+진단 → `b12a996` nbv_dopt →
`5342dbd` uniform 곡선+manual → `c04fc85` oracle-5 지도 → `cfd6b0d` 프레임 특징 →
`51272da` findings 문서 → `61e209b` CSV 전부 tracked.
