# O-SCD Project Notes

프로젝트 컨텍스트(연구 계획·규칙)는 `OSCD_Budgeted_View_Project_Context.md`를 먼저 읽을 것.
Phase 0 베이스라인은 동결됨: 태그 `baseline-freeze-phase0`, 산출물 `artifacts/baseline/`.

## 실행 환경 (2026-07-14 구축, 검증 완료)

모든 실행은 conda env **`oscd`** 에서:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oscd
```

- Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8 툴킷은 **시스템이 아니라 env 안에** 있음
  (`nvcc` = `$CONDA_PREFIX/bin/nvcc`). CUDA 확장을 빌드할 때는:
  ```bash
  export CUDA_HOME=$CONDA_PREFIX
  export PATH=$CUDA_HOME/bin:$PATH
  export TORCH_CUDA_ARCH_LIST="8.6"   # RTX A6000
  ```
- 정확한 패키지 버전: `artifacts/baseline/pip_freeze.txt`
- GPU: RTX A6000 48GB (driver 580.142), Ubuntu 22.04

## 환경 재구축 시 함정 (README대로만 하면 실패함)

1. conda-forge로 env를 만들면 **pip이 없을 수 있음** → env에 pip 설치 후 반드시
   `$CONDA_PREFIX/bin/python -m pip`로 설치할 것. 맨 `pip`은 시스템 pip(python3.10)으로
   떨어져 `~/.local`을 오염시킴.
2. CUDA 확장 3종(`submodules/{diff-gaussian-rasterization_fastgs,fused-ssim,simple-knn}`)은
   setup.py가 torch를 import하므로 **`pip install --no-build-isolation`** 필요.
3. XFeat의 torch.hub 대화형 신뢰 프롬프트: `~/.cache/torch/hub/trusted_list`에
   `verlab_accelerated_features` 한 줄을 미리 넣어둘 것 (이미 등록됨).
4. 체크포인트는 첫 실행 때 자동 다운로드: XFeat(torch.hub), SAM2 `facebook/sam2.1-hiera-tiny`(HF).

## 베이스라인 실행/평가

```bash
bash run_oscd.sh          # 20개 인스턴스 전체 (~10분, warm compile cache 기준)
# 단일 씬:
python oscd.py -s data/PASLCD/Instance_1/Garden/ -m output/Instance_1/Garden/ --resolution 4 --test_hold 5 --refine
python utils/evaluate.py --gt data/PASLCD/Instance_1/Garden/gt_mask/ --pred_binary output/Instance_1/Garden/renders/change_mask/
```

동결된 기준 수치(20-인스턴스 평균): 온라인 mIoU 0.4887 / F1 0.6423, 정제 0.5573 / 0.7009.
씬 단위 run-to-run 노이즈 ±0.005 mIoU (torch.compile max-autotune 영향) — 셀렉터 비교는
20-인스턴스 집계로 할 것. 상세: `artifacts/baseline/discrepancy_report.md`.

## 주의

- 베이스라인 수치를 재생성할 때 `oscd.py`의 설정(16 fusion 반복, resolution 4 등)을 절대 바꾸지 말 것.
- `--test_hold 5`는 플래그만 세팅하고 실제로는 25프레임 전부 `R_change`를 업데이트함 —
  Phase 1 all-query-view 평가는 자체 홀드아웃 구현 필요.
- 데이터셋 `data/PASLCD/`(15GB)와 `output/`은 git에 추적하지 않음.

## 실험 결과 공유 규칙 (필수, 2026-07-18 사용자 지시)

**실험을 돌렸으면 그 결과를 반드시 커밋하고 `develop-claude` 브랜치에 push할 것.**
웹 세션·다른 머신·다른 에이전트가 이 브랜치를 pull 해서 맥락을 이어받으므로,
push 안 된 실험은 없는 실험이다.

- 커밋 대상: 결과 CSV(`experiments/*.csv`), 실험 리포트(`experiments/*.md`),
  종합 문서(`docs/budgeted_view_findings.md`, `docs/experiment_results_tables.md`),
  실험 드라이버·그림 스크립트 변경분.
- untracked 유지: 실행 로그(`*.log`), PNG 그림(스크립트로 재생성),
  `output_subset/`, 데이터셋.
- 실험 하나가 끝날 때마다 위 문서들을 갱신 → 커밋 → push가 한 사이클.
