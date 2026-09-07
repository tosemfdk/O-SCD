# Compact lifespan + joint-channel 학습 renderer: develop 통합

## 적용 범위

2026-09-07에 별도 실험 worktree의 두 기능을 **메인 `/home/rvl/workspace/github/O-SCD-evolving`의 `develop` 작업 트리**에 반영했다.

- 원본: `/home/rvl/workspace/github/O-SCD-evolving-joint-render-ablation`
- Compact snapshot 원본 commit: `7313663409128161e40bcadc3b7993e48c82e220`
- Joint-channel renderer: 해당 worktree의 미커밋 변경. 가져온 파일 SHA256은 [통합 provenance](../outputs/main_compact_joint_integration_20260907/before.json)에 기록했다.
- 메인에는 기존 미커밋/미추적 실험 파일이 많아 전체 branch merge/cherry-pick을 하지 않았다. 필요한 파일과 viewer의 해당 hunk만 적용했다.
- **메인 HEAD/index는 변경하지 않았으며 별도 commit을 만들지 않았다.** `develop` 작업 트리에 기능이 들어온 상태다.
- 원본 worktree와 그 실험 산출물은 수정하지 않았다.

## 1. Compact lifespan의 의미

`temporal/lifespan_state_snapshots.py`에 timestamp별 Gaussian 상태 snapshot을 저장한다.

```text
0 = unborn
1 = NEVER_OPEN
2 = OPEN
3 = CLOSED
```

각 timestamp의 Gaussian **인덱스 순서**에서 같은 상태가 연속되는 구간은 RLE로 압축하고, 그렇지 않으면 Gaussian당 2-bit packing을 사용한다. 두 payload 중 더 작은 것을 선택한다.

이는 **시간축의 lifespan/run length를 새 detector로 표현하는 것이 아니다.** 기존 interval 이력은 유지하며, 이미 처리한 frame의 상태를 재사용하여 반복적인 interval 탐색을 줄이는 최적화다.

- CPU의 compact frame snapshot + 최대 16 timestamp의 decoded device-mask LRU.
- Gaussian ID는 고정하고 archive row는 append-only. 뒤에 추가된 row는 과거 snapshot에서 unborn으로 padding한다.
- 공개 lifecycle mask getter는 사본을 반환해 외부 in-place 수정이 cache를 오염시키지 않게 한다.
- Append/OPEN/CLOSE 시 해당 timestamp 이후 snapshot과 decoded mask를 무효화한다.
- 같은 frame의 학습/densification/pruning이 모두 끝난 뒤 base·seed snapshot을 seal한다.
- Seal되지 않은 과거 diagnostic timestamp는 기존 interval 경로로 정확하게 fallback한다.
- Learned DC/geometry/visibility/optimizer parameter를 cache하는 것이 아니다.

## 2. 학습 GS scene 통합의 의미

`experiments/panel10_split_training.py`의 `joint_partition_colors()`와 `train_partition_update()`에 opt-in joint-channel 경로를 통합했다.

```text
하나의 scene = base OPEN + seed OPEN + 모든 NEVER_OPEN black occluder

R 채널 = seed/NEW radiance
G 채널 = base/나머지 radiance
B 채널 = 0

L = SSF(2 × Q_NEW, R채널) + SSF(2 × (Q − Q_NEW), G채널)
```

한 update에서 scene 조립/학습 render/backward를 각각 한 번 수행한다. 각 채널은 기존 SSF 인터페이스에 맞춰 3채널로 expand하지만, 별도의 scene 또는 render를 만드는 것은 아니다. Sigmoid 위치와 두 target/regularizer는 유지한다.

- Base/seed parameter bank, optimizer, topology 소속은 별도로 유지한다. 통합 대상은 **학습 render scene**이다.
- NEW DC는 NEW 채널, base DC는 BASE 채널에 연결한다.
- **투과율과 가림은 공유한다.** Seed geometry/opacity는 BASE loss에서도 occlusion gradient를 받을 수 있다. 따라서 기존 split-render와 수학적으로 동등한 최적화로 해석하면 안 된다.
- Seed density의 screen-space gradient도 두 loss를 합친 결과다. Threshold/budget/시점은 같아도 이후 topology와 lifecycle은 달라질 수 있다.
- NEVER_OPEN은 frozen black occluder로 한 번 포함하고 CLOSED/future는 제외한다. Historical-only OPEN은 detached, sampled/current 모두 OPEN이고 visible인 row만 갱신한다.
- 최종 mask의 기존 joint renderer와 pre-optimization raw-cue single detector는 변경하지 않았다.

## 활성화

기존 viewer 실행 인자에 다음을 추가한다.

```bash
--training-partition panel10_new \
--lifespan-state-cache snapshot \
--panel10-render-mode joint_channels
```

**기본값은 여전히 `--lifespan-state-cache off --panel10-render-mode split`이다.** Joint render는 `panel10_new`가 아닌 partition과 조합하면 CLI에서 거부한다.

이전 작업의 3D+2D root-birth gate도 그대로 남아 있으므로 아래 옵션과 함께 사용할 수 있다.

```bash
--da3-birth-coverage 3d_plus_2d --da3-coverage-2d-sigma 2
```

이미 실행 중인 live viewer를 재시작하거나 기본 실행 설정을 바꾸지는 않았다.

## 통합 및 보존 확인

- 기존 `panel10_split_training.py`와 해당 test가 snapshot commit의 preimage와 같은 것을 확인한 뒤 joint 변경을 가져왔다.
- Viewer는 원본 전체 파일로 덮어쓰지 않고 **snapshot 42행 + joint CLI 6행, 합계 48행 추가**만 적용했다.
- 메인에만 있던 3D+2D occupancy, 차단 카운터, 영상 패널 재배치, 다른 미커밋 변경을 보존했다.
- 수정 전 상태가 존재하던 파일 중 변경된 것은 viewer/partition training/partition test 3개뿐이다. 다른 기존 dirty 파일은 SHA256이 동일하다.
- Snapshot module, partition module, benchmark, 원본 test 3개는 source worktree와 byte-identical이다.
- 새 외부 dependency는 추가하지 않았다. 기존 상태 조회 API와 render adapter를 재사용했다.

### 반영 파일

- `temporal/lifespan_state_snapshots.py`
- `experiments/view_bayesian_detector_steps.py`
- `experiments/panel10_split_training.py`
- `experiments/benchmark_lifespan_state_snapshots.py`
- `tests/temporal/test_lifespan_state_snapshots.py`
- `tests/experiments/test_lifespan_state_snapshot_regression.py`
- `tests/experiments/test_panel10_split_training.py`
- `tests/experiments/test_snapshot_joint_integration.py` — 메인에 추가한 조합 회귀.

## Fresh 검증

1. 변경 전 기존 189개 회귀 테스트 통과.
2. 원본의 새 snapshot/joint CLI test가 기존 main에서 실패하는 RED gate 확인 후 구현 적용.
3. 최종 관련 회귀 **253 tests passed** — actual CUDA codec/mask, renderer forward/backward, 두 DC 갱신, geometry gradient, history navigation, seed topology, occupancy, 영상 구성 포함.
4. 새 조합 회귀에서 cache off/on의 joint-channel current/replay loss·gradient·parameter·optimizer step 상태와 3D+2D seed 선택/차단 count가 정확히 일치했다.
5. AST/whitespace 및 independent code review 통과, blocking issue 없음. ruff/mypy/LSP는 설치되지 않아 scoped pytest/AST/static review로 대체했다. 전체 repo test suite는 실행하지 않았다.

### 메인 worktree의 실제 CUDA 실행

| 검증 | Cantina audit | ESCD 조합 smoke |
|---|---|---|
| Frames / updates | 25 / u120 | 8 / u120 |
| Snapshot / joint_channels | 켬 / 켬 | 켬 / 켬 |
| Birth coverage | 기존 3D-only | 3D+2D |
| 핵심 확인 | 기존 interval과 1,300건 bitwise 일치 | seed birth 후 단일 joint evidence, 실제 base·seed DC 갱신 |
| 추가 확인 | 3,000 updates에 학습 render 3,000회; bank별 snapshot 25개; normal interval fallback 0 | root birth 339, density child 12, prune 20; 기존2D 차단 450 + 동일 frame 예약 차단 415 |
| 저장 mask 재계산 | 25 frames 정확히 일치 | 8 frames 정확히 일치 |

ESCD smoke의 reference/probe/frozen parameter/frozen Adam drift, future view access, lifespan render violation, binary evidence mixed-row violation은 모두 0이었다. Cantina 원본 runner의 frozen/reference/causal audit도 통과했다.

**이번 검증은 통합 동작 확인이다.** Cantina는 BF30 soft, ESCD는 first-OPEN BF10 + Gaussian 합산 후 binary라는 서로 다른 기존 preset을 사용했다. 이 수치를 서로 비교하거나, 전체 ESCD 304/PASLCD 500-frame 정확도 또는 새로운 속도 향상 결과로 해석하지 않는다. 앞서 3D+2D birth 실험의 full304 수치를 이번 joint renderer 결과로 대체하지 않았다.

## 재현 및 산출물

[통합 검증 root](../outputs/main_compact_joint_integration_20260907/)에 preimage 백업, source hashes, 두 viewer patch, 회귀 로그, CUDA audit/캡처, 최종 검증을 남겼다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=4 \
 /home/rvl/miniforge3/envs/oscd/bin/python \
 -m experiments.benchmark_lifespan_state_snapshots \
 --cache snapshot --render-mode joint_channels --updates 120 \
 --audit --no-profile-components --port 8371 \
 --output-root outputs/main_compact_joint_integration_20260907/fresh_cantina
```

- `tests_before.txt`, `red_snapshot.txt`, `red_joint.txt`, `tests_final.txt`: 테스트 근거.
- `cantina_snapshot_joint_u120/timing.json`: 1,300-query audit, render count, cache 통계.
- `escd_3d2d_snapshot_joint_u120/{configuration.json,audit.json,summary.json,posthoc_verification.json}`: 새 옵션 조합의 실제 실행 근거.
- `cuda_verification.json`, `verification.json`: 통합 확인 요약.
- `before/`, `before.json`, `snapshot_viewer.patch`, `joint_viewer.patch`, `integrated_viewer.diff`: 기존 작업 보존과 적용 범위 추적.

원래 성능 실험 자료는 별도 worktree의 `docs/lifespan-state-snapshot-speed-20260907-ko.md`, `docs/joint-panel10-render-ablation-20260907-ko.md`, `outputs/joint_panel10_render_20260907/`에 그대로 있다.
