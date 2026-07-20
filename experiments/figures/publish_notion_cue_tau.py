# Append the tau=0.5 vs 0.9 case-montage section to the Notion write-up.
# Target: "SCD-NBV Part 3 - change feature quality difference".
# The key is read from <repo>/.notion_token (gitignored); NOTION_TOKEN wins if set.
#   python experiments/figures/publish_notion_cue_tau.py
import json
import os
import subprocess

S = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(S))
TOKEN = os.environ.get("NOTION_TOKEN") or \
    open(os.path.join(REPO, ".notion_token")).read().strip()
PAGE = "3a3cbb7d7937808aa48ef3185963afab"
H = ["-H", f"Authorization: Bearer {TOKEN}", "-H", "Notion-Version: 2022-06-28"]


def api(method, url, payload=None):
    cmd = ["curl", "-s", "-X", method, url] + H
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    if d.get("object") == "error":
        raise RuntimeError(f"{url}: {d.get('message')}")
    return d


def upload_png(name):
    d = api("POST", "https://api.notion.com/v1/file_uploads",
            {"filename": name, "content_type": "image/png"})
    u = json.loads(subprocess.run(["curl", "-s", "-X", "POST", d["upload_url"]] + H
                                  + ["-F", f"file=@{os.path.join(S, name)};type=image/png"],
                                  capture_output=True, text=True).stdout)
    assert u.get("status") == "uploaded", u
    print("uploaded", name)
    return d["id"]


def rt(t, bold=False, code=False):
    return {"type": "text", "text": {"content": t},
            "annotations": {"bold": bold, "code": code}}


def para(*rts):
    return {"type": "paragraph", "paragraph": {"rich_text": list(rts)}}


def h2(t):
    return {"type": "heading_2", "heading_2": {"rich_text": [rt(t)]}}


def bullet(*rts):
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": list(rts)}}


def callout(rts, emoji, color="default"):
    return {"type": "callout", "callout": {"rich_text": rts,
            "icon": {"type": "emoji", "emoji": emoji}, "color": color}}


def image(fid, caption):
    return {"type": "image", "image": {"type": "file_upload",
            "file_upload": {"id": fid}, "caption": [rt(caption)]}}


def row(cells, bold=False):
    return {"type": "table_row", "table_row": {"cells": [[rt(c, bold=bold)] for c in cells]}}


NAMES = [(15, 0.5), (15, 0.9), (23, 0.5), (23, 0.9)]
ids = {(f, t): upload_png(f"cue_case_Garden_f{f:02d}_tau{t}.png") for f, t in NAMES}

blocks = [
    h2("τ 검증: cue 임계값은 0.5로 유지 (Garden 케이스 스터디)"),
    callout([rt("결론: τ=0.5 유지. ", bold=True),
             rt("250프레임 집계만 보면 τ=0.9가 IoU 0.23→0.53으로 더 좋아 보이지만, 이는 "
                "cue를 최종 판정기로 볼 때의 얘기다. cue는 proposal 단계이고 뒤따르는 fusion은 "
                "켜진 것 중 틀린 것을 3D consistency로 걷어낼 뿐, 한 번도 안 켜진 픽셀을 되살리지 "
                "못한다. τ=0.9는 실제 변화의 23%를 회수 불가능하게 버린다(τ=0.5는 3.6%). "
                "cue는 참 마스크를 넉넉히 감싸는 역할이어야 한다.")], "🎯", "blue_background"),

    para(rt("아래 4장은 Garden 두 프레임을 두 임계값에서 나란히 놓은 것이다. 패널 구성: "),
         rt("원본 RGB | 3DGS 렌더(변화 전 장면) | 연속 cue + τ 등고선 | 이진화 결과 vs GT"
            "(TP 초록 / FP 빨강 / FN 파랑)", code=True), rt(".")),

    para(rt("핵심은 두 프레임이 정반대로 움직인다는 것이다 — 즉 τ 상향의 이득은 "
            "프레임마다 다르고, 집계 수치는 그 둘을 뭉갠 평균이다.")),

    {"type": "table", "table": {"table_width": 4, "has_column_header": True, "children": [
        row(["프레임", "τ=0.5", "τ=0.9", "해석"], bold=True),
        row(["frame 15 (렌더 양호)", "prec 0.24 / rec 1.00", "prec 0.54 / rec 0.97",
             "FP가 참 영역을 감싼 halo라서 τ가 halo만 벗겨냄 — 거의 공짜"]),
        row(["frame 23 (렌더 실패)", "prec 0.05 / rec 1.00", "prec 0.08 / rec 0.91",
             "FP가 복원 실패 영역이라 cue 값이 최상위권 — τ로 못 거름"]),
    ]}},

    h2("frame 15 — 렌더가 멀쩡한 경우"),
    image(ids[(15, 0.5)], "frame 15, τ=0.5. FP(빨강)가 TP(초록) 덩어리를 감싼 테두리 형태. "
                          "실제 변화는 전부 포착(recall 1.00)."),
    image(ids[(15, 0.9)], "frame 15, τ=0.9. halo가 벗겨져 precision 0.24→0.54, recall은 0.97로 "
                          "거의 유지. 이 프레임만 보면 τ 상향이 이득처럼 보인다."),

    h2("frame 23 — 3DGS가 왼쪽 아래 바닥을 복원하지 못한 경우"),
    image(ids[(23, 0.5)], "frame 23, τ=0.5. 왼쪽 아래 거대한 FP 덩어리 — 렌더 패널의 같은 자리가 "
                          "어둡게 뭉개져 있다. 변화가 아니라 복원 실패다."),
    image(ids[(23, 0.9)], "frame 23, τ=0.9. precision 0.05→0.08에 그치고 FN(파랑)이 등장하기 "
                          "시작한다. 렌더가 원본보다 0.10 이상 어두운 영역(화면의 9%)이 FP에서 "
                          "차지하는 비중은 τ=0.5에서 37.6%, τ=0.9에서 60.7%로 오히려 커진다."),

    h2("따라서"),
    bullet(rt("τ가 사주는 것은 halo FP뿐이고, 남는 FP는 임계값 문제가 아니라 "),
           rt("복원 커버리지 문제", bold=True), rt("다.")),
    bullet(rt("이는 τ 스윕에서 씬 GT 면적과 오탐률의 상관이 τ와 무관하게 r≈−0.87로 유지되는 것과 "
              "일치한다 — precision이 낮은 씬은 임계값이 잘못 잡힌 씬이 아니라 렌더 지지가 나쁜 씬.")),
    bullet(rt("다음 후보: τ 튜닝이 아니라 렌더 신뢰도 게이팅(가우시안 opacity/누적 투과도가 낮거나 "
              "렌더가 비정상적으로 어두운 픽셀에서 cue를 죽이기) — recall 손실 없이 FP를 줄이는 방향.")),
    para(rt("재현: "), rt("experiments/cue_tau_sweep.py", code=True), rt(" (스윕·상관차트), "),
         rt("experiments/cue_tau_case_montage.py", code=True), rt(" (위 4장).")),
]

for i in range(0, len(blocks), 100):
    api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children",
        {"children": blocks[i:i + 100]})
print(f"appended {len(blocks)} blocks to page {PAGE}")
