# Replace the Notion "reconstruction holes" section with tight-layout images
# (title attached to the panels, no whitespace band). Deletes the 24 blocks the
# previous append added, then re-appends the section with render_holes_tight/.
#   python experiments/figures/republish_notion_render_holes.py
import json
import os
import subprocess

S = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(S))
TIGHT = os.path.join(S, "render_holes_tight")
TOKEN = os.environ.get("NOTION_TOKEN") or \
    open(os.path.join(REPO, ".notion_token")).read().strip()
PAGE = "3a3cbb7d7937808aa48ef3185963afab"
H = ["-H", f"Authorization: Bearer {TOKEN}", "-H", "Notion-Version: 2022-06-28"]


def api(method, url, payload=None):
    cmd = ["curl", "-s", "-X", method, url] + H
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    d = json.loads(out) if out.strip() else {}
    if d.get("object") == "error":
        raise RuntimeError(f"{url}: {d.get('message')}")
    return d


def all_children():
    blocks, cur = [], None
    while True:
        u = f"https://api.notion.com/v1/blocks/{PAGE}/children?page_size=100"
        if cur:
            u += f"&start_cursor={cur}"
        d = api("GET", u)
        blocks += d["results"]
        if not d.get("has_more"):
            return blocks
        cur = d.get("next_cursor")


def upload(path):
    name = os.path.basename(path)
    d = api("POST", "https://api.notion.com/v1/file_uploads",
            {"filename": name, "content_type": "image/png"})
    u = json.loads(subprocess.run(["curl", "-s", "-X", "POST", d["upload_url"]] + H
                                  + ["-F", f"file=@{path};type=image/png"],
                                  capture_output=True, text=True).stdout)
    assert u.get("status") == "uploaded", u
    print("uploaded", name)
    return d["id"]


def rt(t, bold=False, code=False):
    return {"type": "text", "text": {"content": t},
            "annotations": {"bold": bold, "code": code}}


def para(*r):
    return {"type": "paragraph", "paragraph": {"rich_text": list(r)}}


def h2(t):
    return {"type": "heading_2", "heading_2": {"rich_text": [rt(t)]}}


def bullet(*r):
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": list(r)}}


def callout(r, emoji, color="default"):
    return {"type": "callout", "callout": {"rich_text": r,
            "icon": {"type": "emoji", "emoji": emoji}, "color": color}}


def image(fid, cap):
    return {"type": "image", "image": {"type": "file_upload",
            "file_upload": {"id": fid}, "caption": [rt(cap)]}}


def row(cells, bold=False):
    return {"type": "table_row", "table_row": {"cells": [[rt(c, bold=bold)] for c in cells]}}


def table(w, rows):
    return {"type": "table", "table": {"table_width": w,
            "has_column_header": True, "children": rows}}


HEADING = "Cue 오탐의 정체: 레퍼런스 3DGS의 '복원 안 된 영역'이 그대로 새어 나온 것"

# tight image (file, one-line caption). Order = FP-in-bad descending.
FRAMES = [
    ("Zen_f24.png", "Zen/24 — 오른쪽 수목·바닥 복원 실패, cue의 최대 점등 띠가 그 자리. FP의 69%."),
    ("Cantina_f13.png", "Cantina/13 — 복원실패 영역 cue 2.2배, FP의 65%."),
    ("Pots_f03.png", "Pots/3 — 어둡진 않지만 색 복원 오류. FP의 62%."),
    ("Zen_f13.png", "Zen/13 — cue가 화면 54%를 켬(GT 1.6%), 대부분 복원실패 위."),
    ("Playground_f14.png",
     "Playground/14 — 오라클 최악 프레임. 멀치 바닥 전체 복원 실패, precision 0.019."),
    ("Porch_f06.png", "Porch/6 — 복원실패 영역 cue 2.5배, FP의 48%."),
    ("Garden_f23.png", "Garden/23 — 왼쪽 아래 바닥 코너 복원 실패. Part 3 τ 케이스와 동일 프레임."),
    ("Printing_area_f00.png",
     "Printing_area/0 — 복원이 대체로 좋아 precision 0.107로 상대적으로 깨끗."),
    ("Meeting_room_f13.png", "Meeting_room/13 — 복원실패 영역 cue 2.9배."),
    ("Playground_f01.png",
     "Playground/1 — 오라클 최고 프레임. 복원 양호 → 독성 낮음 (14와 같은 씬, 정반대)."),
]


def main():
    # 1) delete the previous section (heading -> end of page)
    blocks = all_children()
    idx = next((i for i, b in enumerate(blocks) if b["type"] == "heading_2"
                and HEADING[:12] in "".join(
                    x["plain_text"] for x in b["heading_2"]["rich_text"])), None)
    if idx is not None:
        victims = blocks[idx:]
        print(f"deleting {len(victims)} old blocks")
        for b in victims:
            api("DELETE", f"https://api.notion.com/v1/blocks/{b['id']}")
    else:
        print("no existing section found — appending fresh")

    # 2) re-append with tight images
    ids = [(cap, upload(os.path.join(TIGHT, fn))) for fn, cap in FRAMES]
    body = [
        h2(HEADING),
        callout([rt("핵심: cue는 변화가 아니라 복원 실패에 반응한다. ", bold=True),
                 rt("레퍼런스 3DGS에서 애초에 잘 관측·복원되지 않은 영역이 inference pose에서 "
                    "원본과 크게 달라 보이고, cue는 그 차이를 '변화'로 오인해 점등한다. "
                    "각 그림의 2번째 패널(3DGS 렌더)의 흐릿·누락 부분과 3번째 패널(cue) 점등이 "
                    "겹친다.")], "🕳️", "blue_background"),
        para(rt("250프레임 전체에서 성립한다 ("), rt("복원실패 = |원본−렌더| > 0.12", code=True),
             rt("):")),
        bullet(rt("복원실패 영역의 cue 평균 / 잘 복원된 영역의 cue 평균 = "),
               rt("중앙값 2.7배", bold=True), rt(", "),
               rt("250프레임 100%가 비율 > 1", bold=True), rt(" (예외 없음).")),
        bullet(rt("복원실패 영역은 화면의 평균 11.6%인데 cue 오탐(FP)의 "),
               rt("33%", bold=True), rt("를 담는다 — 면적 대비 약 3배 집중.")),
        para(rt("아래 그림은 FP가 복원실패 영역에 몰린 정도 순. 각 그림에 수치가 박혀 있다. "
                "패널: "), rt("원본 | 3DGS 렌더 | cue(τ=0.5) | GT", code=True), rt(".")),
    ]
    for cap, fid in ids:
        body.append(image(fid, cap))
    body += [
        callout([rt("Playground 14 vs 1: ", bold=True),
                 rt("같은 씬인데 오라클 최악(14)은 복원 실패, 최고(1)는 복원 양호 — 독성이 "
                    "정확히 렌더 품질 축으로 갈린다.")], "🔀", "gray_background"),
        h2("함의"),
        bullet(rt("이 오탐은 "), rt("cue 임계값(τ)으로 못 거른다", bold=True),
               rt(" — 복원 실패는 photometric 차이가 가장 커서 cue 값이 최상위권 "
                  "(Part 3 τ 스윕 결론과 일치).")),
        bullet(rt("실행 방향: cue를 죽이지 말고 "),
               rt("레퍼런스 렌더 신뢰도가 낮은 픽셀에서 cue를 감쇠", bold=True),
               rt(" — 누적 투과도/알파가 낮거나 |원본−렌더|가 큰 영역. recall 손실 없이 "
                  "FP의 ~1/3을 직접 겨냥.")),
        para(rt("재현: "), rt("experiments/render_holes_tight.py", code=True),
             rt(", "), rt("experiments/render_cue_gt_montage.py", code=True), rt(".")),
    ]
    for i in range(0, len(body), 100):
        api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children",
            {"children": body[i:i + 100]})
    print(f"appended {len(body)} blocks")


if __name__ == "__main__":
    main()
