# Publish the "cue false alarms sit on reference reconstruction holes" section
# to the Notion Part 3 page: 10 hand-picked 4-panel frames + the 250-frame
# aggregate that backs the claim.
#   python experiments/figures/publish_notion_render_holes.py
import json
import os
import subprocess

S = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(S))
FIGROOT = os.path.join(S, "render_cue_gt_instance1")
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


def table(width, rows):
    return {"type": "table", "table": {"table_width": width,
            "has_column_header": True, "children": rows}}


# (scene, frame, filename, one-line caption)
FRAMES = [
    ("Zen", 24, "frame_24_Inst_1_test_IMG_E2847.png",
     "Zen/24 — 오른쪽 수목·바닥이 복원 안 됨. cue의 가장 큰 점등 띠가 그 자리. FP의 56%가 복원 실패 영역."),
    ("Zen", 13, "frame_13_Inst_1_test_IMG_E2821.png",
     "Zen/13 — cue가 화면의 54%를 켬(GT 1.56%). 복원 실패 영역에서 cue 평균이 잘 된 곳의 2.2배."),
    ("Printing_area", 0, "frame_00_Inst_1_test_IMG_2192.png",
     "Printing_area/0 — 이 씬은 복원이 대체로 좋아 dark 6.4%에 그침. 그래서 precision 0.107로 상대적으로 깨끗."),
    ("Pots", 3, "frame_03_Inst_1_test_IMG_E2664.png",
     "Pots/3 — 복원 오차 영역이 38.8%로 넓고, FP의 62%가 거기 있음."),
    ("Porch", 6, "frame_06_Inst_1_test_IMG_E6660.png",
     "Porch/6 — 복원 실패 영역에서 cue 평균 0.786 vs 잘 된 곳 0.320 (2.5배)."),
    ("Playground", 1, "frame_01_Inst_1_test_IMG_E7024.png",
     "Playground/1 — 오라클 최고 기여 프레임. 복원이 좋아 dark 2.3%뿐 → 독성 낮음."),
    ("Playground", 14, "frame_14_Inst_1_test_IMG_E7062.png",
     "Playground/14 — 오라클 최악 프레임. 멀치 바닥 전체 복원 실패(dark 26.6%), cue가 그 위에 대량 점등. precision 0.019."),
    ("Meeting_room", 13, "frame_13_Inst_1_test_IMG_1875.png",
     "Meeting_room/13 — 복원 오차 영역에서 cue 평균 1.003 vs 0.345 (2.9배)."),
    ("Garden", 23, "frame_23_Inst_1_test_IMG_E7277.png",
     "Garden/23 — 왼쪽 아래 바닥 코너 복원 실패. Part 3의 τ 케이스 스터디와 동일한 프레임, 여기선 render 패널로 원인이 직접 보임."),
    ("Cantina", 13, "frame_13_Inst_1_test_IMG_2889.png",
     "Cantina/13 — 복원 실패 영역에서 cue 평균 0.620 vs 0.279 (2.2배), FP의 65%가 거기."),
]

ids = [(sc, fr, cap, upload(os.path.join(FIGROOT, sc, fn)))
       for sc, fr, fn, cap in FRAMES]

blocks = [
    h2("Cue 오탐의 정체: 레퍼런스 3DGS의 '복원 안 된 영역'이 그대로 새어 나온 것"),
    callout([rt("핵심: cue는 변화가 아니라 복원 실패에 반응한다. ", bold=True),
             rt("레퍼런스 3DGS에서 애초에 잘 관측·복원되지 않은 영역이 inference pose에서 "
                "원본과 크게 달라 보이고, cue는 그 차이를 '변화'로 오인해 점등한다. "
                "아래 각 프레임의 2번째 패널(3DGS 렌더)에서 흐릿·누락된 부분과 3번째 패널(cue)의 "
                "점등 영역이 겹치는지 보라 — 겹친다.")], "🕳️", "blue_background"),

    para(rt("이건 몇 장을 고른 인상이 아니라 250프레임 전체에서 성립한다. "),
         rt("bad = |원본 − 렌더| > 0.12", code=True),
         rt(" 로 '복원 실패 픽셀'을 정의하면:")),
    bullet(rt("복원 실패 영역의 cue 평균 / 잘 복원된 영역의 cue 평균 = "),
           rt("중앙값 2.7배", bold=True),
           rt(" (범위 1.5–5.4). "), rt("250프레임 100%가 비율 > 1", bold=True),
           rt(" — 예외 없음.")),
    bullet(rt("복원 실패 영역은 화면의 평균 11.6%인데, cue 오탐(FP)의 "),
           rt("33%(중앙값 30%)", bold=True),
           rt("를 담는다 — 면적 대비 약 3배 집중.")),

    para(rt("선별한 10개 프레임의 수치 (τ=0.5):")),
    table(5, [
        row(["프레임", "복원실패 면적", "cue 점등", "precision",
             "FP 중 복원실패 영역"], bold=True),
        row(["Zen/24", "21.8%", "22.3%", "0.062", "69%"]),
        row(["Zen/13", "33.7%", "53.9%", "0.029", "56%"]),
        row(["Cantina/13", "23.3%", "18.1%", "0.258", "65%"]),
        row(["Pots/3", "38.8%", "20.9%", "0.125", "62%"]),
        row(["Playground/14", "21.1%", "21.1%", "0.019", "49%"]),
        row(["Porch/6", "17.9%", "26.7%", "0.137", "48%"]),
        row(["Garden/23", "13.4%", "17.6%", "0.047", "47%"]),
        row(["Printing_area/0", "15.5%", "27.0%", "0.107", "37%"]),
        row(["Meeting_room/13", "8.1%", "23.0%", "0.217", "19%"]),
        row(["Playground/1", "5.1%", "9.9%", "0.088", "21%"]),
    ]),
    para(rt("복원이 좋은 프레임일수록(아래로 갈수록) FP가 복원 실패 영역 밖으로 흩어지고 "
            "precision이 올라간다. Playground는 같은 씬 안에서 프레임 14(복원 실패, 오라클 "
            "최악)와 프레임 1(복원 양호, 오라클 최고)이 정확히 이 축으로 갈린다.")),

    para(rt("각 프레임 패널: "),
         rt("원본 RGB | 3DGS 레퍼런스 렌더 | combined cue C_v (τ=0.5) | GT 마스크", code=True),
         rt(". 렌더 패널 제목의 %는 화면 중 0.10 이상 어둡게 렌더된 비율.")),
]
for sc, fr, cap, fid in ids:
    blocks.append(image(fid, cap))

blocks += [
    h2("함의"),
    bullet(rt("이 오탐은 "), rt("cue 임계값(τ)으로 못 거른다", bold=True),
           rt(" — 복원 실패는 photometric 차이가 가장 커서 cue 값이 최상위권이기 때문 "
              "(Part 3 τ 스윕 결론과 일치).")),
    bullet(rt("Part 3.1의 per-Gaussian 진단이 "), rt("observability를 잘못된 함수로 읽고 "
            "있었다는 것과 같은 현상", bold=True),
           rt(": 복원이 안 된 곳은 Gaussian 지지 자체가 약하다. 다음 후보인 conditioning "
              "기반 기하 항이 노리는 것이 바로 이 영역이다.")),
    bullet(rt("실행 방향: cue를 죽일 게 아니라(recall 손실), "),
           rt("레퍼런스 렌더 신뢰도가 낮은 픽셀에서 cue를 감쇠", bold=True),
           rt(" — 누적 투과도/알파가 낮거나 |원본−렌더|가 큰 영역. "
              "recall 손실 없이 FP의 ~1/3을 직접 겨냥.")),
    para(rt("재현: "), rt("experiments/render_cue_gt_montage.py", code=True),
         rt(" (250프레임 4패널), "),
         rt("experiments/playground_frame_compare.py", code=True),
         rt(" (프레임 대조 + 렌더 오차).")),
]

for i in range(0, len(blocks), 100):
    api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children",
        {"children": blocks[i:i + 100]})
print(f"appended {len(blocks)} blocks to page {PAGE}")
