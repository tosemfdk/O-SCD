# Swap the Notion page's 10 montages for the GT-overlay versions (red = GT
# change region, per-frame changed-pixel share in each tag) and extend the
# explanation with what the overlay reveals.
import json
import os
import subprocess

S = os.path.dirname(os.path.abspath(__file__))
TOKEN = os.environ["NOTION_TOKEN"]
PAGE = "39fcbb7d793780e2b14fe222272c1f44"
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
    path = os.path.join(S, name)
    d = api("POST", "https://api.notion.com/v1/file_uploads",
            {"filename": name, "content_type": "image/png"})
    u = json.loads(subprocess.run(["curl", "-s", "-X", "POST", d["upload_url"]] + H
                                  + ["-F", f"file=@{path};type=image/png"],
                                  capture_output=True, text=True).stdout)
    assert u.get("status") == "uploaded", u
    print("uploaded", name)
    return d["id"]


def rt(text, bold=False):
    return {"type": "text", "text": {"content": text}, "annotations": {"bold": bold}}


def image(fid, caption):
    return {"type": "image", "image": {"type": "file_upload",
            "file_upload": {"id": fid}, "caption": [rt(caption)]}}


NOTE = " 빨강 = GT 변화 영역, 프레임 라벨의 %는 변화 픽셀 비율."
CAP = {
    "cantina": "Cantina: oracle-5 (위, 0.596) vs 최강 uniform offset (아래, 0.475), +25%." + NOTE,
    "playground": "Playground: oracle-5 (위, 0.520) vs 최강 uniform offset (아래, 0.426), +22%." + NOTE,
    "garden": "Garden: 0.552 vs 0.511 (+8%)", "lounge": "Lounge: 0.604 vs 0.572 (+6%)",
    "lunch_room": "Lunch_room: 0.425 vs 0.382 (+11%)", "meeting_room": "Meeting_room: 0.568 vs 0.532 (+7%)",
    "porch": "Porch: 0.639 vs 0.624 (+2%)", "pots": "Pots: 0.658 vs 0.643 (+2%)",
    "printing_area": "Printing_area: 0.712 vs 0.629 (+13%)", "zen": "Zen: 0.585 vs 0.548 (+7%)",
}

children = api("GET", f"https://api.notion.com/v1/blocks/{PAGE}/children?page_size=100")["results"]
images = [b["id"] for b in children if b["type"] == "image"]
toggles = [b["id"] for b in children if b["type"] == "toggle"]
assert len(images) == 7 and len(toggles) == 1, (len(images), len(toggles))
mont_cantina, mont_playground = images[2], images[3]

# inline pair: insert new after the old playground image, then delete both old
api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children", {
    "after": mont_playground,
    "children": [image(upload_png("montage_cantina.png"), CAP["cantina"]),
                 image(upload_png("montage_playground.png"), CAP["playground"])]})
api("DELETE", f"https://api.notion.com/v1/blocks/{mont_cantina}")
api("DELETE", f"https://api.notion.com/v1/blocks/{mont_playground}")
print("inline montages swapped")

# toggle: drop old children, append the 8 overlay versions
tid = toggles[0]
for b in api("GET", f"https://api.notion.com/v1/blocks/{tid}/children?page_size=20")["results"]:
    api("DELETE", f"https://api.notion.com/v1/blocks/{b['id']}")
api("PATCH", f"https://api.notion.com/v1/blocks/{tid}/children", {
    "children": [image(upload_png(f"montage_{n}.png"), CAP[n] + NOTE)
                 for n in ["garden", "lounge", "lunch_room", "meeting_room",
                           "porch", "pots", "printing_area", "zen"]]})
print("toggle montages swapped")

# explanation paragraph: add what the GT overlay reveals
mexp = None
for b in children:
    if b["type"] == "paragraph":
        t = "".join(x["plain_text"] for x in b["paragraph"]["rich_text"])
        if t.startswith("같은 씬, 같은 예산에서"):
            mexp = b["id"]
            break
assert mexp
api("PATCH", f"https://api.notion.com/v1/blocks/{mexp}", {"paragraph": {"rich_text": [
    rt("같은 씬, 같은 예산에서 승부를 가른 것이 이미지로 보인다 (rev3 최종 지도, 비교 상대는 각 씬의 "),
    rt("최강", bold=True),
    rt(" uniform offset — 체리피킹 없는 비교, 빨강 = GT 변화 영역). Cantina의 oracle 5장은 전부 "
       "후반부(12–21)로, 변화가 몰린 카운터 구역을 서로 다른 각도에서 재관측한다 — 최강 uniform과 "
       "겹치는 프레임이 단 한 장도 없이 +25%. Playground는 변화가 화면의 0–1%뿐인 씬인데, oracle은 "
       "그 작은 변화가 실제로 잡히는 근거리 프레임에 예산을 몰아준다(+22%). GT 오버레이가 드러내는 "
       "중요한 사실 하나: "),
    rt("uniform 프레임들도 변화를 '보고는' 있다 (Cantina uniform은 프레임당 5–16%). 관건은 변화를 "
       "보느냐가 아니라 어떤 각도·거리 조합으로 보느냐다", bold=True),
    rt(" — 이것이 근거 3의 D-opt(시차 다양성) 기준이 형식화하는 성질이고, 단순 'cue 커버리지' "
       "선택기가 uniform에 지는 이유이기도 하다. 나머지 8개 씬은 아래 토글에."),
]}})
print("explanation updated")
