# Bring the Notion feasibility page up to the rev3 map (2026-07-17):
# replace charts A/B/D, swap the 2-scene montage for Cantina+Playground inline
# + a toggle with the remaining 8 scenes, and fix in-text numbers
# (+10.3% -> +11.8%, 109% -> 112%, selection-vs-count effect).
import json
import os
import subprocess

S = os.path.dirname(os.path.abspath(__file__))
TOKEN = os.environ["NOTION_TOKEN"]  # export NOTION_TOKEN=... before running
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


def rt(text, bold=False, color="default"):
    return {"type": "text", "text": {"content": text},
            "annotations": {"bold": bold, "color": color}}


def image(fid, caption):
    return {"type": "image", "image": {"type": "file_upload",
            "file_upload": {"id": fid}, "caption": [rt(caption)]}}


def replace_block(old_id, new_blocks):
    api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children",
        {"after": old_id, "children": new_blocks})
    api("DELETE", f"https://api.notion.com/v1/blocks/{old_id}")


children = api("GET", f"https://api.notion.com/v1/blocks/{PAGE}/children?page_size=100")["results"]
images = [b["id"] for b in children if b["type"] == "image"]
assert len(images) == 7, f"expected 7 images, got {len(images)}"
chart_a, chart_b, mont1, mont2, chart_c, chart_e, chart_d = images


def find_para(prefix):
    for b in children:
        if b["type"] in ("paragraph", "callout"):
            t = "".join(x["plain_text"] for x in b[b["type"]]["rich_text"])
            if t.startswith(prefix):
                return b["id"]
    raise KeyError(prefix)


# ---- charts A / B / D -> rev3 versions --------------------------------------
replace_block(chart_a, [image(upload_png("chart_a_oracle_map.png"),
    "씬별 uniform-5 평균 / all-25 평균 / 최적 5장 (rev3 지도, 씬당 ~590 evals). 파란 라벨은 all-25 대비 이득.")])
replace_block(chart_b, [image(upload_png("chart_b_saturation.png"),
    "uniform 선택의 포화 곡선 (10씬 평균, all-25 대비 %). oracle-5는 같은 예산으로 112%.")])
replace_block(chart_d, [image(upload_png("chart_d_toxic.png"),
    "프레임별 GT-free 특징 공간 (rev3 한계기여도, 자동 라벨). 독성의 절반(Zen류)은 '큰 cue + 낮은 3D 합의' "
    "구석에 모이고, 나머지 절반(Porch/0류)은 프레임 단독 특징으로 설명되지 않는다.")])
print("charts A/B/D replaced")

# ---- montage section: Cantina + Playground inline, 8 scenes in a toggle -----
CAP = {
    "cantina": "Cantina: oracle-5 (위, 0.596) vs 최강 uniform offset (아래, 0.475), +25%. 공유 프레임 없음 — oracle은 전부 후반부(12–21).",
    "playground": "Playground: oracle-5 (위, 0.520) vs 최강 uniform offset (아래, 0.426), +22%. 공유는 21 하나.",
    "garden": "Garden: 0.552 vs 0.511 (+8%)", "lounge": "Lounge: 0.604 vs 0.572 (+6%)",
    "lunch_room": "Lunch_room: 0.425 vs 0.382 (+11%)", "meeting_room": "Meeting_room: 0.568 vs 0.532 (+7%)",
    "porch": "Porch: 0.639 vs 0.624 (+2%)", "pots": "Pots: 0.658 vs 0.643 (+2%)",
    "printing_area": "Printing_area: 0.712 vs 0.629 (+13%)", "zen": "Zen: 0.585 vs 0.548 (+7%)",
}
new_inline = [image(upload_png("montage_cantina.png"), CAP["cantina"]),
              image(upload_png("montage_playground.png"), CAP["playground"])]
replace_block(mont2, new_inline)          # insert new pair after old #2 ...
api("DELETE", f"https://api.notion.com/v1/blocks/{mont1}")  # ... drop old pair
print("inline montages replaced")

toggle = {"type": "toggle", "toggle": {
    "rich_text": [rt("나머지 8개 씬 몽타주 전부 보기 (rev3 최종 지도 vs 씬별 최강 uniform offset)", bold=True)],
    "children": [image(upload_png(f"montage_{n}.png"), CAP[n])
                 for n in ["garden", "lounge", "lunch_room", "meeting_room",
                           "porch", "pots", "printing_area", "zen"]]}}
mexp = find_para("같은 씬, 같은 예산에서")
api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children",
    {"after": mexp, "children": [toggle]})
print("toggle with 8 montages added")

# ---- text updates ------------------------------------------------------------
api("PATCH", f"https://api.notion.com/v1/blocks/{mexp}", {"paragraph": {"rich_text": [
    rt("같은 씬, 같은 예산에서 승부를 가른 것이 이미지로 보인다 (rev3 최종 지도, 비교 상대는 각 씬의 "),
    rt("최강", bold=True),
    rt(" uniform offset — 체리피킹 없는 비교다). Cantina의 oracle 5장은 전부 후반부(12–21)로, "
       "변화가 몰린 싱크대 카운터 구역을 서로 다른 각도에서 재관측한다 — 최강 uniform과 겹치는 프레임이 "
       "단 한 장도 없이 +25%. Playground의 oracle은 변화 물체(파란 타워 구조물)를 근거리·다각도로 "
       "3장 재관측하고 전경 2장을 더해 +22%. 10개 씬 전부에서 패턴은 같다: 이기는 선택의 시각적 정체는 \""),
    rt("변화 지점을, 다양한 시차로, 크게", bold=True),
    rt("\"이며 — 근거 3의 D-opt 기준이 정확히 형식화하는 성질이다. 나머지 8개 씬은 아래 토글에."),
]}})

p2 = find_para("위 곡선은 10개 씬 평균이며")
api("PATCH", f"https://api.notion.com/v1/blocks/{p2}", {"paragraph": {"rich_text": [
    rt("위 곡선은 10개 씬 평균이며, 씬별로는 uniform-5가 all-25를 넘는 씬도 4/10 있다"
       "(Garden, Lounge, Pots, Playground). 즉 \"적게 쓰면 항상 손해\"도 \"uniform이면 충분\"도 "
       "성립하지 않는다. 정확한 명제는 다음이다: "),
    rt("같은 K=5에서 어떤 5장을 고르느냐에 따라 mIoU가 씬당 평균 0.233 (최대 0.337) 벌어지는 반면, "
       "25장 전부와 uniform-5의 차이(장수 효과)는 평균 0.053에 그친다 — 10/10 씬에서 선택 효과가 "
       "장수 효과보다 크다", bold=True),
    rt(" (rev3 지도의 ~590 조합/씬 기준). \"몇 장\"이 아니라 \"어떤 장\"이 지배 변수라는 것이 "
       "selection 연구의 존재 이유이고, oracle-5(평균 112%)는 그 선택 효과의 상단을 보여준다."),
]}})

cal = find_para("Part 1 결론: 가능하다")
api("PATCH", f"https://api.notion.com/v1/blocks/{cal}", {"callout": {"rich_text": [
    rt("Part 1 결론: 가능하다. ", bold=True),
    rt("\"적은 수의 view를 잘 고르면 25장 전부보다 더 정확한 변화 감지를 얻을 수 있다\"가 "
       "PASLCD Instance_1의 10개 씬 전부에서 실측으로 확인되었다 (평균 +11.8% mIoU; 20배 심층 "
       "재탐색(rev3)에서도 상한이 거의 오르지 않아 이 갭은 노이즈가 아닌 안정된 목표물로 확정). "
       "선택기가 갖춰야 할 핵심 부품들도 각각 독립적으로 검증되었다. "
       "남은 것은 새로운 발견이 아니라 조립(온라인화)이다."),
]}})
print("text updates done (montage para, evidence-2, headline callout)")
