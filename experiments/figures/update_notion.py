# Revise the feasibility page: honest claims for evidence 2/3/4 + montage section.
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


def upload_png(path):
    d = api("POST", "https://api.notion.com/v1/file_uploads",
            {"filename": os.path.basename(path), "content_type": "image/png"})
    u = json.loads(subprocess.run(["curl", "-s", "-X", "POST", d["upload_url"]] + H
                                  + ["-F", f"file=@{path};type=image/png"],
                                  capture_output=True, text=True).stdout)
    assert u.get("status") == "uploaded", u
    print("uploaded", os.path.basename(path))
    return d["id"]


def rt(text, bold=False, color="default"):
    return {"type": "text", "text": {"content": text},
            "annotations": {"bold": bold, "color": color}}


children = api("GET", f"https://api.notion.com/v1/blocks/{PAGE}/children?page_size=60")["results"]


def find(prefix):
    for b in children:
        if b["type"] == "paragraph":
            t = "".join(x["plain_text"] for x in b["paragraph"]["rich_text"])
            if t.startswith(prefix):
                return b["id"]
    raise KeyError(prefix)


# ---- evidence 2: replace the overstated paragraph ----------------------------
p2 = find("uniform 선택은 K=5에서 94%")
api("PATCH", f"https://api.notion.com/v1/blocks/{p2}", {"paragraph": {"rich_text": [
    rt("위 곡선은 10개 씬 평균이며, 씬별로는 uniform-5가 all-25를 넘는 씬도 4/10 있다"
       "(Garden, Lounge, Pots, Playground). 즉 \"적게 쓰면 항상 손해\"도 \"uniform이면 충분\"도 "
       "성립하지 않는다. 정확한 명제는 다음이다: "),
    rt("같은 K=5에서 어떤 5장을 고르느냐에 따라 mIoU가 씬당 평균 0.174 (최대 0.276) 벌어지는 반면, "
       "25장 전부와 uniform-5의 차이(장수 효과)는 평균 0.054에 그친다 — 10/10 씬에서 선택 효과가 "
       "장수 효과보다 크다.", bold=True),
    rt(" \"몇 장\"이 아니라 \"어떤 장\"이 지배 변수라는 것이 selection 연구의 존재 이유이고, "
       "oracle-5(평균 109%)는 그 선택 효과의 상단을 보여준다."),
]}})
print("evidence 2 updated")

# ---- evidence 3: precise description of what the chart measures --------------
p3 = find("변화의 3D 국소화에 필요한 것은")
api("PATCH", f"https://api.notion.com/v1/blocks/{p3}", {"paragraph": {"rich_text": [
    rt("주의: 이 그래프의 y축은 mIoU가 아니라 ", bold=True),
    rt("단일 Gaussian의 위치·크기 파라미터에 대한 Fisher 정보량(Δ logdet H)이다. "
       "변화를 3D에서 국소화하려면 같은 지점을 서로 다른 각도(시차)에서 봐야 한다는 기하학적 요구를 "
       "직접 재는 양으로, Garden 실제 재구성 + 실측 카메라 pose 25개 pool + target 10개 평균으로 측정했다. "
       "결과: exact D-opt는 2장으로 uniform 5장의 정보량에 도달(10/10 target)했고, 눈감은 선택은 "
       "예산의 26–28%를 target이 아예 보이지 않는 view에 낭비했다. 이것은 end-task(mIoU) 증명이 아니라 "),
    rt("선택 기준이 딛고 선 메커니즘의 검증", bold=True),
    rt("이며, 106개의 단위·성질 테스트가 수학적 정확성을 뒷받침한다."),
]}})
print("evidence 3 updated")

# ---- evidence 4: single-run caveat + honest overall standing ------------------
p4 = find("1차 선택기(커버리지 기반)는 uniform에 졌다")
api("PATCH", f"https://api.notion.com/v1/blocks/{p4}", {"paragraph": {"rich_text": [
    rt("1차 선택기(커버리지 기반)는 10씬 평균에서 uniform에 졌다(K=5 기준 −0.039 mIoU). 그러나 패인이 "
       "정확히 규명되었고(방향 정보 부재 → 변화 영역의 시차 다양성 미확보), 방향 인지 항을 추가하자 "
       "진단의 표적이었던 Porch가 그대로 반전되었다(0.319 → 0.530). 단, 위 수치는 단일 런이며 씬 단위 "
       "노이즈가 ±0.05 수준임을 감안해야 하고, "),
    rt("전체 평균은 여전히 uniform에 못 미친다(−0.024)", bold=True),
    rt(". 여기서의 주장은 \"이긴다\"가 아니라 — 실패가 블랙박스가 아니라 설명되고, 그 설명이 수정으로 "
       "이어지는 엔지니어링 문제라는 것이다."),
]}})
print("evidence 4 updated")

# ---- montage section: insert right after the evidence-2 paragraph ------------
m1 = upload_png(os.path.join(S, "montage_cantina.png"))
m2 = upload_png(os.path.join(S, "montage_garden.png"))
api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children", {
    "after": p2,
    "children": [
        {"type": "heading_3", "heading_3": {"rich_text": [rt("이미지 레벨 비교: 이기는 5장은 무엇이 다른가")]}},
        {"type": "image", "image": {"type": "file_upload", "file_upload": {"id": m1},
         "caption": [rt("Cantina: oracle-5 (위, 0.590) vs uniform-5 (아래, 0.459). 두 셋이 공유하는 프레임은 15, 20.")]}},
        {"type": "image", "image": {"type": "file_upload", "file_upload": {"id": m2},
         "caption": [rt("Garden: oracle-5 (위, 0.550) vs uniform-5 (아래, 0.511). 공유 프레임은 5, 10.")]}},
        {"type": "paragraph", "paragraph": {"rich_text": [
            rt("같은 씬, 같은 예산에서 승부를 가른 것이 이미지로 보인다. Cantina의 oracle 5장은 "),
            rt("전부 변화가 있는 싱크대 카운터 구역(얼룩·집기)을 좌/중/우 서로 다른 각도에서 재관측", bold=True),
            rt("하는 반면, uniform 5장은 등간격 규칙 때문에 냉장고·분리수거대 등 변화 없는 구역에 "
               "2–3장을 소비한다. 격차가 작은 Garden(+8%)에서도 패턴은 같다: oracle은 변화 물체"
               "(파란 텀블러)가 크고 선명하게 잡히는 근거리·다각도 프레임에 집중한다. 요약하면 이기는 "
               "선택의 시각적 정체는 \""),
            rt("변화 지점을, 다양한 시차로, 크게", bold=True),
            rt("\"이며 — 이는 근거 3의 D-opt 기준이 정확히 형식화하는 성질이다.")]}},
    ]})
print("montage section inserted")
