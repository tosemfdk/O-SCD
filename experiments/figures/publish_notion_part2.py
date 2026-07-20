# Publish the Part-2 (D-opt selector experiments) write-up to Notion.
import json
import os
import subprocess

S = os.path.dirname(os.path.abspath(__file__))
TOKEN = os.environ["NOTION_TOKEN"]  # export NOTION_TOKEN=... before running
PAGE = "3a3cbb7d793780f5999cf6e18767e6db"
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


def rt(text, bold=False, code=False):
    ann = {"bold": bold}
    if code:
        ann["code"] = True
    return {"type": "text", "text": {"content": text}, "annotations": ann}


def para(*rts):
    return {"type": "paragraph", "paragraph": {"rich_text": list(rts)}}


def h2(t):
    return {"type": "heading_2", "heading_2": {"rich_text": [rt(t)]}}


def h3(t):
    return {"type": "heading_3", "heading_3": {"rich_text": [rt(t)]}}


def bullet(*rts):
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": list(rts)}}


def image(fid, caption):
    return {"type": "image", "image": {"type": "file_upload",
            "file_upload": {"id": fid}, "caption": [rt(caption)]}}


def callout(rts, emoji, color="default"):
    return {"type": "callout", "callout": {"rich_text": rts,
            "icon": {"type": "emoji", "emoji": emoji}, "color": color}}


def row(cells, bold=False):
    return {"type": "table_row", "table_row": {"cells": [[rt(c, bold=bold)] for c in cells]}}


ids = {n: upload_png(f"part2_chart_{n}.png") for n in ("a_ladder", "b_inversion", "c_recovery")}

VARIANTS = [
    ("1차 정보질량만 (candidate_only)", "재관측 X / 각도 X", "0.4574", "16.4%", "연속 프레임 3장 중복 선택"),
    ("2차 +의심 재관측 (binary mask)", "재관측 O / 각도 X", "0.4673", "19.2%", "절반의 씬만 개선"),
    ("3차 D-opt 할인 (스칼라 c)", "재관측 반대 / 각도 X", "0.4778", "22.3%", "재관측 회피가 역효과"),
    ("4차 재관측+새 각도 (3×3 방향블록)", "재관측 O / 각도 O", "0.4883", "28.3%", "최초로 uniform 상회"),
]

blocks = [
    callout([rt("Part 2 첫 사이클 결론: 이기는 선택의 두 성분이 실험으로 분리·결합되었다. ", bold=True),
             rt("GT 없는 셀렉터 4종을 같은 프로토콜로 겨루게 한 결과, \"변해 보이는 곳을(재관측) + "
                "아직 안 본 각도에서(방향 다양성)\"를 둘 다 갖춘 4차 셀렉터가 프로젝트 최초로 uniform을 "
                "넘었다 (0.4883 vs 0.4817, 지도 백분위 28% vs 19%; oracle의 83% 수준). 성공 7씬에서는 "
                "oracle 갭의 11–58%를 회수하며, 실패 3씬의 원인은 고정 seed(첫 프레임)로 규명 — "
                "다음 사이클의 목표가 자동으로 정해졌다.")], "🧭"),
    para(rt("설정: Part 1이 남긴 과제 = GT 없이 oracle-5 근처의 5장을 고르는 것. 모든 셀렉터는 "
            "후보의 픽셀을 보지 않고(포즈 + 현재 모델 상태만), 선택된 5장은 표준 파이프라인으로 "
            "시간순 재학습(clean replay) 후 25개 query 시점 전부에서 GT와 비교한다 — 오라클 지도와 "
            "동일 프로토콜이라 씬별 ~1,018개 평가 조합 위의 백분위로도 읽힌다.")),

    h2("루프: 첫 장을 배우고, 다음 장을 고른다"),
    bullet(rt("① 첫 프레임으로 change scene 초벌 학습 (16-iter fusion) → 엉성한 3D 의심 지도")),
    bullet(rt("② 남은 후보를 채점 — 무엇을 점수로 쓰는가가 1~4차의 유일한 차이")),
    bullet(rt("③ 최고점 프레임을 골라 fusion, 의심 지도 갱신 → 5장이 될 때까지 ②로")),
    bullet(rt("④ 채점은 선택과 분리: 고른 5장을 clean replay로 재학습해 평가")),

    h2("1차 → 4차: 실패가 성분을 하나씩 분리했다"),
    image(ids["a_ladder"], "4개 변형의 10씬 평균 mIoU. 점선 = uniform 평균 / all-25 / oracle-5 기준선."),
    {"type": "table", "table": {"table_width": 5, "has_column_header": True, "children":
        [row(["변형", "성분", "mIoU", "지도 백분위", "한 줄 진단"], bold=True)]
        + [row(list(v)) for v in VARIANTS]
        + [row(["(기준) uniform 평균", "—", "0.4817", "18.9%", "눈감은 등간격"], bold=False)
           , row(["(기준) all-25 / oracle-5", "—", "0.5237 / 0.5875", "47.6% / 100%", ""], bold=False)]}},
    para(rt("각 실패가 정보였다: "),
         rt("1차", bold=True),
         rt("(정보량 최대)는 거의 같은 각도의 이웃 프레임을 연달아 골라 무너졌고(Meeting_room에서 "
            "17·18·19 연속 — 다양성 필요의 증명), "),
         rt("2차", bold=True),
         rt("(의심 픽셀 재관측 가중)는 절반의 씬만 고쳤으며(재관측 필요의 증명, 그러나 불충분), "),
         rt("3차", bold=True),
         rt("(교과서적 D-opt 할인)는 재관측을 체계적으로 회피해 우연보다 나쁜 픽을 했다. 그래서 "),
         rt("4차 = 재관측(의심 가중) × 각도 다양성(방향 블록)", bold=True),
         rt("이 유일하게 남는 조합이었고, 실제로 처음으로 uniform을 넘었다.")),

    h2("왜 교과서적 D-opt가 실패하는가 — 222런의 실측"),
    image(ids["b_inversion"], "조건부 실측: 각 후보를 실제로 추가해 잰 진짜 이득과 스코어의 순위상관. "
                              "컨텍스트가 생기는 순간 할인 기준(파랑)만 음수로 반전된다."),
    para(rt("Zen·Porch·Garden에서 후보마다 실제로 학습을 돌려 진짜 한계 이득을 재고(222 replay, "
            "같은 배치), 스코어의 순위 예측력을 쟀다. 아무것도 안 고른 상태에서는 할인 유무가 "
            "무의미하지만(동일 순위), 프레임 하나를 고르는 순간 "),
         rt("\"이미 본 것 할인\" 항이 순위를 뒤집는다", bold=True),
         rt(" (Porch ρ −0.68). 변화 감지에서 같은 영역의 재관측은 비용이 아니라 가치이기 때문 — "
            "구 셀렉터(nbv)의 실패 진단과 같은 결론의 세 번째 독립 확인이며, 처음으로 criterion "
            "수준에서 정량화됐다. 스칼라 change 채널의 diagonal 정보로는 이 구조를 담을 수 없다는 "
            "것이 3차까지의 결론이다.")),

    h2("4차 셀렉터: \"변해 보이는 곳을, 새 각도에서, 다시\""),
    para(rt("의심 Gaussian(현재 change 값 c가 오른 3D 점)마다 "),
         rt("어떤 방향들에서 봤는지를 3×3 행렬로 기억", bold=True),
         rt("한다 — 위치에 대한 렌더링 Jacobian은 시선 방향을 타므로, 새 각도만 행렬의 새 고유방향을 "
            "채운다. 프레임 점수 = 의심 점들에 대한 새-각도 logdet 이득. 물건을 검수할 때 정면만 "
            "세 번 보는 것보다 정면+옆+위가 나은 이유를 그대로 수식화한 것. 부수 성과: 예전 기하 "
            "proxy가 붕괴시켰던 Zen(0.22)이 exact Jacobian으로 바꾸자 그 씬 최고 성적(0.515, "
            "백분위 51%)이 되며 1/d² 근접 편향 진단이 종결됐다.")),
    image(ids["c_recovery"], "씬별 oracle 갭 회수율 = (셀렉터−uniform)/(oracle−uniform). "
                             "빨강 = 실패 3씬, 전부 oracle 조합과 겹침 0/5."),
    para(rt("성공 7씬은 갭의 11–58%를 회수하고 oracle 조합·앵커 프레임과 실제로 겹치는 픽을 한다. "
            "실패 3씬은 겹침이 정확히 0인데, 공통 원인이 "),
         rt("고정 seed", bold=True),
         rt("다: 레시피가 무조건 첫 프레임에서 출발하는데, Porch의 frame 0은 그 씬 최악의 독성 "
            "프레임(한계기여 −0.077)이고 Cantina의 oracle은 시퀀스 후반부에만 산다. 오염된 의심 "
            "지도에서 출발하면 이후의 재관측이 전부 헛발이 되는 연쇄다.")),

    h2("다음 사이클"),
    bullet(rt("seed 안전화: ", bold=True),
           rt("첫 프레임 고정 대신 uniform 2장 seed 또는 독성 가드(cue 크고 합의 낮으면 제외)를 "
              "seed에만 적용 — 실패 3씬이 풀리면 평균 회수율 ~25% 예상")),
    bullet(rt("uniform 상회 +0.007은 단일런 노이즈(±0.02) 경계 — 반복 런으로 확정 필요")),
    bullet(rt("기하 블록 확장(scale/rotation), Instance_2 전이 검증은 그 다음")),

    h2("데이터와 재현"),
    para(rt("셀렉터 구현 "), rt("view_selection/", code=True),
         rt(" + "), rt("subset_oscd.py --frames_method dopt_seq", code=True),
         rt(", 평가 드라이버 "), rt("experiments/run_dopt_seq_eval.py", code=True),
         rt(", 변형별 결과 "), rt("experiments/dopt_seq_results*.csv", code=True),
         rt(", 조건부 실측 "), rt("outputs/change_nbv/s2/", code=True),
         rt(" — 전부 develop-claude 브랜치. 게이트 리포트: "),
         rt("docs/change_nbv_s1_report.md, docs/change_nbv_s2_report.md", code=True),
         rt(".")),
]

r = api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children", {"children": blocks})
print("appended", len(r.get("results", [])), "blocks")
