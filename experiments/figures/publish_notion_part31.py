# Publish the Part 3.1 importance-diagnostic section (with the highlight-vs-TP
# visualization) to the Notion Part 3 page.
#   python experiments/figures/publish_notion_part31.py
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


def h3(t):
    return {"type": "heading_3", "heading_3": {"rich_text": [rt(t)]}}


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


FIGS = ["highlight_Garden_f23.png", "highlight_Zen_f06.png",
        "highlight_Zen_f04.png", "highlight_Porch_f00.png",
        "highlight_curve_Garden.png", "highlight_curve_Zen.png"]
ids = {f: upload_png(f) for f in FIGS}

blocks = [
    h2("Part 3.1 — R_change Gaussian importance/confidence 진단"),
    callout([rt("결론: FP를 걷어내는 신호는 존재하고, 그 정체는 전부 '관측 support'다. ", bold=True),
             rt("agreement는 오히려 해롭고 observability는 측정 가능한 기여가 0이다. "
                "고정된 all-25 R_change 위에서 Gaussian별 change belief·support·"
                "multi-view agreement·uncertainty·geometric observability를 계산하고 "
                "209개 importance 정의를 25개 query pose에 렌더해 비교했다. "
                "10씬 × 25뷰, GT는 importance 계산이 끝난 뒤에만 열리도록 런타임에서 차단. "
                "selector에는 아직 연결하지 않았다.")], "🔬", "blue_background"),

    para(rt("과제 정의: all-25 예측이 켠 영역("), rt("P_v = M_all25 ≥ 0.5", code=True),
         rt(") 안에서 TP 픽셀을 FP 픽셀 위로 정렬할 수 있는가. 기준선은 예측 자체의 "
            "정확도 prevalence = 0.656 (250프레임).")),

    h3("1) 어떤 정의가 이기는가"),
    table(5, [
        row(["importance", "AUPRC", "p@recall .95", "FP 제거율 @.95", "비고"], bold=True),
        row(["I2 = m·S", "0.8295", "0.7248", "0.2729", "종합 1위"]),
        row(["I1 = m (belief)", "0.8281", "0.7021", "0.2406", "TP 보존 최고"]),
        row(["S0 = support (κ₀=8)", "0.8050", "0.7329", "0.2962", "FP 억제 최고"]),
        row(["I0 = raw c (기준선)", "0.8157", "0.7015", "0.2372", "—"]),
        row(["V0 = verification", "0.6275", "0.6567", "0.0481", "prevalence 이하"]),
    ]),
    para(rt("실제 변화의 95%를 유지하는 지점에서 FP를 24% → 30% 걷어낸다. "
            "실재하는 개선이지만 6포인트짜리이지 해결책은 아니다.")),

    h3("2) 192조합 지수 스윕이 이례적으로 깨끗하게 갈렸다"),
    para(rt("I = m^γ · S^α · A^β · O^δ 의 지수별 주변평균 AUPRC:")),
    table(5, [
        row(["지수", "0", "0.5", "1", "2"], bold=True),
        row(["γ (belief)", "—", "0.8221", "0.8203", "0.8166"]),
        row(["α (support)", "0.8178", "0.8204", "0.8206", "0.8200"]),
        row(["β (agreement)", "0.8283", "0.8223", "0.8176", "0.8106"]),
        row(["δ (observability)", "0.8197", "0.8197", "0.8197", "0.8197"]),
    ]),
    bullet(rt("β(agreement)는 "), rt("단조적으로 해롭다", bold=True),
           rt(" — 상위 15개 변형이 전부 β=0.")),
    bullet(rt("δ(observability)는 소수 넷째 자리까지 완전히 평평 — 기여 0.")),
    bullet(rt("원인은 프로브 노이즈가 아니다(split-half ρ ≥ 0.9996). Gaussian의 약 70%가 "
              "damping floor에 정확히 놓여 O=0인 게 원인 — change 렌더의 xyz adjoint는 "
              "c가 평평한 곳에서 소멸한다.")),
    bullet(rt("대신 같은 블록의 "), rt("condition number", bold=True),
           rt("가 TP purity와 ρ=−0.227로 표에서 가장 강한 음의 신호다(logdet은 −0.055로 무의미). "
              "기하 축을 잘못된 함수로 읽고 있었다.")),

    h3("3) 켜진 영역 중 얼마가 진짜인가 — 시각화"),
    para(rt("아래 그림들이 이 진단의 핵심이다. 패널 구성: "),
         rt("원본 RGB | all-25 예측(TP 초록/FP 빨강/FN 파랑) | importance 히트맵 | "
            "상위 10·20·30·50·100%만 남긴 영역", code=True),
         rt(". 각 패널 제목에 켜진 영역의 TP 비율과 recall이 적혀 있다.")),

    image(ids["highlight_Garden_f23.png"],
          "Garden/23 — 3DGS가 왼쪽 아래 바닥을 복원 못 한 프레임. 예측 전체는 13%만 진짜인데, "
          "상위 20%만 남기면 복원 실패 덩어리가 완전히 사라지고 60%가 진짜가 되면서 recall은 0.95를 유지한다."),
    callout([rt("자기 정정: ", bold=True),
             rt("직전 보고서에 \"Garden/23의 FP 덩어리는 m·S·A·O·Conf 어느 축으로도 억제되지 "
                "않는다\"고 썼는데 틀렸다. 그건 state 몽타주의 절대 밝기를 눈으로 보고 내린 "
                "판단이었고, 지표가 쓰는 건 밝기가 아니라 순위다. 그 덩어리는 진짜 변화보다 "
                "아래 순위라서 상위 20% 컷에서 그냥 잘려나간다. 밝기가 아니라 순위로 판단할 것.")],
            "⚠️", "orange_background"),

    image(ids["highlight_Zen_f06.png"],
          "Zen/6 — 정상 프레임. 예측이 이미 70% 정확하고, 상위 10%에서 98%."),
    image(ids["highlight_Zen_f04.png"],
          "Zen/4 — 이 연구 최악의 케이스. 예측이 1.8%만 정확하고, 상위 10%로 잘라도 18.4%까지밖에 "
          "못 간다. 10배 개선이지만 여전히 거의 전부 오탐 — 재정렬로 못 살리는 프레임도 있다."),
    image(ids["highlight_Porch_f00.png"],
          "Porch/0 — toxic으로 분류된 프레임인데 예측이 이미 88% 정확하다. 걷어낼 FP 자체가 없어서 "
          "importance가 할 일이 없다."),

    table(4, [
        row(["씬", "예측 전체", "상위 10%", "상위 1%"], bold=True),
        row(["Garden", "0.500", "0.929", "1.000"]),
        row(["Zen", "0.616", "0.920", "0.992"]),
        row(["Porch", "0.719", "0.906", "0.934"]),
        row(["Printing_area", "0.888", "0.990", "1.000"]),
    ]),
    image(ids["highlight_curve_Garden.png"],
          "Garden — 회색이 개별 프레임, 파랑이 평균. 평균은 매끄럽지만 50~100% 구간에서 프레임이 "
          "크게 벌어지고, 거기에 toxic 프레임들이 있다."),
    image(ids["highlight_curve_Zen.png"], "Zen — 같은 곡선. 아래로 처진 회색 선이 frame 4."),

    h3("4) Zen형과 Porch형은 같은 식으로 못 다룬다"),
    para(rt("아래는 모두 I2 = m·S 기준 (raw c 대비):")),
    table(4, [
        row(["씬", "AUPRC 변화", "p@r.95 변화", "해석"], bold=True),
        row(["Zen", "0.8830 → 0.8991", "0.675 → 0.783 (+10.8pp)",
             "state로 분리 가능 — precision 최대 이득"]),
        row(["Garden", "0.7547 → 0.8089", "0.567 → 0.618",
             "AUPRC 최대 이득 (+0.054)"]),
        row(["Porch", "0.8703 → 0.8557", "0.758 → 0.776",
             "유일하게 raw c를 못 이김 (−0.015)"]),
        row(["Printing_area", "0.9562 → 0.9592", "0.911 → 0.916",
             "toxic 프레임이 통계적으로 구분 불가"]),
    ]),
    para(rt("Porch의 toxic 프레임(0, 7)은 오히려 그 씬에서 가장 깨끗하다(prevalence 0.847 vs 0.708). "
            "즉 Porch형 손상은 \"이 프레임의 cue가 틀렸다\"가 아니라 viewpoint/3D 문제이고, "
            "per-Gaussian 필터링으로는 원리적으로 못 잡는다 — Part 3의 cue 몽타주 결론과 정확히 일치한다.")),

    h3("5) 정직하게 짚어둘 것"),
    bullet(rt("겉보기 FP 억제의 일부는 alpha 커버리지다. alpha로 정규화하면 AUPRC가 "
              "0.03~0.08 떨어지는데, 이는 기준선 대비 전체 이득과 맞먹는 크기다.")),
    bullet(rt("이득의 절대 크기가 작다: 최고 변형이 raw c 대비 +0.014 AUPRC.")),
    bullet(rt("Gaussian 단위 수치는 alpha-compositing 귀속이지 3D GT 라벨이 아니다.")),
    bullet(rt("seed 0 한 개만. 씬 단위 run-to-run 노이즈는 재측정하지 않았다.")),

    h3("6) 다음 selector 실험으로 가져갈 것"),
    bullet(rt("I2 = m·S", code=True), rt(" — 종합 분리 성능 1위")),
    bullet(rt("m^0.5 · S^0.5", code=True),
           rt(" — 최고 스윕 계열. δ가 무의미함이 증명됐으므로 더 싼 형태를 쓴다")),
    bullet(rt("O를 대체할 "), rt("conditioning 기반 기하 항", bold=True),
           rt(" — 검증된 게 아니라 새 후보")),
    bullet(rt("importance와 verification은 반드시 분리 유지 — verification/uncertainty가 "
              "prevalence보다 낮게 나온 것이 실증이다")),
    para(rt("재현: "), rt("experiments/visualize_rchange_importance.py", code=True), rt(" → "),
         rt("experiments/analyze_rchange_importance.py", code=True), rt(" → "),
         rt("experiments/rchange_highlight_vs_tp.py", code=True), rt(". 보고서는 "),
         rt("docs/rchange_importance_heatmap_report.md", code=True), rt(".")),
]

for i in range(0, len(blocks), 100):
    api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children",
        {"children": blocks[i:i + 100]})
print(f"appended {len(blocks)} blocks to page {PAGE}")
