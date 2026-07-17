# Publish the SCD-NBV Part-1 feasibility write-up to the Notion page.
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
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    d = json.loads(out)
    if d.get("object") == "error":
        raise RuntimeError(f"{url}: {d.get('message')}")
    return d


def upload_png(path):
    d = api("POST", "https://api.notion.com/v1/file_uploads",
            {"filename": os.path.basename(path), "content_type": "image/png"})
    up = subprocess.run(["curl", "-s", "-X", "POST", d["upload_url"]] + H
                        + ["-F", f"file=@{path};type=image/png"],
                        capture_output=True, text=True).stdout
    u = json.loads(up)
    if u.get("status") != "uploaded":
        raise RuntimeError(f"upload failed for {path}: {up[:200]}")
    print("uploaded", os.path.basename(path), d["id"])
    return d["id"]


def rt(text, bold=False, color="default"):
    return {"type": "text", "text": {"content": text},
            "annotations": {"bold": bold, "color": color}}


def para(*rts):
    return {"type": "paragraph", "paragraph": {"rich_text": list(rts)}}


def h2(text):
    return {"type": "heading_2", "heading_2": {"rich_text": [rt(text)]}}


def h3(text):
    return {"type": "heading_3", "heading_3": {"rich_text": [rt(text)]}}


def bullet(*rts):
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": list(rts)}}


def image(fid, caption):
    return {"type": "image", "image": {"type": "file_upload",
            "file_upload": {"id": fid}, "caption": [rt(caption)]}}


def callout(rts, emoji="✅", color="green_background"):
    return {"type": "callout", "callout": {"rich_text": rts,
            "icon": {"type": "emoji", "emoji": emoji}, "color": color}}


def divider():
    return {"type": "divider", "divider": {}}


ids = {name: upload_png(os.path.join(S, f"chart_{name}.png"))
       for name in ["a_oracle_map", "b_saturation", "c_target_info", "d_toxic", "e_porch"]}

blocks = [
    callout([rt("Part 1 결론: 가능하다. ", bold=True),
             rt("\"적은 수의 view를 잘 고르면 25장 전부보다 더 정확한 변화 감지를 얻을 수 있다\"가 "
                "PASLCD Instance_1의 10개 씬 전부에서 실측으로 확인되었다 (평균 +10.3% mIoU). "
                "선택기가 갖춰야 할 핵심 부품들도 각각 독립적으로 검증되었다. "
                "남은 것은 새로운 발견이 아니라 조립(온라인화)이다.")],
            emoji="✅"),

    para(rt("모든 수치의 평가 프로토콜은 동일하다: 업데이트에 몇 장을 쓰든, 최종 변화 표현(R_change)을 "
            "25개 query pose 전부에서 렌더하여 25장의 GT 마스크와 비교한 mIoU다. "
            "즉 \"적게 보고도 전체를 잘 설명하는가\"를 측정한다.", color="gray")),

    divider(),
    h2("근거 1 — 잘 고른 5장은 25장 전부를 이긴다 (10/10 씬)"),
    image(ids["a_oracle_map"], "씬별 uniform-5 평균 / all-25 평균 / 최적 5장(oracle). 파란 라벨은 all-25 대비 이득."),
    para(rt("hill-climbing으로 씬당 겨우 ~30개 조합만 탐색했는데도(전체 53,130개의 0.06%) "
            "10개 씬 모두에서 all-25를 넘는 5장 조합이 발견되었다 — 즉 이 수치는 진짜 최적의 "),
         rt("하한", bold=True),
         rt("이다. 핵심은 \"5장이 부족하지 않다\"가 아니라 더 강한 명제다: "),
         rt("25장을 전부 쓰는 것이 오히려 손해", bold=True),
         rt("다. 프레임별 한계기여도 분석 결과, 씬마다 mIoU를 −0.03~−0.11씩 깎아먹는 "),
         rt("독성 프레임", bold=True),
         rt("(오탐 cue 주입원)이 존재하며, all-25는 이를 강제로 전부 섭취한다. "
            "선택은 예산 절약이 아니라 품질 그 자체다.")),

    h2("근거 2 — 눈감은 선택은 100%에 못 미치고, 좋은 선택은 100%를 넘는다"),
    image(ids["b_saturation"], "uniform 선택의 포화 곡선 (10씬 평균, all-25 대비 %). oracle-5는 같은 예산으로 109%."),
    para(rt("uniform 선택은 K=5에서 94%, K=8이 되어야 95%를 넘고, 아무리 늘려도 100%에 수렴할 뿐이다. "
            "반면 잘 고른 5장은 "),
         rt("109%", bold=True),
         rt(" — 곡선 자체의 바깥에 있다. \"어떤 프레임을 고르느냐\"가 \"몇 장을 쓰느냐\"보다 "
            "더 큰 변수라는 뜻이며, 이것이 selection 연구의 존재 이유다.")),

    h2("근거 3 — 선택 기준의 핵심 부품이 이론대로 작동함을 검증했다"),
    image(ids["c_target_info"], "단일 target Gaussian의 기하 정보량 궤적 (Garden 실제 재구성, 실측 pose pool, target 10개 평균)."),
    para(rt("변화의 3D 국소화에 필요한 것은 \"같은 지점을 다른 각도에서 다시 보는 것\"이다. "
            "이를 정량화한 D-optimal 정보 이득 기준을 구현하고 실제 재구성에서 검증했다: "),
         rt("exact D-opt는 2장으로 uniform 5장의 정보량에 도달", bold=True),
         rt("했고(10/10 target), 눈감은 선택은 예산의 26–28%를 target이 보이지도 않는 view에 "
            "낭비했다. 106개의 단위/성질 테스트가 이 수학(정보행렬, 가시성, Beta 불확실성, "
            "렌더러 어드조인트)의 정확성을 뒷받침한다.")),

    h2("근거 4 — 실패조차 진단 가능하고, 진단은 수정으로 이어졌다"),
    image(ids["e_porch"], "Porch (K=5): 방향 무시 선택기의 최악 씬이 방향 인지 도입으로 반전됨."),
    para(rt("1차 선택기(커버리지 기반)는 uniform에 졌다. 그러나 실패의 기제가 정확히 규명되었고"
            "(방향 정보 부재 → 시차 다양성 미확보), 방향 인지를 추가하자 진단된 씬이 그대로 "
            "반전되었다(Porch 0.319 → 0.530). 실패가 블랙박스가 아니라 "),
         rt("설명 가능하고 수정 가능한 엔지니어링 문제", bold=True),
         rt("임이 확인된 것이다.")),
    image(ids["d_toxic"], "프레임별 GT-free 특징 공간. 최악의 독성 유형(큰 cue + 낮은 3D 합의)이 한 구석에 모인다."),
    para(rt("독성 프레임의 정체도 부분적으로 규명되었다: pose 품질은 무관하고(r≈0), "
            "\"크지만 다른 프레임들과 3D상 합의되지 않는 cue\"가 최악 유형이다. "
            "이 신호는 GT 없이 온라인에서 계산 가능하므로(이미 통합한 프레임과의 합의도), "
            "실전 필터의 직접적인 설계 근거가 된다.")),

    divider(),
    h2("왜 \"가능하다\"고 결론짓는가"),
    bullet(rt("상한이 실존한다: ", bold=True),
           rt("찾아야 할 목표물(oracle-5)이 10/10 씬에서 all-25 위에 존재하며, 아주 얕은 탐색으로도 발견된다.")),
    bullet(rt("부품이 검증되었다: ", bold=True),
           rt("방향 인지 정보 기준(2장=uniform 5장), 렌더러 어드조인트 기반 per-Gaussian cue 계량, "
              "독성 cue 시그니처 — 각각 독립 실험으로 확인.")),
    bullet(rt("실패가 해명된다: ", bold=True),
           rt("선택기의 패배는 원인(방향 무시, 근접 편향)이 규명되고 수정이 실증되는, 반복 개선이 "
              "가능한 구조다.")),
    bullet(rt("따라서 Part 2는 발견의 문제가 아니라 조립의 문제다: ", bold=True),
           rt("검증된 부품(D-opt 기준 + Beta 불확실성 + cue 합의 가드)을 온라인 파이프라인에 "
              "통합하고, uniform이 존재할 수 없는 세팅(순차 스트림/로봇 시점 제어)에서 평가하는 것.")),

    h2("정직한 한계 (Part 2에서 다룰 것)"),
    bullet(rt("oracle-5는 GT로 찾은 상한 진단이다 — 방법이 아니다. GT 없이 그 근처에 도달하는 것이 Part 2의 과제다.")),
    bullet(rt("현재 선택기는 아직 uniform을 못 이긴다 (K=5 기준). 단, K=5 구간은 사람도 uniform을 못 이기는 "
              "노이즈 지배 구간임이 확인되어, 승부처는 K≤3과 온라인 세팅으로 재정의되었다.")),
    bullet(rt("Instance_1(10씬)만 측정했다. Instance_2 재현과 벤치마크 노이즈(씬당 ±0.05~0.1, "
              "배치 단위 비결정성) 통제가 필요하다.")),
    bullet(rt("독성 프레임의 절반은 프레임 단독 특징으로 설명되지 않는다 — 셋 맥락(시차/중복) 기반 "
              "기준이 필요하다는 증거이기도 하다.")),

    h2("데이터와 재현"),
    para(rt("모든 실험 코드·결과 CSV·상세 리포트: "),
         {"type": "text", "text": {"content": "github.com/tosemfdk/O-SCD (develop-claude)",
          "link": {"url": "https://github.com/tosemfdk/O-SCD/tree/develop-claude"}}},
         rt(" — 종합 정리는 docs/budgeted_view_findings.md, 실험별 리포트는 experiments/*.md. "
            "본 문서의 모든 그래프는 저장소의 CSV에서 재생성 가능하다.")),
]

# Notion caps children at 100 per request — we're under; single call.
r = api("PATCH", f"https://api.notion.com/v1/blocks/{PAGE}/children", {"children": blocks})
print("appended", len(r.get("results", [])), "blocks")
