# Part 2 cycle 2 analyzer — aggregates keyframe_gl_results.csv into the pilot
# judgment tables (spec §13) and writes docs/change_nbv_keyframe_pilot_report.md.
#
# All comparisons are 3-seed means; per-scene single-run deltas below the
# noise floor (~±0.02) are never used to call a winner (findings §1).
#
#   python experiments/analyze_keyframe_gl.py
#   python experiments/analyze_keyframe_gl.py --no-report   # tables only
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_search import load_done  # noqa: E402

RESULTS_CSV = os.path.join(REPO, "experiments", "keyframe_gl_results.csv")
ALL25_CSV = os.path.join(REPO, "experiments", "keyframe_gl_all25.csv")
REPORT_MD = os.path.join(REPO, "docs", "change_nbv_keyframe_pilot_report.md")

FAILURE_SCENES = ["Cantina", "Porch", "Meeting_room"]
METHOD_ORDER = ["uniform", "dopt_seq_dir", "kf_g_dir", "kf_l_dir_gseed",
                "kf_gu_dir", "kf_gl_dir", "kf_glu_dir", "kf_glu_nodir"]


def oracle_map():
    """Per scene: sorted 5-combo mIoU distribution, best combo, top-3 anchor
    frames (marginal contribution over the full oracle map)."""
    done = load_done()
    vals: dict[str, dict[str, float]] = defaultdict(dict)
    for (sc, c), (miou, _f1) in done.items():
        if len(c.split("-")) == 5 and miou > vals[sc].get(c, -1):
            vals[sc][c] = miou
    out = {}
    for sc, combos in vals.items():
        dist = np.array(sorted(combos.values()))
        best_combo = max(combos, key=combos.get)
        marg = {}
        for f in range(25):
            with_f, without_f = [], []
            for c, v in combos.items():
                (with_f if str(f) in c.split("-") else without_f).append(v)
            if with_f and without_f:
                marg[f] = float(np.mean(with_f) - np.mean(without_f))
        anchors = sorted(marg, key=marg.get, reverse=True)[:3]
        out[sc] = {"dist": dist, "oracle5": float(dist[-1]),
                   "best_combo": set(map(int, best_combo.split("-"))),
                   "anchors": set(anchors), "n_evals": len(combos)}
    return out


def load_rows():
    if not os.path.exists(RESULTS_CSV):
        sys.exit(f"no results yet: {RESULTS_CSV}")
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["miou_replay"] = float(r["miou_replay"])
        r["f1_replay"] = float(r["f1_replay"])
    return rows


def per_scene_seed(rows):
    """(method, scene, seed) -> mIoU; uniform averages its 5 offsets first."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["method"], r["scene"], r["seed"])].append(r["miou_replay"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def scene_means(pss, method, scenes, seeds):
    """scene -> 3-seed mean mIoU (None when incomplete)."""
    out = {}
    for sc in scenes:
        vals = [pss[(method, sc, sd)] for sd in seeds
                if (method, sc, sd) in pss]
        out[sc] = float(np.mean(vals)) if vals else None
    return out


def overlap_stats(rows, omap, method, scenes, seeds):
    """Mean |combo ∩ oracle_best| and |combo ∩ anchors| across scene x seed."""
    ov_best, ov_anchor = [], []
    for r in rows:
        if r["method"] != method or r["scene"] not in scenes \
                or r["seed"] not in seeds or r["scene"] not in omap:
            continue
        combo = set(map(int, r["combo"].split("-")))
        ov_best.append(len(combo & omap[r["scene"]]["best_combo"]))
        ov_anchor.append(len(combo & omap[r["scene"]]["anchors"]))
    return ((float(np.mean(ov_best)) if ov_best else None),
            (float(np.mean(ov_anchor)) if ov_anchor else None))


def map_percentile(omap, scene, miou):
    d = omap.get(scene, {}).get("dist")
    if d is None or not len(d):
        return None
    return float((d < miou).mean()) * 100


def component_contributions(rows):
    """Per method: mean percentile rank of the PICKED frame per component,
    rounds 2+ (from the saved selection manifests)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mp = r.get("manifest_path", "")
        if not mp:
            continue
        path = os.path.join(REPO, mp)
        if not os.path.exists(path):
            continue
        man = json.load(open(path))
        for rnd in man["rounds"][1:]:
            pick = str(rnd["pick"])
            for comp, ranks in rnd["percentile_ranks"].items():
                out[r["method"]][comp].append(ranks[pick])
    return {m: {c: float(np.mean(v)) for c, v in comps.items()}
            for m, comps in out.items()}


def fmt(x, spec=".4f", none="   —  "):
    return none if x is None else f"{x:{spec}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="restrict aggregation (e.g. the pilot 5) — stray "
                         "rows from smoke tests must not pollute paired means")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    if args.scenes:
        rows = [r for r in rows if r["scene"] in args.scenes]
    scenes = sorted({r["scene"] for r in rows})
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    seeds = args.seeds
    omap = oracle_map()
    pss = per_scene_seed(rows)

    all25 = defaultdict(list)
    if os.path.exists(ALL25_CSV):
        for r in csv.DictReader(open(ALL25_CSV)):
            all25[r["scene"]].append(float(r["miou_query"]))

    # ---- main ladder table -------------------------------------------------
    uni = scene_means(pss, "uniform", scenes, seeds)
    d4 = scene_means(pss, "dopt_seq_dir", scenes, seeds)
    lines = []
    header = (f"{'method':16s} {'mIoU':>7s} {'pctile':>7s} {'Δuni':>8s} "
              f"{'Δ4th':>8s} {'ovl_o5':>6s} {'ovl_anc':>7s} {'fail_win':>8s} "
              f"{'zen':>7s}")
    lines.append(header)
    per_method_scene = {}
    for m in methods:
        sm = scene_means(pss, m, scenes, seeds)
        per_method_scene[m] = sm
        valid = [sc for sc in scenes if sm[sc] is not None]
        mean = float(np.mean([sm[sc] for sc in valid])) if valid else None
        paired_u = (float(np.mean([sm[sc] - uni[sc] for sc in valid
                                   if uni[sc] is not None]))
                    if valid and any(uni[sc] is not None for sc in valid)
                    else None)
        paired_4 = (float(np.mean([sm[sc] - d4[sc] for sc in valid
                                   if d4[sc] is not None]))
                    if valid and any(d4[sc] is not None for sc in valid)
                    else None)
        pct = [map_percentile(omap, sc, sm[sc]) for sc in valid]
        pct = float(np.mean([p for p in pct if p is not None])) \
            if any(p is not None for p in pct) else None
        ovb, ova = overlap_stats(rows, omap, m, scenes, seeds)
        fail_win = sum(1 for sc in FAILURE_SCENES
                       if sc in scenes and sm.get(sc) is not None
                       and d4.get(sc) is not None and sm[sc] > d4[sc])
        zen = sm.get("Zen")
        lines.append(f"{m:16s} {fmt(mean, '7.4f')} {fmt(pct, '6.1f')}% "
                     f"{fmt(paired_u, '+8.4f')} {fmt(paired_4, '+8.4f')} "
                     f"{fmt(ovb, '6.2f')} {fmt(ova, '7.2f')} "
                     f"{fail_win if m not in ('uniform', 'dopt_seq_dir') else '—':>8} "
                     f"{fmt(zen, '7.4f')}")
    print("\n".join(lines))

    # ---- per-scene table ----------------------------------------------------
    print(f"\n{'scene':14s}" + "".join(f"{m[:12]:>13s}" for m in methods)
          + f"{'all25':>9s}{'oracle5':>9s}")
    scene_lines = []
    for sc in scenes:
        cells = "".join(
            f"{fmt(per_method_scene[m][sc], '12.4f', '        —   ')} "
            for m in methods)
        a25 = float(np.mean(all25[sc])) if all25.get(sc) else None
        o5 = omap.get(sc, {}).get("oracle5")
        line = f"{sc:14s}{cells}{fmt(a25, '8.4f')} {fmt(o5, '8.4f')}"
        scene_lines.append(line)
        print(line)

    # ---- diagnostics ---------------------------------------------------------
    comp = component_contributions(rows)
    first_frames = defaultdict(list)
    for r in rows:
        if r["method"].startswith("kf_"):
            first_frames[r["method"]].append(int(r["first_frame_id"]))
    print("\nfirst frames picked (kf_*):")
    for m in sorted(first_frames):
        print(f"  {m:16s} {sorted(set(first_frames[m]))}")
    if comp:
        print("\nmean percentile rank of the picked frame (rounds 2+):")
        for m in sorted(comp):
            parts = ", ".join(f"{c}={v:.3f}" for c, v in sorted(comp[m].items()))
            print(f"  {m:16s} {parts}")

    if args.no_report:
        return

    # ---- markdown report ------------------------------------------------------
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write("# ChangeNBV Part 2 Cycle 2 — GL-Keyframe pilot report\n\n")
        f.write("Claim scope: **offline_pool_keyframe_selection** — R_global "
                "consumed all 25 inference images before selection; no active-"
                "NBV/acquisition-saving claims. GT is never used for "
                "selection; the numbers below are clean chronological replays "
                "of each method's 5 picks (same protocol as the oracle map).\n\n")
        f.write(f"Scenes: {', '.join(scenes)} · seeds {seeds} · "
                f"budget 5 · generated by experiments/analyze_keyframe_gl.py\n\n")
        f.write("## Ladder (3-seed means)\n\n```\n")
        f.write("\n".join(lines))
        f.write("\n```\n\n## Per scene\n\n```\n")
        f.write(f"{'scene':14s}"
                + "".join(f"{m[:12]:>13s}" for m in methods)
                + f"{'all25':>9s}{'oracle5':>9s}\n")
        f.write("\n".join(scene_lines))
        f.write("\n```\n\n## Diagnostics\n\n")
        f.write("First frames picked (kf_*):\n\n")
        for m in sorted(first_frames):
            f.write(f"- `{m}`: {sorted(set(first_frames[m]))}\n")
        if comp:
            f.write("\nMean percentile rank of the picked frame "
                    "(rounds 2+):\n\n")
            for m in sorted(comp):
                parts = ", ".join(f"{c}={v:.3f}"
                                  for c, v in sorted(comp[m].items()))
                f.write(f"- `{m}`: {parts}\n")
        f.write("\n## Pilot judgment (spec §13) — fill after review\n\n"
                "- [ ] kf_l_dir_gseed > dopt_seq_dir → frame-0 seed was the "
                "failure cause\n"
                "- [ ] kf_g_dir recovers failure scenes → global context "
                "carries seed value\n"
                "- [ ] kf_gl_dir > kf_g_dir → local re-observation state adds "
                "value\n"
                "- [ ] kf_glu_dir > kf_gl_dir → unresolved coverage adds "
                "value\n"
                "- [ ] kf_glu_dir > kf_glu_nodir → 3×3 direction block "
                "re-confirmed\n\n"
                "GO conditions: ≥2/3 failure scenes improved over the 4th "
                "selector; 5-scene paired mean above uniform AND the 4th; "
                "oracle/anchor overlap increased; Zen intact — all on 3-seed "
                "means. **Full 10-scene sweep only after explicit "
                "approval.**\n")
    print(f"\nreport written: {os.path.relpath(REPORT_MD, REPO)}")


if __name__ == "__main__":
    main()
