"""Builds the two offline HTML pages of this package from the saved results in runs/:

  visualiser.html    interactive explorer of every saved AUROC result (overview, experiments, per category,
                     branch AUROC, localization, X3, architecture, provenance)
  architecture.html  the architecture block diagrams on their own page (printable)

  .venv\\Scripts\\python.exe scripts\\build_visualiser.py

Everything is read from runs/ (AUROC tables, per-image scores and labels, localization records). Every AUROC is
recomputed from the saved per-image scores and must equal the saved table value. Both pages are self-contained: no network access, no external libraries.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "src"))
from hybridad.metrics import auroc  # noqa: E402
from hybridad.prereg import code_hash  # noqa: E402

DATASETS = {
    "mvtec_ad": {"label": "MVTec AD", "categories": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
                                                    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
                                                    "wood", "zipper"],
                 "textures": ["carpet", "grid", "leather", "tile", "wood"]},
    "mvtec_loco": {"label": "MVTec LOCO AD", "categories": ["breakfast_box", "juice_bottle", "pushpins", "screw_bag",
                                                           "splicing_connectors"], "textures": []},
}
DESC = {
    "A_patchcore": ("PatchCore", "–"), "B_local_st": ("local student–teacher", "–"), "C_dino": ("DINOv2 reconstruction", "–"),
    "D_pc_st": ("PatchCore + local ST", "max"), "E_pc_dino": ("PatchCore + DINO", "max"),
    "F_three": ("PatchCore + local ST + DINO", "max"), "G_full": ("all four scores (full model)", "max"),
    "H_composition": ("composition only (diagnostic)", "–"), "G_full_mean": ("full model", "mean"),
    "G_full_or": ("full model", "OR rule"), "G_full_softmax": ("full model, softmax attention", "max"),
    "G_full_identity": ("full model, identity attention", "max"), "G_full_compact": ("full model, compact head", "max"),
    "C_dino_softmax": ("DINO, softmax attention", "–"), "C_dino_identity": ("DINO, identity attention", "–"),
    "C_dino_compact": ("DINO, compact head", "–"), "C_dino_pool_ref": ("DINO, reference pooling", "–"),
    "C_dino_pool_top0.5": ("DINO, top-0.5 % pooling", "–"), "C_dino_pool_top5": ("DINO, top-5 % pooling", "–"),
    "B_local_st_top1": ("local ST, top-1 % pooling", "–"), "B_local_st_nopenalty": ("local ST, no penalty data", "–"),
    "A_patchcore_reweight": ("PatchCore, paper re-weighting", "–"),
    "X3_mean": ("X3: + high-resolution PatchCore and composition", "mean"),
    "X3_max": ("X3: + high-resolution PatchCore and composition", "max"),
}
ORDER = list(DESC)
SLOTS = ["patchcore", "local_st", "global_attention", "composition"]


def r4(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 6)


# ---------------------------------------------------------------- results
def experiment_entry(exp, ds, root, group):
    d = root / exp / ds / "seed0"
    agg = json.loads((d / "aggregate.json").read_text())
    if not agg.get("categories_evaluated"):
        return None
    cm = pd.read_csv(d / "category_metrics.csv").set_index("category")
    expected = DATASETS[ds]["categories"]
    missing = agg.get("categories_missing") or []
    na = [c for c in missing if exp == "H_composition" and c in DATASETS[ds]["textures"]]
    e = {"id": exp, "desc": DESC.get(exp, (exp, "?"))[0], "fusion": DESC.get(exp, ("", "?"))[1], "group": group,
         "status": "confirmatory" if agg["evaluation_status"]["confirmatory"] else "EXPLORATORY",
         "n_eval": len(agg["categories_evaluated"]), "n_expected": len(expected) - len(na), "n_na": len(na),
         "macro": {"auroc": r4(agg.get("macro_auroc")), "pooled": r4(agg.get("pooled_auroc_of_fused_scores")),
                   "logical": r4(agg.get("macro_auroc_logical")), "structural": r4(agg.get("macro_auroc_structural")),
                   "balanced": r4(agg.get("macro_auroc_balanced_logical_structural")),
                   "br": {s: r4(agg.get(f"macro_branch_auroc__{s}")) for s in SLOTS}},
         "cats": {}}
    e["partial"] = e["n_eval"] < e["n_expected"]
    preds = pd.read_csv(d / "predictions.csv")
    for cat, row in cm.iterrows():
        c = {"auroc": r4(row["auroc"]), "lo": r4(row["auroc_ci_lo"]), "hi": r4(row["auroc_ci_hi"]),
             "n_normal": int(row["n_normal"]), "n_anom": int(row["n_anomalous"]),
             "br": {s: r4(row.get(f"branch_auroc__{s}")) for s in SLOTS}}
        if "auroc_logical" in row:
            c["logical"], c["structural"] = r4(row["auroc_logical"]), r4(row["auroc_structural"])
        p = preds[preds["category"] == cat]
        y = p["label"].to_numpy().astype(int)
        subsets = {"all": np.ones(len(p), bool)}
        if ds == "mvtec_loco":
            for sub in ("logical", "structural"):
                subsets[sub] = (y == 0) | (p["subgroup"] == sub).to_numpy()
        br_sub = {}
        for name, sel in subsets.items():
            ys = y[sel]
            fused = p["fused_score"].to_numpy()[sel]
            a = auroc(ys, fused)[0]
            ref = float(row["auroc"] if name == "all" else row[f"auroc_{name}"])
            assert abs(a - ref) < 1e-12, (exp, ds, cat, name, a, ref)   # recomputed AUROC equals the saved table value
            for s in SLOTS:
                col = f"score_{s}"
                if col in p and p[col].notna().all():
                    v = p[col].to_numpy()[sel]
                    br_sub.setdefault(name, {})[s] = r4(auroc(ys, v)[0])
        c["br_sub"] = br_sub          # branch AUROC per anomaly subset, computed from the saved per-image scores
        e["cats"][cat] = c
    return e


def localization(ds):
    out = {}
    for cat in DATASETS[ds]["categories"]:
        p = RUNS / "G_full" / ds / cat / "seed0" / "localization.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if ds == "mvtec_ad":
            out[cat] = {"pixel_auroc": r4(d.get("pixel_auroc")), "au_pro": r4(d.get("au_pro_0.3"))}
        else:
            sp = d.get("au_spro_by_integration_limit") or {}
            lims = sorted(sp.get("mean", {}), key=float)
            out[cat] = {"limits": [float(x) for x in lims],
                        **{k: [r4(sp[key][x]) for x in lims] for k, key in
                           (("mean", "mean"), ("logical", "logical_anomalies"), ("structural", "structural_anomalies"))}}
    return out


def build_data():
    data = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "datasets": {}}
    for ds, meta in DATASETS.items():
        exps = []
        for exp in ORDER:
            root = RUNS / "exploratory" if exp.startswith("X3") else RUNS
            if (root / exp / ds / "seed0" / "aggregate.json").exists():
                e = experiment_entry(exp, ds, root, "exploratory" if exp.startswith("X3") else "pre-registered")
                if e:
                    exps.append(e)
        data["datasets"][ds] = {**meta, "experiments": exps, "localization": localization(ds)}
    pre = json.loads((RUNS / "PREREGISTRATION.json").read_text())
    acc = [json.loads(l) for l in (RUNS / "TEST_ACCESS_LOG_excerpt.jsonl").read_text().splitlines() if l.strip()]
    data["provenance"] = {"prereg_hash": pre["code_hash"], "prereg_created": pre["created_utc"][:16].replace("T", " "),
                          "package_hash": code_hash(),
                          "access": {"total": len(acc), "confirmatory": sum(a["confirmatory"] for a in acc),
                                     "exploratory": sum(not a["confirmatory"] for a in acc)}}
    return data


# ---------------------------------------------------------------- architecture diagrams (inline SVG)
def _box(x, y, w, h, kind, lines, rx=6):
    """kind: frozen | learned | plain | out. lines: [(text, 'title'|'body')]; text is centred."""
    n = len(lines)
    lh = 15
    y0 = y + h / 2 - (n - 1) * lh / 2 + 4
    t = "".join(f'<tspan x="{x + w / 2:.0f}" y="{y0 + i * lh:.0f}" class="t-{k}">{s}</tspan>' for i, (s, k) in enumerate(lines))
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="b-{kind}"/><text text-anchor="middle">{t}</text>'


def _arrow(x1, y1, x2, y2, mid, label=None, lx=None, ly=None):
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" marker-end="url(#{mid})"/>'
    if label:
        s += f'<text x="{lx}" y="{ly}" class="t-arrow" text-anchor="middle">{label}</text>'
    return s


def _path(d, mid=None):
    return f'<path d="{d}" class="ln" fill="none"' + (f' marker-end="url(#{mid})"' if mid else "") + "/>"


def _defs(mid):
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mk"/></marker></defs>')


def _legend(x, y):
    items = [("frozen", "pretrained, frozen (never updated)"), ("learned", "fitted on normal images of the category"),
             ("plain", "fixed computation"), ("out", "score / result")]
    s, cx = "", x
    for kind, label in items:
        s += f'<rect x="{cx}" y="{y}" width="26" height="16" rx="4" class="b-{kind}"/>'
        s += f'<text x="{cx + 34}" y="{y + 12}" class="t-legend">{label}</text>'
        cx += 34 + 7.2 * len(label) + 26
    return s


def figure_main():
    mid = "arrA"
    rows = [
        ("PatchCore", ["AD: 256, crop 224", "LOCO: 256 letterbox"],
         ["WideResNet-50-2", "layers 2+3, 3×3 patches", "→ 1024-d per patch"],
         ["Coreset memory", "10% of normal patches", "(no gradient training)"],
         ["nearest-neighbour", "distance per patch", "→ anomaly map"], ["max over", "patches"]),
        ("Local student–teacher", ["AD: 256 × 256", "LOCO: 256 letterbox"],
         ["PDN-S teacher", "33×33 receptive field", "channel-normalized"],
         ["PDN-S student", "trained to imitate the", "teacher (70k steps)"],
         ["‖teacher − student‖²", "channel mean", "→ anomaly map"], ["max over", "pixels"]),
        ("DINO reconstruction", ["AD: 448 × 448", "LOCO: 448 letterbox"],
         ["DINOv2 ViT-B/14 + reg.", "layers 2–9", "32×32 tokens"],
         ["bottleneck + 8-block", "linear-attention decoder", "(5k iterations)"],
         ["1 − cosine(encoder,", "reconstruction)", "→ anomaly map"], ["mean of", "top 1%"]),
        ("Composition", ["objects only", "448 / 448 letterbox"],
         ["DINOv2 final-layer", "patch tokens", "→ PCA-64"],
         ["10 visual clusters;", "mass, appearance, layout,", "adjacency; normal memory"],
         ["distance to the 3", "nearest normal images", "(no map)"], ["joint", "kNN score"]),
    ]
    X = {"in": (16, 112), "pre": (160, 164), "fro": (340, 170), "lea": (526, 190), "evi": (732, 160), "pool": (908, 96),
         "nor": (1020, 120), "fuse": (1172, 104)}
    ys = [70, 172, 274, 376]
    h = 82
    s = [_defs(mid)]
    heads = [("pre", "1  Preprocess"), ("fro", "2  Frozen backbone"), ("lea", "3  Fitted on normals"),
             ("evi", "4  Anomaly evidence"), ("pool", "5  Image score"), ("nor", "6  Normalize"), ("fuse", "7  Fuse")]
    for k, t in heads:
        x, w = X[k]
        s.append(f'<text x="{x + w / 2}" y="44" class="t-head" text-anchor="middle">{t}</text>')
    xi, wi = X["in"]
    s.append(_box(xi, 196, wi, 140, "plain", [("Test image", "title"), ("category known:", "body"),
                                               ("routes to that", "body"), ("category's models", "body")]))
    bus = xi + wi + 16
    s.append(_path(f"M{xi + wi},266 H{bus}"))
    s.append(_path(f"M{bus},{ys[0] + h / 2} V{ys[3] + h / 2}"))
    for (name, pre, fro, lea, evi, pool), y in zip(rows, ys):
        cy = y + h / 2
        s.append(_arrow(bus, cy, X["pre"][0], cy, mid))
        s.append(_box(*[X["pre"][0], y, X["pre"][1], h], "plain", [(name, "title")] + [(t, "body") for t in pre]))
        s.append(_arrow(X["pre"][0] + X["pre"][1], cy, X["fro"][0], cy, mid))
        s.append(_box(X["fro"][0], y, X["fro"][1], h, "frozen", [(fro[0], "title")] + [(t, "body") for t in fro[1:]]))
        s.append(_arrow(X["fro"][0] + X["fro"][1], cy, X["lea"][0], cy, mid))
        s.append(_box(X["lea"][0], y, X["lea"][1], h, "learned", [(lea[0], "title")] + [(t, "body") for t in lea[1:]]))
        s.append(_arrow(X["lea"][0] + X["lea"][1], cy, X["evi"][0], cy, mid))
        s.append(_box(X["evi"][0], y, X["evi"][1], h, "plain", [(t, "body") for t in evi]))
        s.append(_arrow(X["evi"][0] + X["evi"][1], cy, X["pool"][0], cy, mid))
        s.append(_box(X["pool"][0], y, X["pool"][1], h, "plain", [(t, "body") for t in pool]))
        s.append(_arrow(X["pool"][0] + X["pool"][1], cy, X["nor"][0], cy, mid))
        s.append(_box(X["nor"][0], y, X["nor"][1], h, "learned", [("z = (s − m)", "title"), ("÷ (Q95 − m)", "title"),
                                                                   ("held-out normals", "body")]))
    fb = X["fuse"][0] - 12
    for y in ys:
        s.append(_path(f"M{X['nor'][0] + X['nor'][1]},{y + h / 2} H{fb}"))
    s.append(_path(f"M{fb},{ys[0] + h / 2} V{ys[3] + h / 2}"))
    fx, fw = X["fuse"]
    s.append(_arrow(fb, 241, fx, 241, mid))
    s.append(_box(fx, 196, fw, 90, "plain", [("max", "title"), ("(G_full)", "body"), ("mean / OR:", "body"),
                                             ("ablations", "body")]))
    s.append(_arrow(fx + fw / 2, 286, fx + fw / 2, 318, mid))
    s.append(_box(fx, 318, fw, 60, "out", [("anomaly", "title"), ("score", "title")]))
    s.append(_arrow(fx + fw / 2, 378, fx + fw / 2, 410, mid))
    s.append(_box(fx, 410, fw, 48, "out", [("image AUROC", "title"), ("vs test labels", "body")]))
    s.append(_legend(160, 488))
    return ('<svg viewBox="0 0 1292 520" role="img" class="arch" aria-label="Architecture: a test image is routed to four '
            'branches (PatchCore, local student-teacher, DINO reconstruction, composition); each preprocesses the image, '
            'applies a frozen pretrained backbone and a component fitted on normal images of the category, produces anomaly '
            'evidence and an image score; scores are normalized with held-out normal images and fused by maximum into one '
            'anomaly score, evaluated by image AUROC.">' + "".join(s) + "</svg>")


def figure_data_roles():
    mid = "arrB"
    s = [_defs(mid)]
    s.append(_box(16, 30, 200, 120, "plain", [("Normal images", "title"), ("MVTec AD: train/good", "body"),
                                              ("MVTec LOCO: train/good +", "body"), ("validation/good", "body")]))
    rows = [(20, "fit", "train every branch: memories,", "student, decoder, clusters"),
            (78, "calib_scale", "fit the score normalizers", "(median, Q95 per branch)"),
            (136, "calib_threshold", "OR-rule offsets only", "(unused by max fusion)")]
    for y, role, a, b in rows:
        s.append(_path(f"M216,90 C250,90 250,{y + 22} 284,{y + 22}", mid))
        s.append(_box(284, y, 150, 44, "learned", [(role, "title")]))
        s.append(_arrow(434, y + 22, 470, y + 22, mid))
        s.append(_box(470, y, 230, 44, "plain", [(a, "body"), (b, "body")]))
    s.append(_box(752, 30, 196, 120, "plain", [("Test images", "title"), ("normal + anomalous", "body"),
                                               ("(official test split)", "body")]))
    s.append(_arrow(948, 65, 1040, 65, mid, "fixed models", 994, 56))
    s.append(_box(1040, 40, 236, 50, "plain", [("per-image scores", "title"), ("and maps (no labels)", "body")]))
    s.append(_arrow(1158, 90, 1158, 120, mid))
    s.append(_box(1040, 120, 236, 50, "out", [("AUROC: labels used", "title"), ("only at this step", "body")]))
    return ('<svg viewBox="0 0 1292 196" role="img" class="arch" aria-label="Data roles: normal training images are split '
            'into fit (trains every branch), calib_scale (fits score normalizers) and calib_threshold (OR-rule offsets '
            'only); test images are scored by the fixed models and labels are used only to compute AUROC.">'
            + "".join(s) + "</svg>")


def figure_localization():
    mid = "arrC"
    s = [_defs(mid)]
    s.append(_box(16, 24, 190, 120, "plain", [("Anomaly maps", "title"), ("PatchCore", "body"), ("local student–teacher", "body"),
                                              ("DINO reconstruction", "body"), ("(composition has none)", "body")]))
    steps = [(250, 190, "plain", [("resample to the", "body"), ("original pixels", "body"), ("(recorded geometry)", "body")]),
             (476, 190, "learned", [("pixel normalizer", "title"), ("median, Q95 of", "body"), ("normal calib maps", "body")]),
             (702, 150, "plain", [("fuse maps", "title"), ("max (G_full)", "body")])]
    prev = 206
    for x, w, kind, lines in steps:
        s.append(_arrow(prev, 84, x, 84, mid))
        s.append(_box(x, 44, w, 80, kind, lines))
        prev = x + w
    s.append(_path(f"M{prev},84 C{prev + 30},84 {prev + 30},44 {prev + 60},44", mid))
    s.append(_path(f"M{prev},84 C{prev + 30},84 {prev + 30},124 {prev + 60},124", mid))
    s.append(_box(prev + 60, 16, 370, 56, "out", [("MVTec AD: pixel AUROC and AU-PRO@0.3", "title"),
                                                  ("re-implementation, checked by brute force", "body")]))
    s.append(_box(prev + 60, 96, 370, 56, "out", [("MVTec LOCO: AU-sPRO (official evaluator v2.0)", "title"),
                                                  ("TIFF maps written one image at a time", "body")]))
    return ('<svg viewBox="0 0 1292 168" role="img" class="arch" aria-label="Localization: the PatchCore, student-teacher '
            'and DINO maps are resampled to original pixels, normalized with normal calibration maps, fused by maximum, and '
            'scored by pixel AUROC and AU-PRO (MVTec AD) or AU-sPRO with the official evaluator (MVTec LOCO).">'
            + "".join(s) + "</svg>")


def figure_x3():
    mid = "arrD"
    s = [_defs(mid)]
    rows = [("PatchCore (as in G_full)", "plain"), ("PatchCore-HR: 512 letterbox, 1% coreset", "learned"),
            ("local student–teacher (as in G_full)", "plain"), ("DINO reconstruction (as in G_full)", "plain"),
            ("Composition-HR: 896 letterbox, 64×64 grid, stable counts", "learned")]
    ys = [14, 56, 98, 140, 182]
    s.append(_box(16, 86, 150, 70, "plain", [("LOCO image", "title")]))
    s.append(_path("M166,121 H190"))
    s.append(_path(f"M190,{ys[0] + 17} V{ys[-1] + 17}"))
    for (label, kind), y in zip(rows, ys):
        s.append(_arrow(190, y + 17, 214, y + 17, mid))
        s.append(_box(214, y, 420, 34, kind, [(label, "body")]))
        s.append(_path(f"M634,{y + 17} H700"))
    s.append(_path(f"M700,{ys[0] + 17} V{ys[-1] + 17}"))
    s.append(_arrow(700, 121, 740, 121, mid))
    s.append(_box(740, 91, 190, 60, "learned", [("normalize each score", "title"), ("(held-out normals)", "body")]))
    s.append(_arrow(930, 121, 970, 121, mid))
    s.append(_box(970, 91, 130, 60, "plain", [("mean", "title"), ("(X3_max: max)", "body")]))
    s.append(_arrow(1100, 121, 1140, 121, mid))
    s.append(_box(1140, 86, 136, 70, "out", [("X3 score", "title"), ("EXPLORATORY", "body")]))
    return ('<svg viewBox="0 0 1292 232" role="img" class="arch" aria-label="X3, exploratory: five LOCO branch scores - '
            'PatchCore, a 512-pixel PatchCore, the student-teacher, DINO reconstruction and an 896-pixel composition with '
            'stable count evidence - are normalized and averaged (X3_mean) or maximized (X3_max).">' + "".join(s) + "</svg>")


FIGURES = [
    ("Architecture of the full model (G_full)", figure_main(),
     "A test image of a known category passes through four independent branches. Each one preprocesses the image its own "
     "way, extracts features with a frozen pretrained network, compares them with what it learned from normal images of "
     "that category, and reduces the evidence to one image score. Each score is normalized with held-out normal images, "
     "and the maximum of the four is the anomaly score."),
    ("Which images are used where", figure_data_roles(),
     "Only normal images are used before testing. Test labels are read at one place only: when the AUROC is computed."),
    ("Localization path", figure_localization(),
     "The three branches that produce spatial maps are combined the same way at pixel level. Composition is image-level "
     "evidence only and is left out."),
    ("X3 (exploratory, MVTec LOCO)", figure_x3(),
     "X3 adds a higher-resolution PatchCore and replaces composition with a higher-resolution version that uses part counts "
     "where they are stable on normal images. It was designed after the LOCO test results were seen, so it is exploratory."),
]

ARCH_CSS = """
svg.arch{width:100%;height:auto;display:block;color:var(--ink)}
svg.arch .b-frozen{fill:var(--frozen-fill);stroke:var(--muted);stroke-width:1.2}
svg.arch .b-learned{fill:var(--accent-wash);stroke:var(--accent);stroke-width:1.4}
svg.arch .b-plain{fill:var(--surface);stroke:var(--axis);stroke-width:1.2}
svg.arch .b-out{fill:var(--surface);stroke:var(--ink);stroke-width:1.6}
svg.arch text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink)}
svg.arch .t-title{font-size:12.5px;font-weight:600}
svg.arch .t-body{font-size:11.5px;fill:var(--ink2)}
svg.arch .t-head{font-size:12px;font-weight:600;fill:var(--ink2);letter-spacing:.02em}
svg.arch .t-arrow,svg.arch .t-legend{font-size:11.5px;fill:var(--ink2)}
svg.arch .ln{stroke:var(--ink2);stroke-width:1.4}
svg.arch .mk{fill:var(--ink2)}
"""


def figures_html(numbered=True):
    out = []
    for i, (title, svg, caption) in enumerate(FIGURES, 1):
        out.append(f'<figure class="arch-fig"><h3>{"Figure " + str(i) + ". " if numbered else ""}{title}</h3>{svg}'
                   f'<figcaption>{caption}</figcaption></figure>')
    return "\n".join(out)


def main():
    data = build_data()
    tpl = (ROOT / "scripts" / "visualiser_template.html").read_text(encoding="utf-8")
    page = (tpl.replace("/*__ARCH_CSS__*/", ARCH_CSS).replace("<!--__ARCH_FIGURES__-->", figures_html())
            .replace("__GENERATED__", data["generated"])
            .replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")))
    (ROOT / "visualiser.html").write_text(page, encoding="utf-8")
    arch = (tpl.split("<!--__ARCH_PAGE_SPLIT__-->")[0]   # shared <head> with tokens and styles
            .replace("/*__ARCH_CSS__*/", ARCH_CSS).replace("<title>Kutty Dinosaur results</title>",
                                                           "<title>Kutty Dinosaur architecture</title>"))
    arch += ('<body><main class="page arch-page"><header class="top"><div><h1>Kutty Dinosaur architecture</h1>'
             '<p class="sub">How an image becomes an anomaly score: four independent branches, per-branch normalization '
             'and fusion. Read each diagram left to right.</p></div></header>' + figures_html() +
             '<p class="foot">Generated by scripts/build_visualiser.py from the package configuration. Interactive results: '
             'visualiser.html.</p></main></body></html>')
    (ROOT / "architecture.html").write_text(arch, encoding="utf-8")
    size = (ROOT / "visualiser.html").stat().st_size / 1e6
    print(f"wrote visualiser.html ({size:.2f} MB) and architecture.html; "
          f"experiments: " + ", ".join(f"{ds}={len(v['experiments'])}" for ds, v in data["datasets"].items()))


if __name__ == "__main__":
    main()
