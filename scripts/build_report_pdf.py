r"""Rebuilds 'kutty dinosaur.pdf' (AUROC report) from the saved results in runs/. HTML -> PDF with Microsoft Edge (Windows).

  .venv\Scripts\python.exe scripts\build_report_pdf.py
"""
import base64
import html
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
SCRATCH = Path(tempfile.gettempdir())
OUT_HTML = SCRATCH / "kutty_dinosaur_report.html"
OUT_PDF = PKG / "kutty dinosaur.pdf"
EDGE = next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                         r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).exists()), "msedge")
sys.path.insert(0, str(PKG / "src"))
from hybridad.prereg import code_hash  # noqa: E402

RUNS = PKG / "runs"
AD_CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile",
           "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
LOCO_CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
BR = [("patchcore", "PatchCore"), ("local_st", "local ST"), ("global_attention", "DINO"), ("composition", "composition")]
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
}
ORDER = list(DESC)


def f3(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def esc(s):
    return html.escape(str(s))


def agg(exp, ds, root=RUNS):
    p = root / exp / ds / "seed0" / "aggregate.json"
    return json.loads(p.read_text()) if p.exists() else None


def cm(exp, ds, root=RUNS):
    return pd.read_csv(root / exp / ds / "seed0" / "category_metrics.csv").set_index("category")


def table(headers, rows, cls="num"):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def fig_to_img(fig, alt, width="100%"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="{esc(alt)}" style="width:{width}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def file_img(p, alt, width="100%"):
    return f'<img alt="{esc(alt)}" style="width:{width}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def experiments(ds, cats):
    full, partial = [], []
    for exp in ORDER:
        a = agg(exp, ds)
        if not a:
            continue
        n = len(a.get("categories_evaluated") or [])
        if n == len(cats) and not a.get("partial_run"):
            full.append((exp, a))
        elif exp == "H_composition" and set(a.get("categories_missing") or []) == TEXTURES and n == len(cats) - 5:
            full.append((exp, a))
        elif n > 0:
            partial.append((exp, n))
    return full, partial


g_ad, g_loco = agg("G_full", "mvtec_ad"), agg("G_full", "mvtec_loco")
x3m, x3x = agg("X3_mean", "mvtec_loco", RUNS / "exploratory"), agg("X3_max", "mvtec_loco", RUNS / "exploratory")
ad_full, ad_partial = experiments("mvtec_ad", AD_CATS)
loco_full, loco_partial = experiments("mvtec_loco", LOCO_CATS)
loc_ad = {c: json.loads((RUNS / f"G_full/mvtec_ad/{c}/seed0/localization.json").read_text()) for c in AD_CATS}
loc_loco = {c: json.loads((RUNS / f"G_full/mvtec_loco/{c}/seed0/localization.json").read_text()) for c in LOCO_CATS}
access = [json.loads(l) for l in (RUNS / "TEST_ACCESS_LOG_excerpt.jsonl").read_text().splitlines() if l.strip()]
prereg = json.loads((RUNS / "PREREGISTRATION.json").read_text())
hash_pkg = code_hash()
cA, cG = cm("A_patchcore", "mvtec_ad"), cm("G_full", "mvtec_ad")
lA, lG, lX = cm("A_patchcore", "mvtec_loco"), cm("G_full", "mvtec_loco"), cm("X3_mean", "mvtec_loco", RUNS / "exploratory")

# ------------------------------------------------------------------ charts
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C_A, C_G, C_X = "#9aa5b1", "#1f4e79", "#c07a12"

fig, ax = plt.subplots(figsize=(7.2, 2.3))
x = np.arange(len(AD_CATS)); w = 0.38
ax.bar(x - w / 2, [cA.loc[c, "auroc"] for c in AD_CATS], w, color=C_A, label="A: PatchCore alone")
ax.bar(x + w / 2, [cG.loc[c, "auroc"] for c in AD_CATS], w, color=C_G, label="G: full model")
ax.errorbar(x + w / 2, [cG.loc[c, "auroc"] for c in AD_CATS],
            yerr=[[cG.loc[c, "auroc"] - cG.loc[c, "auroc_ci_lo"] for c in AD_CATS],
                  [cG.loc[c, "auroc_ci_hi"] - cG.loc[c, "auroc"] for c in AD_CATS]], fmt="none", ecolor="black", lw=0.7, capsize=1.5)
ax.set_xticks(x, [c.replace("_", " ") for c in AD_CATS], rotation=35, ha="right")
ax.set_ylim(0.90, 1.005); ax.set_ylabel("image AUROC")
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
chart_ad = fig_to_img(fig, "MVTec AD per-category AUROC")

fig, ax = plt.subplots(figsize=(7.2, 2.4))
x = np.arange(len(LOCO_CATS)); w = 0.27
ax.bar(x - w, [lA.loc[c, "auroc"] for c in LOCO_CATS], w, color=C_A, label="A: PatchCore alone")
ax.bar(x, [lG.loc[c, "auroc"] for c in LOCO_CATS], w, color=C_G, label="G: full model (confirmatory)")
ax.errorbar(x, [lG.loc[c, "auroc"] for c in LOCO_CATS],
            yerr=[[lG.loc[c, "auroc"] - lG.loc[c, "auroc_ci_lo"] for c in LOCO_CATS],
                  [lG.loc[c, "auroc_ci_hi"] - lG.loc[c, "auroc"] for c in LOCO_CATS]], fmt="none", ecolor="black", lw=0.7, capsize=2)
ax.bar(x + w, [lX.loc[c, "auroc"] for c in LOCO_CATS], w, color="white", edgecolor=C_X, hatch="////",
       label="X3_mean (EXPLORATORY)")
ax.set_xticks(x, [c.replace("_", " ") for c in LOCO_CATS])
ax.set_ylim(0.6, 1.0); ax.set_ylabel("image AUROC")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
chart_loco = fig_to_img(fig, "MVTec LOCO per-category AUROC")

fig, ax = plt.subplots(figsize=(7.2, 2.3))
x = np.arange(len(LOCO_CATS)); w = 0.2
for i, (vals, lab, col, hatch) in enumerate([
        ([lG.loc[c, "auroc_logical"] for c in LOCO_CATS], "G: logical", C_G, None),
        ([lG.loc[c, "auroc_structural"] for c in LOCO_CATS], "G: structural", "#6f9fd8", None),
        ([lX.loc[c, "auroc_logical"] for c in LOCO_CATS], "X3_mean: logical (EXPL.)", C_X, "////"),
        ([lX.loc[c, "auroc_structural"] for c in LOCO_CATS], "X3_mean: structural (EXPL.)", "#e9c27a", "////")]):
    ax.bar(x + (i - 1.5) * w, vals, w, color=col if hatch is None else "white", edgecolor=col, hatch=hatch, label=lab)
ax.set_xticks(x, [c.replace("_", " ") for c in LOCO_CATS]); ax.set_ylim(0.6, 1.0); ax.set_ylabel("AUROC")
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
chart_ls = fig_to_img(fig, "MVTec LOCO logical vs structural AUROC")


def heatmap(cmf, cats, title_cols, alt, height):
    cols = ["auroc"] + [f"branch_auroc__{b}" for b, _ in BR]
    M = np.array([[cmf.loc[c, k] if k in cmf.columns else np.nan for k in cols] for c in cats], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, height))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="Blues", vmin=0.6, vmax=1.0, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, "–" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                color="white" if (not np.isnan(v) and v > 0.9) else "black")
    ax.set_xticks(range(len(cols)), title_cols, fontsize=7)
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats], fontsize=7)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    return fig_to_img(fig, alt, "78%")


hm_titles = ["fused (G)"] + [n for _, n in BR]
heat_ad = heatmap(cG, AD_CATS, hm_titles, "MVTec AD AUROC by branch", 3.6)
heat_loco = heatmap(lG, LOCO_CATS, hm_titles, "MVTec LOCO AUROC by branch", 1.6)
roc_loco = file_img(RUNS / "G_full/mvtec_loco/seed0/figures/roc_curves.png", "G_full LOCO ROC curves")
roc_ad = file_img(RUNS / "G_full/mvtec_ad/seed0/figures/roc_curves.png", "G_full MVTec AD ROC curves")


# ------------------------------------------------------------------ tables
def exp_rows(full, loco=False):
    rows = []
    for exp, a in full:
        desc = DESC[exp][0] + (" — 10 object categories; n/a for textures" if exp == "H_composition" and not loco else "")
        r = [f"<b>{exp}</b>" if exp == "G_full" else exp, esc(desc), DESC[exp][1], f3(a["macro_auroc"]),
             f3(a.get("pooled_auroc_of_fused_scores"))]
        if loco:
            r += [f3(a.get("macro_auroc_logical")), f3(a.get("macro_auroc_structural"))]
        rows.append(r)
    return rows


ad_exp_tbl = table(["experiment", "scores", "fusion", "macro AUROC", "pooled AUROC"], exp_rows(ad_full))
loco_exp_tbl = table(["experiment", "scores", "fusion", "macro AUROC", "pooled AUROC", "logical AUROC", "structural AUROC"],
                     exp_rows(loco_full, True))


def ci(r):
    return f"{f3(r.auroc)} <span class='ci'>[{f3(r.auroc_ci_lo)}, {f3(r.auroc_ci_hi)}]</span>"


ad_rows = [[c.replace("_", " "), ci(cG.loc[c]), f3(loc_ad[c]["pixel_auroc"]), f3(loc_ad[c]["au_pro_0.3"])] for c in AD_CATS]
ad_rows.append(["<b>macro mean</b>", f"<b>{f3(g_ad['macro_auroc'])}</b>",
                f"<b>{f3(np.mean([loc_ad[c]['pixel_auroc'] for c in AD_CATS]))}</b>",
                f"<b>{f3(np.mean([loc_ad[c]['au_pro_0.3'] for c in AD_CATS]))}</b>"])
ad_cat_tbl = table(["category", "image AUROC [95% CI]", "pixel AUROC", "AU-PRO@0.3"], ad_rows)

loco_rows = []
for c in LOCO_CATS:
    r, L = lG.loc[c], loc_loco[c]["au_spro_by_integration_limit"]
    loco_rows.append([c.replace("_", " "), ci(r), f3(r.auroc_logical), f3(r.auroc_structural), f3(r.auroc_balanced_logical_structural),
                      f"{f3(L['logical_anomalies']['0.05'])} / {f3(L['structural_anomalies']['0.05'])} / {f3(L['mean']['0.05'])}"])
loco_rows.append(["<b>macro mean</b>", f"<b>{f3(g_loco['macro_auroc'])}</b>", f"<b>{f3(g_loco['macro_auroc_logical'])}</b>",
                  f"<b>{f3(g_loco['macro_auroc_structural'])}</b>", f"<b>{f3(g_loco['macro_auroc_balanced_logical_structural'])}</b>",
                  f"<b>{f3(np.mean([loc_loco[c]['au_spro_by_integration_limit']['mean']['0.05'] for c in LOCO_CATS]))}</b>"])
loco_cat_tbl = table(["category", "image AUROC [95% CI]", "logical", "structural", "balanced", "AU-sPRO@0.05 (log. / str. / mean)"], loco_rows)

x3_rows = []
for lab, a, st in (("G_full", g_loco, "confirmatory"), ("G_full_mean", agg("G_full_mean", "mvtec_loco"), "pre-registered ablation"),
                   ("<b>X3_mean</b> (primary)", x3m, "<span class='tag'>EXPLORATORY</span>"),
                   ("X3_max", x3x, "<span class='tag'>EXPLORATORY</span>")):
    x3_rows.append([lab, st, f3(a["macro_auroc"]), f3(a["pooled_auroc_of_fused_scores"]), f3(a["macro_auroc_logical"]),
                    f3(a["macro_auroc_structural"]), f3(a["macro_auroc_balanced_logical_structural"])])
x3_tbl = table(["model", "status", "macro AUROC", "pooled AUROC", "logical", "structural", "balanced"], x3_rows, "num")
x3_cat_tbl = table(["category", "image AUROC [95% CI]", "logical", "structural", "Δ AUROC vs G_full"],
                   [[c.replace("_", " "), ci(lX.loc[c]), f3(lX.loc[c, "auroc_logical"]), f3(lX.loc[c, "auroc_structural"]),
                     f"{lX.loc[c, 'auroc'] - lG.loc[c, 'auroc']:+.3f}"] for c in LOCO_CATS])

headline = table(["dataset", "model", "status", "macro AUROC", "pooled AUROC", "logical / structural AUROC"], [
    ["MVTec AD (15 categories)", "<b>G_full</b>", "confirmatory", f"<b>{f3(g_ad['macro_auroc'])}</b>",
     f3(g_ad["pooled_auroc_of_fused_scores"]), "–"],
    ["MVTec LOCO (5 categories)", "<b>G_full</b>", "confirmatory", f"<b>{f3(g_loco['macro_auroc'])}</b>",
     f3(g_loco["pooled_auroc_of_fused_scores"]), f"{f3(g_loco['macro_auroc_logical'])} / {f3(g_loco['macro_auroc_structural'])}"],
    ["MVTec LOCO (5 categories)", "X3_mean", "<span class='tag'>EXPLORATORY</span>", f3(x3m["macro_auroc"]),
     f3(x3m["pooled_auroc_of_fused_scores"]), f"{f3(x3m['macro_auroc_logical'])} / {f3(x3m['macro_auroc_structural'])}"],
], "num l3")

n_ad = sum(1 for a in access if a["dataset"] == "mvtec_ad")
n_lc = sum(1 for a in access if a["dataset"] == "mvtec_loco" and a["confirmatory"])
n_lx = sum(1 for a in access if a["dataset"] == "mvtec_loco" and not a["confirmatory"])
partial_txt = "; ".join(f"{e} ({n}/15)" for e, n in ad_partial) or "none"
partial_loco_txt = "; ".join(f"{e} ({n}/5)" for e, n in loco_partial) or "none"
ad_obj = [c for c in AD_CATS if c not in TEXTURES]
tex_auc = [cG.loc[c, "auroc"] for c in AD_CATS if c in TEXTURES]
lowest_ad = cG["auroc"].idxmin()

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1c2430; font-size: 9.4pt; line-height: 1.42; }
h1 { font-family: Georgia, 'Times New Roman', serif; font-size: 30pt; margin: 0 0 4px; color: #0f2a44; letter-spacing: -0.5px; }
h2 { font-family: Georgia, serif; font-size: 15pt; color: #0f2a44; border-bottom: 1.5px solid #1f4e79; padding-bottom: 3px;
     margin: 22px 0 8px; break-after: avoid; }
h3 { font-size: 10.5pt; color: #1f4e79; margin: 14px 0 5px; break-after: avoid; }
p { margin: 5px 0 7px; }
.sub { font-size: 12pt; color: #34495e; margin: 2px 0; }
.meta { font-size: 8.6pt; color: #5b6b7b; margin-top: 8px; }
.cover { border-left: 5px solid #1f4e79; padding: 6px 0 6px 16px; margin: 6px 0 14px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.2pt; break-inside: avoid; }
th { background: #eef2f6; color: #1c2430; font-weight: 600; text-align: left; padding: 4px 5px; border-bottom: 1px solid #b8c4d0; }
td { padding: 3px 5px; border-bottom: 1px solid #e3e8ee; vertical-align: top; }
table.num td:not(:first-child), table.num th:not(:first-child) { text-align: right; }
table.num td:nth-child(2), table.num th:nth-child(2) { text-align: left; }
table.l3 td:nth-child(3), table.l3 th:nth-child(3) { text-align: left; }
.ci { color: #6b7a89; font-size: 7.4pt; }
.tag { background: #fff3dd; color: #8a5300; border: 1px solid #e0b060; border-radius: 3px; padding: 0 4px; font-size: 7.4pt;
       font-weight: 600; white-space: nowrap; }
.box { border: 1px solid #e0b060; background: #fffaf0; padding: 7px 10px; border-radius: 4px; margin: 8px 0; break-inside: avoid; }
ul { margin: 4px 0 8px 18px; padding: 0; } li { margin: 2px 0; }
code { font-family: Consolas, monospace; font-size: 8.2pt; background: #f1f3f5; padding: 0 3px; border-radius: 2px; }
img { display: block; margin: 4px auto 6px; }
.pb { break-before: page; }
.small { font-size: 8.2pt; color: #4a5866; }
.fig { break-inside: avoid; }
"""

body = f"""
<div class="cover">
<h1>Kutty Dinosaur</h1>
<div class="sub">Hybrid image-level anomaly detection on MVTec AD and MVTec LOCO AD: AUROC results</div>
<div class="sub" style="font-size:10pt">PatchCore + local student–teacher + DINOv2 reconstruction + composition evidence</div>
<div class="meta">Results snapshot of 28 September 2026 · seed 0 · results produced by the pre-registered pipeline (code hash
<code>{prereg['code_hash']}</code>) · all result tables and charts are generated from the saved outputs in <code>runs/</code></div>
</div>

<h2>1. Summary</h2>
{headline}
<ul>
<li><b>MVTec AD:</b> the pre-registered full model reaches macro image AUROC {f3(g_ad['macro_auroc'])} over 15 categories
(pooled {f3(g_ad['pooled_auroc_of_fused_scores'])}); the five texture categories reach {f3(min(tex_auc))}–{f3(max(tex_auc))}, and the
lowest category is {lowest_ad.replace('_', ' ')} ({f3(cG.loc[lowest_ad, 'auroc'])}). Localization: pixel AUROC
{f3(np.mean([loc_ad[c]['pixel_auroc'] for c in AD_CATS]))}, AU-PRO {f3(np.mean([loc_ad[c]['au_pro_0.3'] for c in AD_CATS]))}.</li>
<li><b>MVTec LOCO:</b> the pre-registered full model reaches macro AUROC {f3(g_loco['macro_auroc'])}: {f3(g_loco['macro_auroc_structural'])}
on structural and {f3(g_loco['macro_auroc_logical'])} on logical anomalies. The composition score adds most on logical anomalies
(breakfast box AUROC {f3(lA.loc['breakfast_box','auroc'])} for PatchCore alone vs {f3(lG.loc['breakfast_box','auroc'])} for the full model).</li>
<li><b>X3 (exploratory, LOCO):</b> higher-resolution composition and PatchCore branches raise macro AUROC to {f3(x3m['macro_auroc'])}
(structural {f3(x3m['macro_auroc_structural'])}, logical {f3(x3m['macro_auroc_logical'])}). X3 was designed after the LOCO test results were
seen, so it is reported separately and is not confirmatory.</li>
</ul>

<h2>2. Method</h2>
<p>Each image receives four independently observable anomaly scores:</p>
<ul>
<li><b>PatchCore</b> (protected, faithful port of the author code): WideResNet-50-2 layer 2 + 3 patch features, a 10 % greedy coreset memory of
normal patches, nearest-neighbour distance, max over patches.</li>
<li><b>Local student–teacher</b> (adaptation of EfficientAD's local pathway): a frozen PDN-S teacher with a 33×33 receptive field and a student trained
to imitate it on normal images (hard-feature loss, ImageNette penalty, 70,000 steps).</li>
<li><b>DINOv2 global-attention reconstruction</b> (Dinomaly-style): a frozen DINOv2 ViT-B/14 (with registers) encoder and a linear-attention decoder with a
noisy bottleneck trained to reconstruct the encoder features of normal images; the anomaly map is the cosine reconstruction error.</li>
<li><b>Composition evidence</b> (this project's own design, the Part-Occupancy Composition Memory): DINOv2 patch tokens are grouped into ten visual
clusters; each image is described by cluster mass, appearance, layout and adjacency, and scored by its distance to the nearest normal assemblies.
It targets logical anomalies (missing, extra or misplaced parts) and is disabled for texture categories.</li>
</ul>
<p><b>Normalization and fusion.</b> Each score is robustly standardized on held-out normal images, z = (s − median) / (Q95 − median), and the scores
are fused (pre-registered rule: max; mean and OR are ablations). No test image or label is used to fit, normalize or fuse.</p>
<p><b>Evaluation.</b> Image-level AUROC from the continuous fused score, per category with a stratified bootstrap 95 % confidence interval
(2,000 resamples), averaged over categories (macro) and, as a secondary number, pooled over all images. Each branch's own AUROC is reported, and on
MVTec LOCO the logical and structural anomaly types are evaluated separately against all normal images.</p>
<p><b>Protocol.</b> Immutable split manifests with file hashes; the experiment suite and the source code were frozen (pre-registration, code hash
<code>{prereg['code_hash']}</code>) before the confirmatory test evaluation.</p>

<h2 class="pb">3. MVTec AD (confirmatory)</h2>
<h3>3.1 All completed experiments (15 categories)</h3>
{ad_exp_tbl}
<div class="fig">{chart_ad}
<p class="small">Figure 1. Per-category image AUROC, PatchCore alone vs the full model; black bars are the full model's 95 % bootstrap CIs (y-axis starts at 0.90).</p></div>
<h3>3.2 Full model per category</h3>
{ad_cat_tbl}
<div class="fig">{heat_ad}
<p class="small">Figure 2. Image AUROC of the fused score and of each branch's own score, per category (composition is disabled on textures).</p></div>

<h2 class="pb">4. MVTec LOCO AD (confirmatory)</h2>
<h3>4.1 All completed experiments (5 categories)</h3>
{loco_exp_tbl}
<div class="fig">{chart_loco}
<p class="small">Figure 3. Per-category image AUROC with the full model's 95 % bootstrap CIs. Hatched bars: X3_mean, EXPLORATORY (section 5).</p></div>
<h3>4.2 Full model per category</h3>
{loco_cat_tbl}
<div class="fig">{chart_ls}
<p class="small">Figure 4. Logical vs structural AUROC per category (each anomaly type against all normal images).</p></div>
<div class="fig">{heat_loco}
<p class="small">Figure 5. Image AUROC of the fused score and of each branch's own score, per LOCO category.</p></div>

<h2 class="pb">5. X3: higher-resolution model for MVTec LOCO <span class="tag">EXPLORATORY</span></h2>
<div class="box"><b>Status.</b> X3 was specified and built after the confirmatory LOCO results had been seen, to improve MVTec LOCO detection.
Its numbers are measured on the same test split that motivated it, are optimistically biased, and cannot be confirmed without new labelled data.
The confirmatory LOCO result is the pre-registered full model (section 4).</div>
<p><b>Motivation.</b> Pushpins and screw-bag parts are smaller than one 14-pixel DINOv2 token at the 448-pixel letterbox, so part-level evidence was
not resolved. <b>Design</b> (written before X3 was built or evaluated): <code>composition_hr</code>, the composition memory at an 896-pixel letterbox
(64×64 tokens) with count evidence enabled per visual cluster only where counts are stable on normal images (≥ 90 % at the modal count);
<code>patchcore_hr</code>, a separately named PatchCore at a 512-pixel letterbox with a 1 % coreset. X3 fuses the protected PatchCore,
<code>patchcore_hr</code>, the local student–teacher, the DINOv2 reconstruction and <code>composition_hr</code>, with mean fusion (primary; mean was the
better pre-registered fusion ablation, a preference formed after the test results were seen) or max fusion (secondary).</p>
{x3_tbl}
<h3>X3_mean per category</h3>
{x3_cat_tbl}
<ul>
<li>X3_mean improves macro AUROC over the full model ({f3(g_loco['macro_auroc'])} → {f3(x3m['macro_auroc'])}), mostly on structural anomalies
({f3(g_loco['macro_auroc_structural'])} → {f3(x3m['macro_auroc_structural'])}) and on breakfast box.</li>
<li>Logical anomalies remain the bottleneck ({f3(g_loco['macro_auroc_logical'])} → {f3(x3m['macro_auroc_logical'])}), especially pushpins
({f3(lX.loc['pushpins','auroc_logical'])}) and screw bag ({f3(lX.loc['screw_bag','auroc_logical'])}).</li>
<li>Disclosure: the X3 test evaluation was executed twice on 26 September 2026 (13:55 and 13:59 local) with identical code and scores, and it
produced identical results; nothing was changed between or after the two executions.</li>
</ul>

<h2 class="pb">6. ROC curves (full model)</h2>
<div class="fig">{roc_loco}
<p class="small">Figure 6. MVTec LOCO: ROC curves of the fused score and of each branch, per category; the legend gives each curve's AUROC.</p></div>
<div class="fig">{roc_ad}
<p class="small">Figure 7. MVTec AD: ROC curves of the fused score and of each branch, per category.</p></div>

<h2 class="pb">7. Protocol, disclosures and limitations</h2>
<h3>Pre-registration history</h3>
{table(["version", "date (local)", "reason", "code hash"], [
    ["v1", "2026-09-26 01:23", "initial freeze, before any real test evaluation", "135b7716…"],
    ["v2", "2026-09-26 01:55", "TF32 matmuls for every DINO variant and exact speed-ups (approved by the project owner); superseded v1 before any affected evaluation", "baf601f7…"],
    ["v3", "2026-09-26 14:29", "bug fix only: resuming an interrupted DINO training failed on RNG-state device; no configuration change", "a682c370…"],
], "")}
<h3>What this report covers</h3>
<ul>
<li>This report and package present AUROC-based metrics only. The pre-registered evaluation also computed threshold-dependent metrics; they are
omitted here.</li>
<li>The results were produced by the pre-registered pipeline (code hash <code>{prereg['code_hash']}</code>). The code shipped in the package is a
derivative with the threshold-dependent computation removed (code hash <code>{hash_pkg}</code>), so re-running it labels outputs exploratory. Every
AUROC value in the package was recomputed with the package code from the saved predictions and matches the original exactly.</li>
</ul>
<h3>Use of the test sets</h3>
<ul>
<li>Recorded test-set evaluations for the experiments in this report: MVTec AD {n_ad} (all confirmatory); MVTec LOCO {n_lc} confirmatory and
{n_lx} exploratory (the two X3 executions, two models each). Source: <code>runs/TEST_ACCESS_LOG_excerpt.jsonl</code>.</li>
<li>Earlier MVTec AD evaluations were progress looks at the same pre-registered configurations as categories finished (including a 13-category and a
14-category interim look on 27 September). No configuration, model or fusion rule was selected, tuned or changed because of them.</li>
<li>Before this project, a separate project on the same machine had evaluated on the official MVTec AD and LOCO test splits, so no untouched real-anomaly
set exists for independent confirmation.</li>
</ul>
<h3>Limitations</h3>
<ul>
<li><b>One seed</b> (seed 0); multi-seed runs are pending.</li>
<li><b>Small normalization sets:</b> branch normalizers are fitted on small held-out normal sets (12 shared images for toothbrush).</li>
<li><b>Documented deviations:</b> TF32 matmuls for DINO (vs fp32 in the reference); a third-party PDN-S teacher (anomalib release); letterbox
adaptations for LOCO; composition clusters are not semantically verified.</li>
<li><b>Run incident:</b> leather's DINO training ran out of memory four times on Windows because PyTorch's cache grew into system memory during the
hard-mining warm-up (live memory never exceeded 3.93 GB). It was trained with the unchanged pipeline under a per-process CUDA memory cap, which
changes neither the method nor its math.</li>
<li><b>Not complete in this snapshot:</b> attention / head / penalty ablations and published-protocol reproductions. Partial experiments (categories
completed) — MVTec AD: {esc(partial_txt)}; MVTec LOCO: {esc(partial_loco_txt)}. They are not comparable to the complete rows and are excluded from the
tables above.</li>
</ul>

<h2>8. Reproducing and package contents</h2>
<p>The package folder runs on its own. It bundles the Python 3.12 environment (torch 2.11 + CUDA 12.8), the pretrained weights, the
MVTec AD and MVTec LOCO datasets, the ImageNette penalty images, and the trained models and score files behind every saved result; all
paths are relative to the folder. <code>check.cmd</code> verifies the environment, the weight checksums and every dataset image against
the manifests, recomputes all AUROCs of the full model (both datasets) and X3_mean from the bundled score files (identical to the saved
results) and runs the test suite. <code>hybridad.cmd</code> and <code>x3.cmd</code> run the pipeline and X3; new outputs go to
<code>runs_rerun/</code>, so the saved results in <code>runs/</code> are never overwritten. Details: <code>README.md</code>;
methods and paper-vs-code differences: <code>docs/METHODS.md</code>; AUROC tables: <code>docs/RESULTS.md</code>; X3:
<code>docs/X3.md</code>; data and licence terms: <code>DATA_AND_LICENSES.md</code>.</p>
"""

OUT_HTML.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Kutty Dinosaur</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
if OUT_PDF.exists():
    OUT_PDF.unlink()
subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
                f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, timeout=180,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("pdf", OUT_PDF.stat().st_size, "bytes; package code hash", hash_pkg)
print("AD partial:", ad_partial, "LOCO partial:", loco_partial)
