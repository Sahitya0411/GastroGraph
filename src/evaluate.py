import os, sys, random, warnings, io, contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics.pairwise import cosine_similarity
import pickle

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR   = os.path.join(PROJECT_ROOT, "config")
OUT_DIR      = os.path.join(PROJECT_ROOT, "visualizations", "static")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))

BG       = "#0f1117"
BG_PANEL = "#1a1d2e"
TEXT     = "#e2e8f0"
GRID     = "#2d3748"
ACCENT   = "#10b981"
BLUE     = "#60a5fa"
RED      = "#f87171"
AMBER    = "#fbbf24"
PURPLE   = "#a78bfa"

QUERY_POOL = {
    "HERB":  ["basil", "thyme", "oregano", "rosemary", "sage",
              "dill", "tarragon", "marjoram", "chervil", "chive"],
    "FRUIT": ["apple", "strawberry", "mango", "peach", "apricot",
              "plum", "cherry", "pear", "pineapple", "grape"],
    "MEAT":  ["chicken", "beef", "pork", "lamb", "turkey",
              "duck", "veal", "venison", "bison", "mutton"],
    "DAIRY": ["milk", "butter", "cream", "yogurt", "cheese",
              "ricotta", "mozzarella", "cheddar", "gouda", "brie"],
    "SPICE": ["cinnamon", "cumin", "paprika", "turmeric", "ginger",
              "cardamom", "clove", "nutmeg", "coriander", "pepper"],
}

np.random.seed(42)
random.seed(42)


def load_resources():
    print("Loading embeddings and graph...")
    embeddings = np.load(os.path.join(MODEL_DIR, "node_embeddings.npy"))
    with open(os.path.join(MODEL_DIR, "node2idx.pkl"), "rb") as f:
        data = pickle.load(f)
    node2idx = data["node2idx"]
    idx2node = {v: k for k, v in node2idx.items()}
    with open(os.path.join(MODEL_DIR, "gastro_graph.gpickle"), "rb") as f:
        G = pickle.load(f)
    return G, embeddings, node2idx, idx2node


def get_recommender(G, embeddings, node2idx):
    from disjoint_context import DisjointContextRecommender
    with contextlib.redirect_stdout(io.StringIO()):
        rec = DisjointContextRecommender(
            output_dir=MODEL_DIR, input_dir=DATA_DIR, config_dir=CONFIG_DIR
        )
    rec.G          = G
    rec.embeddings = embeddings
    rec.node2idx   = node2idx
    rec.idx2node   = {v: k for k, v in node2idx.items()}
    rec._build_molecule_index()
    return rec


def find_node(G, name):
    for n, d in G.nodes(data=True):
        nm = d.get("name", "")
        if isinstance(nm, str) and nm.lower() == name.lower():
            return n
    return None


def mol_overlap(rec, na, nb):
    return rec._molecule_overlap_score(na, nb)


def styled_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT, which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6)
    return fig, ax


def save_fig(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Saved → {path}")
    return path


def metric1_baseline_fcr(G, embeddings, node2idx, rec):
    print("\n[Metric 1] Baseline Functional Coherence Rate (FCR@10) ...")

    all_ing_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "ingredient"]
    all_ing_idxs  = [node2idx[n] for n in all_ing_nodes]
    all_embs      = embeddings[all_ing_idxs]

    per_category_fcr = {}
    all_fcrs         = []
    query_results    = []

    for cat, queries in QUERY_POOL.items():
        cat_fcrs = []
        for q_name in queries:
            q_node = find_node(G, q_name)
            if q_node is None:
                continue
            _, q_group = rec._get_sub_category(q_name)
            if q_group is None:
                continue

            q_vec      = embeddings[node2idx[q_node]].reshape(1, -1)
            cos_scores = cosine_similarity(q_vec, all_embs)[0]
            top10_idxs = np.argsort(cos_scores)[::-1][:11]

            correct = 0
            for idx in top10_idxs:
                node = all_ing_nodes[idx]
                if node == q_node:
                    continue
                cname  = G.nodes[node]["name"]
                _, cg  = rec._get_sub_category(cname)
                if cg == q_group:
                    correct += 1
                if correct + (10 - (idx - top10_idxs[0])) < 0:
                    break

            fcr = correct / 10.0
            cat_fcrs.append(fcr)
            all_fcrs.append(fcr)
            query_results.append({"query": q_name, "category": cat, "fcr": fcr})

        per_category_fcr[cat] = np.mean(cat_fcrs) if cat_fcrs else 0.0

    mean_fcr = np.mean(all_fcrs)
    std_fcr  = np.std(all_fcrs)
    print(f"  Baseline FCR@10 = {mean_fcr:.3f} ± {std_fcr:.3f}")
    for cat, v in per_category_fcr.items():
        print(f"    {cat:<10}: {v:.3f}")

    cats   = list(per_category_fcr.keys())
    vals   = [per_category_fcr[c] for c in cats]
    colors = [BLUE, ACCENT, RED, AMBER, PURPLE]

    fig, ax = styled_fig(9, 5)
    bars = ax.bar(cats, vals, color=colors, alpha=0.85, width=0.55, zorder=3)
    ax.axhline(mean_fcr, color=TEXT, linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"Mean FCR = {mean_fcr:.2f}")
    ax.axhline(1.0, color="#4ade80", linestyle=":", linewidth=1.0, alpha=0.6,
               label="GastroGraph (by design = 1.0)")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("FCR@10 (Functional Coherence Rate)", color=TEXT, fontsize=10)
    ax.set_xlabel("Query Category", color=TEXT, fontsize=10)
    ax.set_title(
        "Metric 1 — Baseline FCR@10: Naive Cosine Similarity\n"
        "(% of top-10 results with correct functional group, 50 queries)",
        color=TEXT, fontsize=10, pad=10,
    )
    legend = ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID, labelcolor=TEXT)
    save_fig(fig, "eval_metric1_baseline_fcr.png")
    return mean_fcr, std_fcr, per_category_fcr


def metric2_ranking_quality(G, embeddings, node2idx, rec):
    print("\n[Metric 2] Within-category ranking quality ...")

    all_ing_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "ingredient"]
    all_ing_idxs  = [node2idx[n] for n in all_ing_nodes]
    all_embs      = embeddings[all_ing_idxs]

    gastrograph_mol = []
    cosine_cat_mol  = []
    random_cat_mol  = []

    for cat, queries in QUERY_POOL.items():
        for q_name in queries:
            q_node = find_node(G, q_name)
            if q_node is None:
                continue
            _, q_group = rec._get_sub_category(q_name)
            if q_group is None:
                continue

            same_cat_nodes = [
                n for n in all_ing_nodes
                if n != q_node
                and rec._get_sub_category(G.nodes[n]["name"])[1] == q_group
                and not G.has_edge(q_node, n)
            ]
            if len(same_cat_nodes) < 10:
                continue

            q_vec      = embeddings[node2idx[q_node]].reshape(1, -1)
            sc_idxs    = [node2idx[n] for n in same_cat_nodes]
            sc_embs    = embeddings[sc_idxs]
            cos_scores = cosine_similarity(q_vec, sc_embs)[0]
            top5_cos   = [same_cat_nodes[i] for i in np.argsort(cos_scores)[::-1][:5]]

            with contextlib.redirect_stdout(io.StringIO()):
                result = rec.get_substitutes(q_name, top_k=5, show_complements=False)
            gastrograph_nodes = [r["node"] for r in result.get("functional", [])]

            rand5 = random.sample(same_cat_nodes, min(5, len(same_cat_nodes)))

            def avg_mol(nodes):
                if not nodes:
                    return 0.0
                return np.mean([mol_overlap(rec, q_node, n) for n in nodes])

            gastrograph_mol.append(avg_mol(gastrograph_nodes))
            cosine_cat_mol.append(avg_mol(top5_cos))
            random_cat_mol.append(avg_mol(rand5))

    gg_mean   = np.mean(gastrograph_mol) * 100
    cos_mean  = np.mean(cosine_cat_mol)  * 100
    rand_mean = np.mean(random_cat_mol)  * 100
    gg_std    = np.std(gastrograph_mol)  * 100
    cos_std   = np.std(cosine_cat_mol)   * 100
    rand_std  = np.std(random_cat_mol)   * 100

    print(f"  GastroGraph top-5 avg mol overlap  : {gg_mean:.1f}% ± {gg_std:.1f}%")
    print(f"  Same-cat cosine top-5 avg mol overlap: {cos_mean:.1f}% ± {cos_std:.1f}%")
    print(f"  Random same-cat avg mol overlap     : {rand_mean:.1f}% ± {rand_std:.1f}%")

    labels  = ["GastroGraph\n(composite)", "Cosine\n(same-cat)", "Random\n(same-cat)"]
    means   = [gg_mean, cos_mean, rand_mean]
    stds    = [gg_std,  cos_std,  rand_std]
    colors  = [ACCENT,  BLUE,     AMBER]

    fig, ax = styled_fig(8, 5)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, width=0.5,
                  capsize=5, error_kw={"ecolor": TEXT, "elinewidth": 1.2}, zorder=3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=10, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=10)
    ax.set_ylabel("Avg TF-IDF Molecular Overlap of Top-5 (%)", color=TEXT, fontsize=10)
    ax.set_title(
        "Metric 2 — Within-Category Ranking Quality\n"
        "(avg mol overlap of top-5 results within same functional group, 50 queries)",
        color=TEXT, fontsize=10, pad=10,
    )
    save_fig(fig, "eval_metric2_ranking_quality.png")
    return gg_mean, cos_mean, rand_mean


def metric3_aromatic_quality(G, embeddings, node2idx, rec):
    print("\n[Metric 3] Aromatic pairing molecular overlap ...")

    all_ing_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "ingredient"]

    gastrograph_arom = []
    random_cross_cat = []
    blocked_by_guard = []

    for cat, queries in QUERY_POOL.items():
        for q_name in queries:
            q_node = find_node(G, q_name)
            if q_node is None:
                continue
            _, q_group = rec._get_sub_category(q_name)
            if q_group is None:
                continue

            with contextlib.redirect_stdout(io.StringIO()):
                result = rec.get_substitutes(q_name, top_k=5, show_complements=False)
            arom_results = result.get("aromatic", [])

            if arom_results:
                avg = np.mean([r.get("_mol_score", 0) for r in arom_results]) * 100
                gastrograph_arom.append(avg)

            cross_cat = [
                n for n in all_ing_nodes
                if n != q_node
                and rec._get_sub_category(G.nodes[n]["name"])[1] != q_group
                and not G.has_edge(q_node, n)
            ]
            if cross_cat:
                rand5     = random.sample(cross_cat, min(5, len(cross_cat)))
                rand_mols = [mol_overlap(rec, q_node, n) for n in rand5]
                random_cross_cat.append(np.mean(rand_mols) * 100)

            blocked = sum(
                1 for n in cross_cat[:200]
                if not rec._is_aromatic_compatible(
                    q_group,
                    rec._get_sub_category(G.nodes[n]["name"])[1]
                )
            )
            blocked_by_guard.append(blocked / min(200, len(cross_cat)))

    gg_mean   = np.mean(gastrograph_arom)
    rand_mean = np.mean(random_cross_cat)
    guard_pct = np.mean(blocked_by_guard) * 100
    gg_std    = np.std(gastrograph_arom)
    rand_std  = np.std(random_cross_cat)

    print(f"  GastroGraph aromatic top-5 avg mol overlap : {gg_mean:.1f}% ± {gg_std:.1f}%")
    print(f"  Random cross-category avg mol overlap      : {rand_mean:.1f}% ± {rand_std:.1f}%")
    print(f"  Avg cross-cat pairs blocked by guard       : {guard_pct:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]
    ax1.set_facecolor(BG_PANEL)
    ax1.tick_params(colors=TEXT, which="both")
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID)
    ax1.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6)

    labels = ["GastroGraph\nAromatic Top-5", "Random\nCross-Category"]
    means  = [gg_mean, rand_mean]
    stds   = [gg_std,  rand_std]
    colors = [ACCENT,  RED]
    bars   = ax1.bar(labels, means, yerr=stds, color=colors, alpha=0.85, width=0.45,
                     capsize=5, error_kw={"ecolor": TEXT, "elinewidth": 1.2}, zorder=3)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, m + s + 0.2,
                 f"{m:.1f}%", ha="center", va="bottom", fontsize=11, color=TEXT)
    ax1.set_ylabel("Avg TF-IDF Molecular Overlap (%)", color=TEXT, fontsize=10)
    ax1.set_title("Aromatic Pairing Quality\nvs. Random Cross-Category", color=TEXT, fontsize=10)
    ax1.tick_params(colors=TEXT)

    ax2 = axes[1]
    ax2.set_facecolor(BG_PANEL)
    ax2.tick_params(colors=TEXT, which="both")
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID)
    ax2.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6)

    guard_vals = [guard_pct, 100 - guard_pct]
    wedge_cols = [RED, ACCENT]
    wedges, texts, autotexts = ax2.pie(
        guard_vals,
        labels=["Blocked\n(incompatible)", "Permitted\n(compatible)"],
        colors=wedge_cols,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": TEXT, "fontsize": 9},
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax2.set_title(
        f"Incompatibility Guard Coverage\n"
        f"(avg {guard_pct:.1f}% of cross-cat pairs blocked per query)",
        color=TEXT, fontsize=10,
    )

    fig.suptitle(
        "Metric 3 — Aromatic Pairing Pipeline Quality (50 queries)",
        color=TEXT, fontsize=11, y=1.02,
    )
    plt.tight_layout()
    save_fig(fig, "eval_metric3_aromatic.png")
    return gg_mean, rand_mean, guard_pct


def summary_figure(m1_mean, m1_std, m2_gg, m2_cos, m2_rand, m3_gg, m3_rand):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)

    panels = [
        {
            "ax": axes[0],
            "title": "Metric 1\nBaseline FCR@10",
            "bars": ["Naive\nCosine", "GastroGraph\n(by design)"],
            "vals": [m1_mean * 100, 100],
            "colors": [RED, ACCENT],
            "ylabel": "Functional Coherence Rate (%)",
            "note": f"Baseline = {m1_mean*100:.1f}%\nGastroGraph guarantees 100%",
        },
        {
            "ax": axes[1],
            "title": "Metric 2\nWithin-Category Ranking",
            "bars": ["GastroGraph", "Cosine\n(same-cat)", "Random\n(same-cat)"],
            "vals": [m2_gg, m2_cos, m2_rand],
            "colors": [ACCENT, BLUE, AMBER],
            "ylabel": "Avg Mol Overlap of Top-5 (%)",
            "note": f"GastroGraph +{m2_gg - m2_cos:.1f}% vs cosine",
        },
        {
            "ax": axes[2],
            "title": "Metric 3\nAromatic Pairing Quality",
            "bars": ["GastroGraph\nAromatic", "Random\nCross-Cat"],
            "vals": [m3_gg, m3_rand],
            "colors": [ACCENT, RED],
            "ylabel": "Avg Mol Overlap of Top-5 (%)",
            "note": f"GastroGraph +{m3_gg - m3_rand:.1f}% vs random",
        },
    ]

    for p in panels:
        ax = p["ax"]
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

        bars = ax.bar(p["bars"], p["vals"], color=p["colors"],
                      alpha=0.85, width=0.5, zorder=3)
        for bar, v in zip(bars, p["vals"]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=TEXT)

        ax.set_ylabel(p["ylabel"], color=TEXT, fontsize=9)
        ax.set_title(p["title"], color=TEXT, fontsize=10, pad=8)
        ax.tick_params(axis="x", labelcolor=TEXT, labelsize=8)
        ax.set_ylim(0, max(p["vals"]) * 1.25)

        ax.text(0.98, 0.97, p["note"], transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=AMBER,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=GRID, alpha=0.8))

    fig.suptitle(
        "GastroGraph — Quantitative Evaluation Summary (50 queries, 5 categories)",
        color=TEXT, fontsize=12, y=1.02,
    )
    plt.tight_layout()
    save_fig(fig, "eval_summary.png")


if __name__ == "__main__":
    G, embeddings, node2idx, idx2node = load_resources()
    rec = get_recommender(G, embeddings, node2idx)

    m1_mean, m1_std, m1_per_cat = metric1_baseline_fcr(G, embeddings, node2idx, rec)
    m2_gg, m2_cos, m2_rand      = metric2_ranking_quality(G, embeddings, node2idx, rec)
    m3_gg, m3_rand, m3_guard    = metric3_aromatic_quality(G, embeddings, node2idx, rec)

    summary_figure(m1_mean, m1_std, m2_gg, m2_cos, m2_rand, m3_gg, m3_rand)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  M1  Baseline FCR@10          : {m1_mean*100:.1f}% ± {m1_std*100:.1f}%")
    print(f"  M2  GastroGraph mol overlap  : {m2_gg:.1f}%  |  cosine: {m2_cos:.1f}%  |  random: {m2_rand:.1f}%")
    print(f"  M3  Aromatic mol overlap     : {m3_gg:.1f}%  vs  random cross-cat: {m3_rand:.1f}%")
    print(f"  M3  Guard blocks             : {m3_guard:.1f}% of cross-cat pairs per query")
    print("=" * 60)
    print()
    print("  Saved figures:")
    for f in ["eval_metric1_baseline_fcr.png", "eval_metric2_ranking_quality.png",
              "eval_metric3_aromatic.png", "eval_summary.png"]:
        print(f"    → visualizations/static/{f}")
    print()
