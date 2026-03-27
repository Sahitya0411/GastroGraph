import os, sys, random, warnings, io, contextlib, time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix, csr_matrix

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR   = os.path.join(PROJECT_ROOT, "config")
OUT_DIR      = os.path.join(PROJECT_ROOT, "visualizations", "static")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BG       = "#0f1117"
BG_PANEL = "#1a1d2e"
TEXT     = "#e2e8f0"
GRID     = "#2d3748"
ACCENT   = "#10b981"
BLUE     = "#60a5fa"
AMBER    = "#fbbf24"
RED      = "#f87171"
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


def load_resources():
    print("Loading graph, embeddings, and recommender ...")
    embeddings = np.load(os.path.join(MODEL_DIR, "node_embeddings.npy"))

    with open(os.path.join(MODEL_DIR, "node2idx.pkl"), "rb") as f:
        data = pickle.load(f)
    node2idx = data["node2idx"]

    with open(os.path.join(MODEL_DIR, "gastro_graph.gpickle"), "rb") as f:
        G = pickle.load(f)

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

    print(f"  Graph : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Embeds: {embeddings.shape}")
    return G, embeddings, node2idx, rec


def build_tfidf_embeddings(G, node2idx):
    print("\n[Variant A] Building TF-IDF feature matrix ...")

    ing_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "ingredient"]
    mol_nodes = sorted({v for u, v, d in G.edges(data=True)
                        if d.get("type") == "CONTAINS"
                        and G.nodes[v]["type"] == "molecule"} |
                       {u for u, v, d in G.edges(data=True)
                        if d.get("type") == "CONTAINS"
                        and G.nodes[u]["type"] == "molecule"})

    mol2col  = {m: c for c, m in enumerate(mol_nodes)}
    N_nodes  = len(node2idx)
    N_mols   = len(mol_nodes)

    mat = lil_matrix((N_nodes, N_mols), dtype=np.float32)

    for u, v, d in G.edges(data=True):
        if d.get("type") != "CONTAINS":
            continue
        ing    = u if G.nodes[u]["type"] == "ingredient" else v
        mol    = v if G.nodes[u]["type"] == "ingredient" else u
        weight = d.get("weight", 1.0)
        if ing in node2idx and mol in mol2col:
            mat[node2idx[ing], mol2col[mol]] = weight

    mat_csr  = mat.tocsr()
    mat_norm = normalize(mat_csr, norm="l2")
    tfidf_embs = mat_norm.toarray()

    n_nonzero = np.count_nonzero(tfidf_embs.sum(axis=1))
    print(f"  TF-IDF matrix: {tfidf_embs.shape}  ({n_nonzero} ingredients with ≥1 molecule)")
    return tfidf_embs


def build_node2vec_embeddings(G, node2idx, dim=64):
    print("\n[Variant B] Training node2vec ...")
    t0 = time.time()

    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  ✗  gensim not installed. Run:  pip install gensim")
        return None

    nodes     = list(G.nodes())
    n2i_local = {n: i for i, n in enumerate(nodes)}

    adj_prob = {}
    for node in nodes:
        nbrs    = list(G.neighbors(node))
        weights = np.array([G[node][nb].get("weight", 1.0) for nb in nbrs], dtype=np.float64)
        if weights.sum() == 0:
            weights = np.ones(len(nbrs), dtype=np.float64)
        weights /= weights.sum()
        adj_prob[node] = (nbrs, weights)

    rng = np.random.default_rng(SEED)

    def random_walk(start, walk_length):
        walk = [start]
        for _ in range(walk_length - 1):
            cur   = walk[-1]
            nbrs, probs = adj_prob[cur]
            if not nbrs:
                break
            walk.append(rng.choice(nbrs, p=probs))
        return [str(n) for n in walk]

    walk_length = 80
    num_walks   = 10

    print(f"  Generating {num_walks * len(nodes):,} walks (length {walk_length}) ...")
    walks = []
    node_list = nodes.copy()
    for _ in range(num_walks):
        rng.shuffle(node_list)
        for n in node_list:
            walks.append(random_walk(n, walk_length))

    print(f"  Walk generation done in {time.time()-t0:.1f}s. Training Word2Vec ...")
    t1 = time.time()

    model = Word2Vec(
        sentences  = walks,
        vector_size= dim,
        window     = 5,
        min_count  = 1,
        sg         = 1,
        workers    = 4,
        epochs     = 5,
        seed       = SEED,
    )

    print(f"  Word2Vec trained in {time.time()-t1:.1f}s.")

    N_nodes   = len(node2idx)
    embs      = np.zeros((N_nodes, dim), dtype=np.float32)
    missing   = 0
    for node, idx in node2idx.items():
        key = str(node)
        if key in model.wv:
            embs[idx] = model.wv[key]
        else:
            missing += 1

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    print(f"  node2vec embeddings: {embs.shape}  ({missing} nodes without walk coverage)")
    return embs


def find_node(G, name):
    for n, d in G.nodes(data=True):
        nm = d.get("name", "")
        if isinstance(nm, str) and nm.lower() == name.lower():
            return n
    return None


def mol_overlap(rec, na, nb):
    return rec._molecule_overlap_score(na, nb)


def metric2_with_embeddings(G, embs, node2idx, rec, label="System"):
    all_ing_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "ingredient"]
    all_ing_idxs  = [node2idx[n] for n in all_ing_nodes]
    all_embs      = embs[all_ing_idxs]

    results = []

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

            q_vec      = embs[node2idx[q_node]].reshape(1, -1)
            sc_idxs    = [node2idx[n] for n in same_cat_nodes]
            sc_embs    = embs[sc_idxs]

            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                continue

            cos_scores  = cosine_similarity(q_vec, sc_embs)[0]
            top5_nodes  = [same_cat_nodes[i] for i in np.argsort(cos_scores)[::-1][:5]]

            avg = np.mean([mol_overlap(rec, q_node, n) for n in top5_nodes])
            results.append(avg * 100)

    return results


def summarise(vals, label):
    arr  = np.array(vals)
    mean = arr.mean()
    std  = arr.std()
    med  = np.median(arr)
    print(f"  {label:<30}: {mean:.1f}% ± {std:.1f}%  (median {med:.1f}%,  n={len(arr)})")
    return mean, std


def mann_whitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return p
    except ImportError:
        return float("nan")


def plot_ablation(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT, which="both")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

    labels = [r[0] for r in results]
    colors = [r[1] for r in results]
    means  = [r[2] for r in results]
    stds   = [r[3] for r in results]
    x      = np.arange(len(labels))

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.80, width=0.5,
                  capsize=6, error_kw={"ecolor": TEXT, "elinewidth": 1.4}, zorder=3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.8,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=10,
                color=TEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=10)
    ax.set_ylabel("Avg TF-IDF Molecular Overlap of Top-5 (%)", color=TEXT, fontsize=10)
    ax.set_title("Metric 2 — Within-Category Ranking Quality\n"
                 "(50 queries, composite scorer, taxonomy gate held constant)",
                 color=TEXT, fontsize=10, pad=10)
    ax.set_ylim(0, max(means) * 1.35)

    ax2 = axes[1]
    ax2.set_facecolor(BG_PANEL)
    ax2.tick_params(colors=TEXT, which="both")
    for sp in ax2.spines.values():
        sp.set_edgecolor(GRID)
    ax2.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

    rng_jitter = np.random.default_rng(SEED)
    for i, (label, color, mean, std, vals) in enumerate(results):
        jitter = rng_jitter.uniform(-0.15, 0.15, size=len(vals))
        ax2.scatter(np.full(len(vals), i) + jitter, vals,
                    color=color, alpha=0.45, s=18, zorder=4)
        ax2.plot([i - 0.25, i + 0.25], [mean, mean],
                 color=color, linewidth=2.5, zorder=5)

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, color=TEXT, fontsize=10)
    ax2.set_ylabel("Avg Mol Overlap per Query (%)", color=TEXT, fontsize=10)
    ax2.set_title("Per-Query Distribution\n(dots = individual queries, line = mean)",
                  color=TEXT, fontsize=10, pad=10)

    fig.suptitle(
        "GastroGraph Embedding Ablation: TF-IDF Cosine vs. node2vec vs. GraphSAGE",
        color=TEXT, fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n  ✓  Figure saved → {path}")


def print_latex_table(results, p_tfidf_sage, p_n2v_sage):
    print("\n" + "=" * 60)
    print("  LaTeX table (paste into paper):")
    print("=" * 60)
    print(r"""\begin{table}[H]
\centering
\caption{Embedding Ablation: Within-Category Ranking Quality
  (Metric~2, 50 queries). All variants use the identical taxonomy gate
  and composite scorer; only the embedding source differs.
  $p$-values from two-sided Mann--Whitney $U$ test vs.\ GraphSAGE.}
\label{tab:ablation_embed}
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c c c}
  \toprule
  \textbf{Embedding} & \textbf{Mean Mol Overlap} & \textbf{Std Dev} & \textbf{$p$ vs.\ GraphSAGE} \\
  \midrule""")

    for i, (label, color, mean, std, vals) in enumerate(results):
        if label == "GraphSAGE":
            p_str = "—"
        elif label == "TF-IDF Cosine\n(no GNN)":
            p_str = f"{p_tfidf_sage:.3f}" if not (p_tfidf_sage != p_tfidf_sage) else "N/A"
        else:
            p_str = f"{p_n2v_sage:.3f}" if not (p_n2v_sage != p_n2v_sage) else "N/A"

        clean_label = label.replace("\n", " ")
        row_cmd = r"\RS " if i % 2 == 0 else "    "
        print(f"  {row_cmd}{clean_label:<30} & ${mean:.1f}\\%$ & $\\pm{std:.1f}\\%$ & {p_str} \\\\")

    print(r"""  \bottomrule
\end{tabular}
\end{table}""")
    print("=" * 60)


if __name__ == "__main__":
    G, sage_embs, node2idx, rec = load_resources()

    tfidf_embs = build_tfidf_embeddings(G, node2idx)

    n2v_embs = build_node2vec_embeddings(G, node2idx, dim=64)

    print("\n" + "=" * 60)
    print("  METRIC 2 — WITHIN-CATEGORY RANKING QUALITY")
    print("  (avg TF-IDF molecular overlap of top-5 substitutes)")
    print("=" * 60)

    tfidf_vals = metric2_with_embeddings(G, tfidf_embs, node2idx, rec, "TF-IDF Cosine")
    n2v_vals   = (metric2_with_embeddings(G, n2v_embs, node2idx, rec, "node2vec")
                  if n2v_embs is not None else [])
    sage_vals  = metric2_with_embeddings(G, sage_embs, node2idx, rec, "GraphSAGE")

    print()
    tfidf_mean, tfidf_std = summarise(tfidf_vals, "TF-IDF Cosine (no GNN)")
    n2v_mean,   n2v_std   = summarise(n2v_vals,   "node2vec") if n2v_vals else (0.0, 0.0)
    sage_mean,  sage_std  = summarise(sage_vals,   "GraphSAGE (full)")

    p_tfidf = mann_whitney_p(tfidf_vals, sage_vals)
    p_n2v   = mann_whitney_p(n2v_vals,   sage_vals) if n2v_vals else float("nan")

    print(f"\n  Mann-Whitney U (TF-IDF vs GraphSAGE) : p = {p_tfidf:.4f}")
    if n2v_vals:
        print(f"  Mann-Whitney U (node2vec vs GraphSAGE): p = {p_n2v:.4f}")

    results = [
        ("TF-IDF Cosine\n(no GNN)", AMBER,  tfidf_mean, tfidf_std, tfidf_vals),
    ]
    if n2v_vals:
        results.append(("node2vec",          BLUE,   n2v_mean,  n2v_std,  n2v_vals))
    results.append(("GraphSAGE",             ACCENT, sage_mean, sage_std, sage_vals))

    fig_path = os.path.join(OUT_DIR, "ablation_embeddings.png")
    plot_ablation(results, fig_path)

    print_latex_table(results, p_tfidf, p_n2v)

    print("\nDone.")
