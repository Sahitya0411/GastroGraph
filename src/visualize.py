"""
visualize.py — GastroGraph Comprehensive Visualization Suite
=============================================================
Generates ALL visualizations in one run:

  1. t-SNE Ingredient Universe Map   (static PNG  → visualizations/static/viz_tsne_universe.png)
  2. Category Distribution Chart     (static PNG  → visualizations/static/viz_category_dist.png)
  3. Molecule Overlap Heatmap        (static PNG  → visualizations/static/viz_molecule_heatmap.png)
  4. Neighbourhood Subgraph          (static PNG  → visualizations/static/viz_neighbourhood_{name}.png)
  5. Interactive t-SNE Explorer      (HTML        → visualizations/interactive/explorer.html)
  6. Interactive Ingredient Network  (HTML        → visualizations/interactive/network.html)

Usage:
    cd /path/to/GastroGraph
    python3 src/visualize.py                      # all visualizations
    python3 src/visualize.py --ingredient apple   # also draws apple's neighbourhood
    python3 src/visualize.py --skip-tsne           # skip slow t-SNE (use PCA instead)
"""

import sys, os, argparse, pickle, json, warnings
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')                                                    
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

                                                                               
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SRC_DIR)
OUTPUT_DIR  = os.path.join(ROOT_DIR, 'models')
INPUT_DIR   = os.path.join(ROOT_DIR, 'data')
CONFIG_DIR  = os.path.join(ROOT_DIR, 'config')
VIZ_DIR     = os.path.join(ROOT_DIR, 'visualizations')
VIZ_STATIC  = os.path.join(VIZ_DIR, 'static')
VIZ_INTER   = os.path.join(VIZ_DIR, 'interactive')

sys.path.insert(0, SRC_DIR)
os.chdir(ROOT_DIR)

                                                                               
PALETTE = {
    'Herb':            '#00c896',               
    'Spice':           '#ff8c42',                
    'Allium':          '#c084fc',                
    'Solid Fat':       '#facc15',           
    'Liquid Oil':      '#f97316',          
    'Dairy Liquid':    '#a78bfa',             
    'Dairy Soft':      '#8b5cf6',           
    'Cheese':          '#7c3aed',                
    'Sweetener':       '#fb7185',         
    'Acid/Sour':       '#38bdf8',             
    'Root Vegetable':  '#86efac',               
    'Leafy Green':     '#22c55e',          
    'Fruiting Veg':    '#4ade80',                
    'Brassica':        '#bbf7d0',               
    'Legume':          '#fde68a',                
    'Whole Grain':     '#fbbf24',          
    'Starch/Flour':    '#fcd34d',           
    'Bread':           '#d97706',            
    'Tree Nut':        '#a16207',          
    'Seed':            '#78350f',               
    'Mushroom':        '#713f12',           
    'Meat':            '#ef4444',        
    'Seafood':         '#3b82f6',         
    'Egg':             '#fef08a',          
    'Plant Protein':   '#6ee7b7',         
    'Fruit':           '#fbbf24',         
    'Citrus':          '#fde047',                  
    'Chocolate':       '#92400e',              
    'Coffee/Tea':      '#78350f',               
    'Alcohol':         '#dc2626',             
    'Condiment':           '#9ca3af',              
    'Flower':              '#f472b6',         
    'Beverage':            '#67e8f9',         
    'Beverage Alcoholic':  '#dc2626',                     
    'Plant/Vegetable':     '#4ade80',               
    'Plant Protein':       '#6ee7b7',               
    'Bakery/Dessert/Snack':'#f9a8d4',               
    'Sauce/Powder/Dressing':'#d1d5db',          
    'Cereal/Crop/Bean':    '#fde68a',                
    'Dish/End Product':    '#a3a3a3',                
    'Meat/Animal Product': '#b91c1c',             
    'ETC':                 '#6b7280',          
    'Nut/Seed':            '#92400e',          
    'Unknown':             '#374151',              
}

BG        = '#0f1117'
BG_PANEL  = '#1a1d2e'
TEXT      = '#e2e8f0'
GRID      = '#2d3748'
ACCENT    = '#6366f1'          

                                                                               
def load_resources():
    print("Loading graph, embeddings and mappings...")
    embeddings = np.load(os.path.join(OUTPUT_DIR, 'node_embeddings.npy'))
    with open(os.path.join(OUTPUT_DIR, 'node2idx.pkl'), 'rb') as f:
        m = pickle.load(f)
        node2idx = m['node2idx']
    with open(os.path.join(OUTPUT_DIR, 'gastro_graph.gpickle'), 'rb') as f:
        G = pickle.load(f)
    return G, embeddings, node2idx


def get_recommender():
    from disjoint_context import DisjointContextRecommender
    return DisjointContextRecommender(output_dir=OUTPUT_DIR, input_dir=INPUT_DIR, config_dir=CONFIG_DIR)


def reduce_2d(X, use_tsne=True, n_pca=50, perplexity=40, seed=42):
    """PCA → (optional t-SNE) to 2-D."""
    n_pca = min(n_pca, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pca, random_state=seed)
    X_pca = pca.fit_transform(X)
    if use_tsne:
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=seed, init='pca', learning_rate='auto',
                    n_jobs=-1 if hasattr(TSNE, '_fit') else 1)
        return tsne.fit_transform(X_pca)
    return PCA(n_components=2, random_state=seed).fit_transform(X_pca)


def styled_fig(w=16, h=10, title=''):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG_PANEL)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=18, fontweight='bold', y=0.97)
    return fig, ax


def save(fig, fname):
    os.makedirs(VIZ_STATIC, exist_ok=True)
    path = os.path.join(VIZ_STATIC, fname)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Saved → {path}")
    return path


def plot_tsne_universe(G, embeddings, node2idx, rec, use_tsne=True):
    print("\n[1/6] t-SNE Ingredient Universe Map...")

    rows = []
    for n, d in G.nodes(data=True):
        if d.get('type') != 'ingredient':
            continue
        name = d.get('name', '')
        if not isinstance(name, str):
            continue
        idx = node2idx.get(n)
        if idx is None:
            continue
        label, group = rec._get_sub_category(name)
        rows.append({'emb': embeddings[idx], 'label': label or 'Unknown', 'name': name})

    X      = np.array([r['emb'] for r in rows])
    labels = [r['label'] for r in rows]
    names  = [r['name']  for r in rows]

    print(f"     Reducing {X.shape[0]} ingredient vectors to 2-D ({'t-SNE' if use_tsne else 'PCA'})...")
    xy = reduce_2d(X, use_tsne=use_tsne)


    cat_counts = Counter(labels)
    real_cats  = sorted([c for c in cat_counts if c != 'Unknown'],
                        key=lambda c: cat_counts[c])


    fig = plt.figure(figsize=(24, 15), facecolor=BG)
    ax  = fig.add_subplot(111, facecolor=BG_PANEL)
    ax.set_aspect('equal', adjustable='datalim')


    unk_idx = [i for i, l in enumerate(labels) if l == 'Unknown']
    if unk_idx:
        ax.scatter(
            xy[unk_idx, 0], xy[unk_idx, 1],
            c='#374151', s=8, alpha=0.25,
            linewidths=0, label='Unknown', zorder=1
        )


    for cat in real_cats:
        idx_list = [i for i, l in enumerate(labels) if l == cat]
        ax.scatter(
            xy[idx_list, 0], xy[idx_list, 1],
            c=PALETTE.get(cat, '#94a3b8'),
            label=cat,
            s=28, alpha=0.85,
            linewidths=0.4, edgecolors='white',
            zorder=2
        )


    MIN_LABEL_SIZE = 5
    for cat in real_cats:
        idx_list = [i for i, l in enumerate(labels) if l == cat]
        if len(idx_list) < MIN_LABEL_SIZE:
            continue
        cx = np.mean(xy[idx_list, 0])
        cy = np.mean(xy[idx_list, 1])
        ax.text(
            cx, cy,
            cat.replace('/', '\n'),
            ha='center', va='center',
            color='white', fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      fc=PALETTE.get(cat, '#374151'),
                      ec='none', alpha=0.75),
            zorder=5
        )


    LANDMARKS = {
        'butter', 'basil', 'beef', 'salmon', 'cinnamon', 'lemon', 'honey',
        'milk', 'garlic', 'chocolate', 'coffee', 'truffle', 'saffron',
        'vanilla', 'miso', 'tahini', 'anchovy', 'lavender',
    }
    for i, name in enumerate(names):
        if name in LANDMARKS:
            ax.annotate(
                name.replace('_', ' '),
                (xy[i, 0], xy[i, 1]),
                color='white', fontsize=7, fontweight='bold',
                xytext=(6, 4), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.18', fc='#0f172a', ec='none', alpha=0.8),
                zorder=6
            )

    ax.set_title(
        'GastroGraph · Ingredient Universe\n'
        '(culinary categories revealed by t-SNE on graph embeddings)',
        color=TEXT, fontsize=17, pad=16, fontweight='bold'
    )
    ax.axis('off')


    handles, lbls = ax.get_legend_handles_labels()

    paired = sorted(zip(lbls, handles), key=lambda x: (x[0] == 'Unknown', x[0]))
    lbls_sorted, handles_sorted = zip(*paired) if paired else ([], [])

    legend = ax.legend(
        handles_sorted, lbls_sorted,
        title='Category', loc='center left', bbox_to_anchor=(1.01, 0.5),
        ncol=2, fontsize=8, title_fontsize=10,
        facecolor=BG_PANEL, edgecolor=GRID, labelcolor=TEXT,
        markerscale=1.8, framealpha=0.9
    )
    plt.setp(legend.get_title(), color=TEXT)
    plt.tight_layout()
    return save(fig, 'viz_tsne_universe.png')


def plot_category_distribution(G, rec):
    print("\n[2/6] Category Distribution Chart...")

    counter = Counter()
    for n, d in G.nodes(data=True):
        if d.get('type') != 'ingredient':
            continue
        name = d.get('name', '')
        if not isinstance(name, str):
            continue
        label, _ = rec._get_sub_category(name)
        counter[label or 'Unknown'] += 1

    cats   = [k for k, _ in counter.most_common()]
    counts = [counter[k] for k in cats]
    colors = [PALETTE.get(c, '#374151') for c in cats]

    fig, ax = styled_fig(18, 8, 'GastroGraph · Ingredient Count by Category')
    bars = ax.barh(cats, counts, color=colors, edgecolor=BG, linewidth=0.5, height=0.7)


    for bar, val in zip(bars, counts):
        ax.text(val + 6, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', color=TEXT, fontsize=8)

    ax.set_xlabel('Number of Ingredients', color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    plt.tight_layout()
    return save(fig, 'viz_category_dist.png')


def plot_molecule_heatmap(G, rec):
    print("\n[3/6] Molecule-Overlap Heatmap...")

    INGREDIENTS = [
        'butter', 'milk', 'cream_cheese', 'basil', 'thyme', 'oregano',
        'lemon', 'lime', 'orange', 'cinnamon', 'clove', 'cardamom',
        'beef', 'salmon', 'shrimp', 'garlic', 'onion', 'ginger',
        'chocolate', 'coffee', 'vanilla', 'honey', 'sugar', 'olive_oil',
    ]


    name2node = {}
    for n, d in G.nodes(data=True):
        if d.get('type') == 'ingredient':
            nm = d.get('name', '')
            if isinstance(nm, str):
                name2node[nm] = n

    resolved = [(ing, name2node[ing]) for ing in INGREDIENTS if ing in name2node]
    labels   = [ing.replace('_', ' ') for ing, _ in resolved]
    n_ing    = len(resolved)


    mat = np.zeros((n_ing, n_ing))
    for i, (_, na) in enumerate(resolved):
        for j, (_, nb) in enumerate(resolved):
            if i == j:
                mat[i, j] = 1.0
            elif j > i:
                score = rec._molecule_overlap_score(na, nb)
                mat[i, j] = score
                mat[j, i] = score

    fig, ax = styled_fig(16, 14, 'GastroGraph · Molecule-Overlap Heatmap\n(shared flavour compounds, Jaccard-weighted by TF-IDF)')
    cmap = matplotlib.colormaps['RdYlGn'].resampled(256)
    im   = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.6, aspect='equal')

    ax.set_xticks(range(n_ing))
    ax.set_yticks(range(n_ing))
    ax.set_xticklabels(labels, rotation=45, ha='right', color=TEXT, fontsize=9)
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT)


    for i in range(n_ing):
        for j in range(n_ing):
            val = mat[i, j]
            color = 'black' if val > 0.35 else 'white'
            if val > 0.01:
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        color=color, fontsize=6.5, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    cbar.ax.set_ylabel('Molecule Overlap (Jaccard)', color=TEXT, fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    plt.tight_layout()
    return save(fig, 'viz_molecule_heatmap.png')


def plot_neighbourhood(G, ingredient_name='apple', top_ing=20, top_mol=20):
    print(f"\n[4/6] Neighbourhood Subgraph for '{ingredient_name}'...")


    target = None
    for n, d in G.nodes(data=True):
        nm = d.get('name', '')
        if isinstance(nm, str) and nm.lower() == ingredient_name.lower():
            target = n
            break
    if target is None:
        print(f"     ⚠   '{ingredient_name}' not found — skipping neighbourhood plot.")
        return None

    t_name = G.nodes[target]['name']

    def edge_w(n):
        d = G.get_edge_data(target, n) or {}
        return d.get('weight', 0.0)

    nbrs      = list(G.neighbors(target))
    ing_nbrs  = sorted([n for n in nbrs if G.nodes[n].get('type') == 'ingredient'],
                        key=edge_w, reverse=True)[:top_ing]
    mol_nbrs  = sorted([n for n in nbrs if G.nodes[n].get('type') == 'molecule'],
                        key=edge_w, reverse=True)[:top_mol]

    sub_nodes = [target] + ing_nbrs + mol_nbrs
    sg        = G.subgraph(sub_nodes)


    pos = {target: np.array([0.0, 0.0])}
    for i, n in enumerate(ing_nbrs):
        angle = np.pi / 2 - i * np.pi / max(len(ing_nbrs), 1)
        r     = 1.4 + (i % 3) * 0.35
        pos[n] = np.array([r * np.cos(angle * 0.6), r * np.sin(angle)])
    for i, n in enumerate(mol_nbrs):
        angle = np.pi / 2 - i * np.pi / max(len(mol_nbrs), 1)
        r     = 1.4 + (i % 3) * 0.35
        pos[n] = np.array([-r * np.cos(angle * 0.6), r * np.sin(angle)])

    fig, ax = plt.subplots(figsize=(22, 14), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')


    for u, v, d in sg.edges(data=True):
        if u not in pos or v not in pos:
            continue
        etype = d.get('type', '')
        color = '#34d39944' if etype == 'CO_OCCURS' else '#f97316' + '44'
        lw    = max(0.3, min(2.0, d.get('weight', 0.5) * 3))
        p0, p1 = pos[u], pos[v]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, zorder=1)


    for n in sg.nodes():
        if n not in pos:
            continue
        p = pos[n]
        ntype = G.nodes[n].get('type', '')
        nm    = str(G.nodes[n].get('name', n)).replace('_', ' ')
        if n == target:
            circ = plt.Circle(p, 0.12, color='#6366f1', zorder=5)
            ax.add_patch(circ)
            ax.text(p[0], p[1] - 0.18, nm, ha='center', va='top',
                    color='white', fontsize=11, fontweight='bold', zorder=6)
        elif ntype == 'ingredient':
            circ = plt.Circle(p, 0.07, color='#22c55e', zorder=4, alpha=0.9)
            ax.add_patch(circ)
            w = edge_w(n)
            ax.text(p[0], p[1] + 0.10, nm, ha='center', va='bottom',
                    color=TEXT, fontsize=7, zorder=5, alpha=0.9)
            ax.text(p[0], p[1] - 0.10, f'{w:.2f}', ha='center', va='top',
                    color='#94a3b8', fontsize=6, zorder=5)
        else:
            circ = plt.Circle(p, 0.055, color='#f97316', zorder=4, alpha=0.85)
            ax.add_patch(circ)
            ax.text(p[0], p[1] + 0.085, nm, ha='center', va='bottom',
                    color=TEXT, fontsize=6.5, zorder=5, alpha=0.85)


    patches = [
        mpatches.Patch(color='#6366f1', label='Query ingredient'),
        mpatches.Patch(color='#22c55e', label='Co-occurring ingredients'),
        mpatches.Patch(color='#f97316', label='Flavour molecules'),
    ]
    ax.legend(handles=patches, loc='lower right', facecolor=BG_PANEL,
              edgecolor=GRID, labelcolor=TEXT, fontsize=10, framealpha=0.85)

    ax.set_title(
        f'GastroGraph · Neighbourhood of  "{t_name}"\n'
        f'(green = co-occurring ingredients · orange = contained molecules)',
        color=TEXT, fontsize=15, pad=10
    )
    ax.autoscale_view()
    plt.tight_layout()
    fname = f'viz_neighbourhood_{ingredient_name}.png'
    return save(fig, fname)


def plot_interactive_explorer(G, embeddings, node2idx, rec, use_tsne=True):
    print("\n[5/6] Interactive Ingredient Explorer (HTML)...")

    rows = []
    for n, d in G.nodes(data=True):
        if d.get('type') != 'ingredient':
            continue
        name = d.get('name', '')
        if not isinstance(name, str):
            continue
        idx = node2idx.get(n)
        if idx is None:
            continue
        label, group = rec._get_sub_category(name)
        mol_count    = len(rec._mol_neighbors.get(n, set()))
        deg          = G.degree(n)
        rows.append({
            'emb':       embeddings[idx],
            'name':      name.replace('_', ' ').title(),
            'raw_name':  name,
            'category':  label or 'Unknown',
            'group':     group or 'Unknown',
            'molecules': mol_count,
            'degree':    deg,
        })

    X  = np.array([r['emb'] for r in rows])
    df = pd.DataFrame(rows).drop(columns=['emb'])

    print(f"     Reducing {X.shape[0]} vectors ({'t-SNE' if use_tsne else 'PCA'})…")
    xy = reduce_2d(X, use_tsne=use_tsne)
    df['x'], df['y'] = xy[:, 0], xy[:, 1]


    df['color_hex'] = df['category'].map(lambda c: PALETTE.get(c, '#374151'))

    fig = px.scatter(
        df, x='x', y='y',
        color='category',
        color_discrete_map=PALETTE,
        hover_name='name',
        hover_data={
            'x': False, 'y': False,
            'category': True, 'group': True,
            'molecules': True, 'degree': True,
        },
        template='plotly_dark',
        title='GastroGraph · Interactive Ingredient Explorer',
        opacity=0.82,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.4, color='white')))
    fig.update_layout(
        width=1400, height=900,
        paper_bgcolor='#0f1117', plot_bgcolor='#1a1d2e',
        font=dict(family='Inter, sans-serif', size=13, color=TEXT),
        title_font=dict(size=22, color=TEXT),
        legend=dict(
            title='Category', font=dict(size=10),
            bgcolor='#1a1d2e', bordercolor='#2d3748',
            itemsizing='constant',
        ),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        dragmode='pan',
    )
    os.makedirs(VIZ_INTER, exist_ok=True)
    out = os.path.join(VIZ_INTER, 'explorer.html')
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  ✓  Saved → {out}")
    return out


def plot_interactive_network(G, embeddings, node2idx, rec, use_tsne=True,
                             max_cooccur=4000, max_contains=8000):
    print("\n[6/6] Interactive Ingredient–Molecule Network (HTML)...")

    valid_nodes = [n for n in G.nodes() if n in node2idx]
    X = np.array([embeddings[node2idx[n]] for n in valid_nodes])

    print(f"     Reducing {len(valid_nodes)} nodes ({'t-SNE' if use_tsne else 'PCA'})…")
    xy = reduce_2d(X, use_tsne=use_tsne, n_pca=min(50, X.shape[1]))
    pos = {n: xy[i] for i, n in enumerate(valid_nodes)}


    ing_x, ing_y, ing_text, ing_col, ing_sz = [], [], [], [], []
    mol_x, mol_y, mol_text = [], [], []

    for n in valid_nodes:
        d    = G.nodes[n]
        ntype = d.get('type', '')
        nm   = str(d.get('name', n)).replace('_', ' ').title()
        p    = pos[n]

        if ntype == 'ingredient':
            label, _ = rec._get_sub_category(str(d.get('name', '')))
            col = PALETTE.get(label or 'Unknown', '#374151')
            deg = G.degree(n)
            ing_x.append(p[0]); ing_y.append(p[1])
            ing_text.append(f"<b>{nm}</b><br>Category: {label or 'Unknown'}<br>Degree: {deg}")
            ing_col.append(col)
            ing_sz.append(max(5, min(14, 5 + np.log1p(deg))))
        else:
            mol_x.append(p[0]); mol_y.append(p[1])
            mol_text.append(f"<b>{nm}</b> [molecule]")


    cooc_x,  cooc_y  = [], []
    cont_x,  cont_y  = [], []
    cooc_cnt = cont_cnt = 0

    import random; random.seed(42)
    edge_list = list(G.edges(data=True))
    random.shuffle(edge_list)

    for u, v, d in edge_list:
        if u not in pos or v not in pos:
            continue
        etype = d.get('type', '')
        p0, p1 = pos[u], pos[v]
        if etype == 'CO_OCCURS' and cooc_cnt < max_cooccur:
            cooc_x += [p0[0], p1[0], None]
            cooc_y += [p0[1], p1[1], None]
            cooc_cnt += 1
        elif etype == 'CONTAINS' and cont_cnt < max_contains:
            cont_x += [p0[0], p1[0], None]
            cont_y += [p0[1], p1[1], None]
            cont_cnt += 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cooc_x, y=cooc_y, mode='lines', name='Co-occurs (ingr–ingr)',
        line=dict(color='rgba(99,102,241,0.18)', width=0.6),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=cont_x, y=cont_y, mode='lines', name='Contains (ingr–mol)',
        line=dict(color='rgba(249,115,22,0.22)', width=0.5),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=mol_x, y=mol_y, mode='markers', name='Flavour Molecule',
        marker=dict(color='#f97316', size=4.5, opacity=0.55,
                    line=dict(width=0.3, color='white')),
        text=mol_text, hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=ing_x, y=ing_y, mode='markers', name='Ingredient',
        marker=dict(color=ing_col, size=ing_sz, opacity=0.85,
                    line=dict(width=0.4, color='white')),
        text=ing_text, hoverinfo='text'
    ))

    fig.update_layout(
        title='GastroGraph · Ingredient & Molecule Network',
        template='plotly_dark',
        paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        font=dict(family='Inter, sans-serif', size=13, color=TEXT),
        title_font=dict(size=22, color=TEXT),
        width=1400, height=950,
        showlegend=True,
        legend=dict(bgcolor='#1a1d2e', bordercolor='#2d3748', font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode='pan',
        margin=dict(t=60, b=10, l=10, r=10),
    )

    os.makedirs(VIZ_INTER, exist_ok=True)
    out = os.path.join(VIZ_INTER, 'network.html')
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  ✓  Saved → {out}")
    return out


def plot_interactive_heatmap(G, rec, min_molecules=3):
    """
    Pre-computes the full pairwise Jaccard molecule-overlap matrix for every
    ingredient that has ≥ min_molecules FlavorDB compounds, embeds it as JSON,
    then writes a self-contained HTML with a live search-and-select panel so
    the user can pick any subset of ingredients and see their heatmap instantly.
    """
    print(f"\n[7/7] Interactive Molecule-Overlap Heatmap (HTML — all ingredients)...")


    name2node = {}
    for n, d in G.nodes(data=True):
        if d.get('type') == 'ingredient':
            nm = d.get('name', '')
            if isinstance(nm, str):
                name2node[nm] = n

    candidates = [
        (nm, node)
        for nm, node in name2node.items()
        if len(rec._mol_neighbors.get(node, set())) >= min_molecules
    ]

    candidates.sort(key=lambda x: -len(rec._mol_neighbors.get(x[1], set())))

    names = [nm  for nm, _ in candidates]
    nodes = [nid for _, nid in candidates]
    N = len(names)
    print(f"     Building {N}×{N} Jaccard matrix ({N} ingredients with ≥{min_molecules} molecules)...")


    mat = [[0.0] * N for _ in range(N)]
    for i in range(N):
        mat[i][i] = 1.0
        for j in range(i + 1, N):
            score = rec._molecule_overlap_score(nodes[i], nodes[j])
            mat[i][j] = score
            mat[j][i] = score


    mol_counts = [len(rec._mol_neighbors.get(nid, set())) for nid in nodes]
    categories = []
    for nm in names:
        label, _ = rec._get_sub_category(nm)
        categories.append(label or 'Unknown')


    display = [nm.replace('_', ' ').title() for nm in names]


    pal = [PALETTE.get(cat, '#374151') for cat in categories]

    import json
    data_json = json.dumps({
        'names':      display,
        'raw':        names,
        'matrix':     mat,
        'mol_counts': mol_counts,
        'categories': categories,
        'palette':    pal,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GastroGraph · Molecule Overlap Heatmap</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.1/plotly.min.js"></script>
<style>
  :root {{
    --bg:      #0f1117;
    --panel:   #1a1d2e;
    --border:  #2d3748;
    --text:    #e2e8f0;
    --accent:  #6366f1;
    --accent2: #818cf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body     {{ background: var(--bg); color: var(--text); font-family: 'Inter', 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  header   {{ padding: 14px 24px; background: var(--panel); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  header h1 {{ font-size: 1.15rem; font-weight: 700; color: var(--text); }}
  header p  {{ font-size: 0.78rem; color: #94a3b8; }}
  .badge   {{ background: var(--accent); color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; font-weight: 600; }}
  .body    {{ display: flex; flex: 1; overflow: hidden; }}

  /* ── Sidebar ── */
  .sidebar {{ width: 300px; min-width: 260px; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 14px; gap: 10px; overflow: hidden; }}
  .sidebar h2 {{ font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }}
  .search-wrap {{ position: relative; }}
  #search    {{ width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; color: var(--text); font-size: 0.88rem; outline: none; transition: border .2s; }}
  #search:focus {{ border-color: var(--accent); }}
  .dropdown  {{ position: absolute; z-index: 99; top: calc(100% + 4px); left: 0; right: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; max-height: 220px; overflow-y: auto; display: none; box-shadow: 0 8px 24px rgba(0,0,0,.5); }}
  .dropdown.open {{ display: block; }}
  .dd-item   {{ padding: 7px 12px; font-size: 0.83rem; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: background .15s; }}
  .dd-item:hover {{ background: #252840; }}
  .dd-dot    {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
  .dd-cat    {{ color: #64748b; font-size: 0.73rem; flex-shrink: 0; }}

  /* ── Selected chips ── */
  .selected-label  {{ font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }}
  .selected-count  {{ font-size: 0.75rem; color: #64748b; }}
  .chips  {{ display: flex; flex-wrap: wrap; gap: 5px; overflow-y: auto; flex: 1; align-content: flex-start; }}
  .chip   {{ display: flex; align-items: center; gap: 5px; background: #252840; border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px; font-size: 0.78rem; cursor: pointer; transition: background .15s; }}
  .chip:hover {{ background: #2d3155; }}
  .chip .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .chip .x   {{ color: #64748b; font-size: 0.85rem; margin-left: 2px; }}

  /* ── Quick selectors ── */
  .quick  {{ display: flex; flex-wrap: wrap; gap: 5px; }}
  .qbtn   {{ background: transparent; border: 1px solid var(--border); border-radius: 6px; color: #94a3b8; font-size: 0.72rem; padding: 4px 9px; cursor: pointer; transition: all .15s; }}
  .qbtn:hover {{ background: var(--accent); border-color: var(--accent); color: white; }}
  .qbtn.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
  .qbtn-all    {{ border-color: #06b6d4; color: #06b6d4; font-weight: 600; }}
  .qbtn-all:hover, .qbtn-all.active {{ background: #06b6d4; border-color: #06b6d4; color: #0f172a; }}

  .divider {{ border: none; border-top: 1px solid var(--border); }}

  /* ── Clear button ── */
  #clear-btn {{ width: 100%; background: transparent; border: 1px solid var(--border); border-radius: 8px; color: #ef4444; font-size: 0.8rem; padding: 7px; cursor: pointer; transition: all .15s; }}
  #clear-btn:hover {{ background: rgba(239,68,68,.1); }}

  /* ── Chart pane ── */
  .chart-wrap {{ flex: 1; position: relative; overflow: hidden; }}
  #heatmap    {{ width: 100%; height: 100%; }}

  /* ── Empty state ── */
  .empty {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: #475569; pointer-events: none; }}
  .empty svg {{ opacity: .18; }}
  .empty p  {{ font-size: 1rem; }}
  .empty small {{ font-size: 0.82rem; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>GastroGraph &nbsp;·&nbsp; Molecule-Overlap Heatmap</h1>
    <p>Jaccard similarity of shared FlavorDB compounds · hover any cell for details</p>
  </div>
  <span class="badge" id="total-badge">Loading…</span>
</header>

<div class="body">

  <!-- ── Sidebar ────────────────────────────────── -->
  <div class="sidebar">
    <h2>Ingredients</h2>

    <!-- search -->
    <div class="search-wrap">
      <input id="search" type="text" placeholder="Search & add ingredients…" autocomplete="off">
      <div class="dropdown" id="dropdown"></div>
    </div>

    <!-- quick presets -->
    <h2>Presets</h2>
    <div class="quick">
      <button class="qbtn qbtn-all" data-preset="all">⊞ All ({N})</button>
      <button class="qbtn" data-preset="top20">Top 20 richest</button>
      <button class="qbtn" data-preset="herbs">Herbs &amp; Spices</button>
      <button class="qbtn" data-preset="citrus">Citrus</button>
      <button class="qbtn" data-preset="dairy">Dairy</button>
      <button class="qbtn" data-preset="meat">Meat &amp; Fish</button>
      <button class="qbtn" data-preset="sweets">Sweets</button>
      <button class="qbtn" data-preset="drinks">Drinks</button>
      <button class="qbtn" data-preset="nuts">Nuts &amp; Seeds</button>
    </div>

    <hr class="divider">

    <!-- selected chips -->
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span class="selected-label">Selected</span>
      <span class="selected-count" id="sel-count">0 ingredients</span>
    </div>
    <div class="chips" id="chips"></div>

    <button id="clear-btn">✕ Clear all</button>
  </div>

  <!-- ── Chart ─────────────────────────────────── -->
  <div class="chart-wrap">
    <div id="heatmap"></div>
    <div class="empty" id="empty-state">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4">
        <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
        <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
      </svg>
      <p>Select ingredients to compare</p>
      <small>Use the search box or a preset on the left</small>
    </div>
  </div>

</div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const DATA = {data_json};
// Index for fast lookup
const nameToIdx = {{}};
DATA.names.forEach((n,i) => nameToIdx[n] = i);
const rawToIdx  = {{}};
DATA.raw.forEach((n,i) => rawToIdx[n]  = i);

document.getElementById('total-badge').textContent = DATA.names.length + ' ingredients';

// ── State ─────────────────────────────────────────────────────────────────
let selected = [];   // array of display name strings

// ── Presets ───────────────────────────────────────────────────────────────
const PRESETS = {{
  all:    () => DATA.names,
  top20:  () => DATA.names.slice(0, 20),
  herbs:  () => DATA.names.filter((_,i) => ['Herb','Spice','Allium','Flower'].includes(DATA.categories[i])),
  citrus: () => DATA.names.filter((_,i) => DATA.categories[i] === 'Citrus'),
  dairy:  () => DATA.names.filter((_,i) => ['Dairy Liquid','Dairy Soft','Cheese'].includes(DATA.categories[i])),
  meat:   () => DATA.names.filter((_,i) => ['Meat','Seafood'].includes(DATA.categories[i])),
  sweets: () => DATA.names.filter((_,i) => ['Sweetener','Chocolate','Fruit','Bakery/Dessert/Snack'].includes(DATA.categories[i])),
  drinks: () => DATA.names.filter((_,i) => ['Alcohol','Beverage','Coffee/Tea'].includes(DATA.categories[i])),
  nuts:   () => DATA.names.filter((_,i) => ['Tree Nut','Seed','Legume'].includes(DATA.categories[i])),
}};

// ── Dropdown search ───────────────────────────────────────────────────────
const searchEl   = document.getElementById('search');
const dropEl     = document.getElementById('dropdown');

function renderDropdown(query) {{
  const q = query.toLowerCase();
  const hits = DATA.names.filter(n => n.toLowerCase().includes(q)).slice(0, 40);
  dropEl.innerHTML = hits.map(n => {{
    const i   = nameToIdx[n];
    const col = DATA.palette[i];
    const cat = DATA.categories[i];
    const mol = DATA.mol_counts[i];
    const sel = selected.includes(n);
    return `<div class="dd-item" data-name="${{n}}" style="opacity:${{sel?0.4:1}}">
      <span class="dd-dot" style="background:${{col}}"></span>
      <span>${{n}}</span>
      <span class="dd-cat" style="margin-left:auto">${{cat}} · ${{mol}}🧪</span>
      ${{sel ? '✓' : ''}}
    </div>`;
  }}).join('');
  dropEl.classList.toggle('open', hits.length > 0 && q.length > 0);
  dropEl.querySelectorAll('.dd-item').forEach(el => {{
    el.addEventListener('mousedown', e => {{
      e.preventDefault();
      toggleIngredient(el.dataset.name);
      searchEl.value = '';
      dropEl.classList.remove('open');
    }});
  }});
}}

searchEl.addEventListener('input', () => renderDropdown(searchEl.value));
searchEl.addEventListener('focus', () => {{if (searchEl.value) renderDropdown(searchEl.value);}});
document.addEventListener('click', e => {{
  if (!searchEl.contains(e.target) && !dropEl.contains(e.target))
    dropEl.classList.remove('open');
}});

// ── Chips (selected list) ─────────────────────────────────────────────────
function renderChips() {{
  const el = document.getElementById('chips');
  document.getElementById('sel-count').textContent = selected.length + ' ingredient' + (selected.length===1?'':'s');
  el.innerHTML = selected.map(n => {{
    const i   = nameToIdx[n];
    const col = DATA.palette[i];
    return `<div class="chip" data-name="${{n}}">
      <span class="dot" style="background:${{col}}"></span>
      <span>${{n}}</span>
      <span class="x">✕</span>
    </div>`;
  }}).join('');
  el.querySelectorAll('.chip').forEach(el => {{
    el.addEventListener('click', () => toggleIngredient(el.dataset.name));
  }});
}}

// ── Toggle selection ──────────────────────────────────────────────────────
function toggleIngredient(name) {{
  const idx = selected.indexOf(name);
  if (idx === -1) selected.push(name);
  else            selected.splice(idx, 1);
  renderChips();
  renderHeatmap();
}}

function setSelected(arr) {{
  selected = [...new Set(arr)];
  renderChips();
  renderHeatmap();
}}

// ── Preset buttons ────────────────────────────────────────────────────────
document.querySelectorAll('.qbtn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const preset = btn.dataset.preset;
    const hits   = PRESETS[preset]();
    const subset = preset === 'all' ? hits : hits.slice(0, 50);  // 'all' loads every ingredient; others cap at 50
    setSelected(subset);
    document.querySelectorAll('.qbtn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }});
}});

document.getElementById('clear-btn').addEventListener('click', () => {{
  setSelected([]);
  document.querySelectorAll('.qbtn').forEach(b => b.classList.remove('active'));
}});

// ── Heatmap rendering (Plotly) ────────────────────────────────────────────
const emptyEl = document.getElementById('empty-state');

function renderHeatmap() {{
  if (selected.length < 2) {{
    emptyEl.style.display = 'flex';
    Plotly.purge('heatmap');
    return;
  }}
  emptyEl.style.display = 'none';

  const idxs  = selected.map(n => nameToIdx[n]);
  const labels = selected;
  const n = selected.length;

  const zData = [];
  const textData = [];
  const customData = [];
  for (let i = 0; i < n; i++) {{
    const row = [];
    const trow = [];
    const crow = [];
    for (let j = 0; j < n; j++) {{
      const v = DATA.matrix[idxs[i]][idxs[j]];
      row.push(v);
      trow.push((v*100).toFixed(1) + '%');
      crow.push([labels[i], labels[j], DATA.mol_counts[idxs[i]], DATA.mol_counts[idxs[j]], DATA.categories[idxs[i]], DATA.categories[idxs[j]]]);
    }}
    zData.push(row);
    textData.push(trow);
    customData.push(crow);
  }}

  const trace = {{
    type: 'heatmap',
    z: zData,
    x: labels,
    y: labels,
    text: textData,
    customdata: customData,
    texttemplate: '%{{text}}',
    hovertemplate:
      '<b>%{{customdata[0]}}</b> vs <b>%{{customdata[1]}}</b><br>' +
      'Jaccard overlap: <b>%{{text}}</b><br>' +
      'Molecules — %{{customdata[0]}}: %{{customdata[2]}} · %{{customdata[1]}}: %{{customdata[3]}}<br>' +
      'Category — %{{customdata[4]}} · %{{customdata[5]}}<extra></extra>',
    colorscale: [
      [0,    '#fef9c3'],
      [0.25, '#fde68a'],
      [0.4,  '#34d399'],
      [0.6,  '#059669'],
      [0.8,  '#065f46'],
      [0.95, '#1e3a2f'],
      [1.0,  '#0f1117'],
    ],
    zmin: 0,
    zmax: 0.7,
    showscale: true,
    colorbar: {{
      title: {{
        text: 'Jaccard Overlap',
        font: {{color: '#e2e8f0', size: 12}},
        side: 'right'
      }},
      tickfont: {{color: '#e2e8f0'}},
      thickness: 16,
      len: 0.85,
    }},
  }};

  // font size scales down with count
  const fs = n <= 15 ? 11 : n <= 30 ? 9 : n <= 50 ? 7 : 5;

  const layout = {{
    paper_bgcolor: '#0f1117',
    plot_bgcolor:  '#1a1d2e',
    font: {{family: 'Inter, Segoe UI, sans-serif', color: '#e2e8f0', size: fs}},
    margin: {{t: 30, b: 120, l: 160, r: 100}},
    xaxis: {{
      tickangle: -45,
      tickfont: {{size: fs}},
      gridcolor: '#2d3748',
      zeroline: false,
    }},
    yaxis: {{
      autorange: 'reversed',
      tickfont: {{size: fs}},
      gridcolor: '#2d3748',
      zeroline: false,
    }},
    hoverlabel: {{
      bgcolor: '#1e2235',
      bordercolor: '#4f46e5',
      font: {{size: 13, color: '#e2e8f0'}},
    }},
    dragmode: 'pan',
  }};

  const config = {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['select2d','lasso2d'],
  }};

  Plotly.react('heatmap', [trace], layout, config);
}}

// ── No auto-boot: user selects ingredients manually ──────────────────────
</script>
</body>
</html>"""

    os.makedirs(VIZ_INTER, exist_ok=True)
    out = os.path.join(VIZ_INTER, 'molecule_heatmap.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓  Saved → {out}  ({N} ingredients, {N*N:,} cells pre-computed)")
    return out


def plot_substitute_score_heatmap(G, embeddings, node2idx, rec, min_molecules=3):
    """
    Interactive HTML heatmap showing the composite substitute score:
        score = 0.30 * mol_overlap (Jaccard)
              + 0.40 * category_bonus (1.0 same group, 0.5 same family, 0.0 diff)
              + 0.30 * embedding_cosine (normalised to [0,1])

    Same interactive UI as viz_heatmap_interactive.html but with a distinct
    green-yellow colorscale and breakdown tooltip.
    """
    print(f"\n[8/8] Interactive Composite Substitute Score Heatmap...")


    name2node = {}
    for n, d in G.nodes(data=True):
        if d.get('type') == 'ingredient':
            nm = d.get('name', '')
            if isinstance(nm, str):
                name2node[nm] = n

    candidates = [
        (nm, node)
        for nm, node in name2node.items()
        if len(rec._mol_neighbors.get(node, set())) >= min_molecules
    ]
    candidates.sort(key=lambda x: -len(rec._mol_neighbors.get(x[1], set())))

    names = [nm  for nm, _ in candidates]
    nodes = [nid for _, nid in candidates]
    N = len(names)
    print(f"     Building {N}×{N} composite score matrix ({N} ingredients)...")


    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim_bulk

    idxs = [node2idx[n] for n in nodes]
    emb_mat = embeddings[idxs]
    cos_mat = cos_sim_bulk(emb_mat)
    cos_norm_mat = (cos_mat + 1) / 2


    groups = []
    categories = []
    for nm in names:
        label, grp = rec._get_sub_category(nm)
        groups.append(grp)
        categories.append(label or 'Unknown')


    mol_mat  = [[0.0] * N for _ in range(N)]
    cat_mat  = [[0.0] * N for _ in range(N)]
    comp_mat = [[0.0] * N for _ in range(N)]

    for i in range(N):
        mol_mat[i][i]  = 1.0
        cat_mat[i][i]  = 1.0
        comp_mat[i][i] = 1.0
        for j in range(i + 1, N):
            mol  = rec._molecule_overlap_score(nodes[i], nodes[j])
            catb = rec._category_bonus(groups[i], groups[j])
            cosn = float(cos_norm_mat[i][j])
            cmp  = 0.30 * mol + 0.40 * catb + 0.30 * cosn

            mol_mat[i][j]  = mol_mat[j][i]  = mol
            cat_mat[i][j]  = cat_mat[j][i]  = catb
            comp_mat[i][j] = comp_mat[j][i] = cmp


    mol_counts = [len(rec._mol_neighbors.get(nid, set())) for nid in nodes]
    display    = [nm.replace('_', ' ').title() for nm in names]
    pal        = [PALETTE.get(cat, '#374151') for cat in categories]

    import json
    data_json = json.dumps({
        'names':      display,
        'raw':        names,
        'matrix':     comp_mat,
        'mol_matrix': mol_mat,
        'cat_matrix': cat_mat,
        'mol_counts': mol_counts,
        'categories': categories,
        'groups':     groups,
        'palette':    pal,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GastroGraph · Substitute Score Heatmap</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.1/plotly.min.js"></script>
<style>
  :root {{
    --bg:      #0f1117;
    --panel:   #1a1d2e;
    --border:  #2d3748;
    --text:    #e2e8f0;
    --accent:  #10b981;
    --accent2: #34d399;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body     {{ background: var(--bg); color: var(--text); font-family: 'Inter', 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  header   {{ padding: 14px 24px; background: var(--panel); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  header h1 {{ font-size: 1.15rem; font-weight: 700; color: var(--text); }}
  header p  {{ font-size: 0.78rem; color: #94a3b8; }}
  .badge   {{ background: var(--accent); color: #0f1117; font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; font-weight: 700; }}
  .formula {{ font-size: 0.72rem; color: #64748b; background: #1e2235; border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; font-family: monospace; }}
  .body    {{ display: flex; flex: 1; overflow: hidden; }}

  .sidebar {{ width: 300px; min-width: 260px; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 14px; gap: 10px; overflow: hidden; }}
  .sidebar h2 {{ font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }}
  .search-wrap {{ position: relative; }}
  #search    {{ width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; color: var(--text); font-size: 0.88rem; outline: none; transition: border .2s; }}
  #search:focus {{ border-color: var(--accent); }}
  .dropdown  {{ position: absolute; z-index: 99; top: calc(100% + 4px); left: 0; right: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; max-height: 220px; overflow-y: auto; display: none; box-shadow: 0 8px 24px rgba(0,0,0,.5); }}
  .dropdown.open {{ display: block; }}
  .dd-item   {{ padding: 7px 12px; font-size: 0.83rem; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: background .15s; }}
  .dd-item:hover {{ background: #252840; }}
  .dd-dot    {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
  .dd-cat    {{ color: #64748b; font-size: 0.73rem; flex-shrink: 0; }}

  .selected-label  {{ font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }}
  .selected-count  {{ font-size: 0.75rem; color: #64748b; }}
  .chips  {{ display: flex; flex-wrap: wrap; gap: 5px; overflow-y: auto; flex: 1; align-content: flex-start; }}
  .chip   {{ display: flex; align-items: center; gap: 5px; background: #252840; border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px; font-size: 0.78rem; cursor: pointer; transition: background .15s; }}
  .chip:hover {{ background: #2d3155; }}
  .chip .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .chip .x   {{ color: #64748b; font-size: 0.85rem; margin-left: 2px; }}

  .quick  {{ display: flex; flex-wrap: wrap; gap: 5px; }}
  .qbtn   {{ background: transparent; border: 1px solid var(--border); border-radius: 6px; color: #94a3b8; font-size: 0.72rem; padding: 4px 9px; cursor: pointer; transition: all .15s; }}
  .qbtn:hover {{ background: var(--accent); border-color: var(--accent); color: #0f1117; }}
  .qbtn.active {{ background: var(--accent); border-color: var(--accent); color: #0f1117; }}
  .qbtn-all    {{ border-color: #06b6d4; color: #06b6d4; font-weight: 600; }}
  .qbtn-all:hover, .qbtn-all.active {{ background: #06b6d4; border-color: #06b6d4; color: #0f172a; }}

  .divider {{ border: none; border-top: 1px solid var(--border); }}
  #clear-btn {{ width: 100%; background: transparent; border: 1px solid var(--border); border-radius: 8px; color: #ef4444; font-size: 0.8rem; padding: 7px; cursor: pointer; transition: all .15s; }}
  #clear-btn:hover {{ background: rgba(239,68,68,.1); }}

  .chart-wrap {{ flex: 1; position: relative; overflow: hidden; }}
  #heatmap    {{ width: 100%; height: 100%; }}
  .empty {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: #475569; pointer-events: none; }}
  .empty svg {{ opacity: .18; }}
  .empty p  {{ font-size: 1rem; }}
  .empty small {{ font-size: 0.82rem; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>GastroGraph &nbsp;·&nbsp; Composite Substitute Score</h1>
    <p>Overall substitutability · hover any cell to see mol / category / embedding breakdown</p>
  </div>
  <div class="formula">score = 0.30×mol + 0.40×category + 0.30×embedding</div>
  <span class="badge" id="total-badge">Loading…</span>
</header>

<div class="body">

  <!-- ── Sidebar ── -->
  <div class="sidebar">
    <h2>Ingredients</h2>
    <div class="search-wrap">
      <input id="search" type="text" placeholder="Search &amp; add ingredients…" autocomplete="off">
      <div class="dropdown" id="dropdown"></div>
    </div>

    <h2>Presets</h2>
    <div class="quick">
      <button class="qbtn qbtn-all" data-preset="all">⊞ All ({N})</button>
      <button class="qbtn" data-preset="top20">Top 20 richest</button>
      <button class="qbtn" data-preset="grains">Grains &amp; Starch</button>
      <button class="qbtn" data-preset="herbs">Herbs &amp; Spices</button>
      <button class="qbtn" data-preset="citrus">Citrus</button>
      <button class="qbtn" data-preset="dairy">Dairy</button>
      <button class="qbtn" data-preset="meat">Meat &amp; Fish</button>
      <button class="qbtn" data-preset="sweets">Sweets</button>
      <button class="qbtn" data-preset="nuts">Nuts &amp; Seeds</button>
    </div>

    <hr class="divider">

    <div style="display:flex;justify-content:space-between;align-items:center">
      <span class="selected-label">Selected</span>
      <span class="selected-count" id="sel-count">0 ingredients</span>
    </div>
    <div class="chips" id="chips"></div>

    <button id="clear-btn">✕ Clear all</button>
  </div>

  <!-- ── Chart ── -->
  <div class="chart-wrap">
    <div id="heatmap"></div>
    <div class="empty" id="empty-state">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4">
        <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
        <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
      </svg>
      <p>Select ingredients to compare</p>
      <small>Use the search box or a preset on the left</small>
    </div>
  </div>

</div>

<script>
// ── Embedded data ────────────────────────────────────────────────────────────
const DATA = {data_json};
const nameToIdx = {{}};
DATA.names.forEach((n,i) => nameToIdx[n] = i);

document.getElementById('total-badge').textContent = DATA.names.length + ' ingredients';

// ── State ────────────────────────────────────────────────────────────────────
let selected = [];

// ── Presets ──────────────────────────────────────────────────────────────────
const PRESETS = {{
  all:    () => DATA.names,
  top20:  () => DATA.names.slice(0, 20),
  grains: () => DATA.names.filter((_,i) => ['Whole Grain','Starch/Flour','Bread','Cereal/Crop/Bean'].includes(DATA.categories[i])),
  herbs:  () => DATA.names.filter((_,i) => ['Herb','Spice','Allium','Flower'].includes(DATA.categories[i])),
  citrus: () => DATA.names.filter((_,i) => DATA.categories[i] === 'Citrus'),
  dairy:  () => DATA.names.filter((_,i) => ['Dairy Liquid','Dairy Soft','Cheese','Dairy'].includes(DATA.categories[i])),
  meat:   () => DATA.names.filter((_,i) => ['Meat','Seafood'].includes(DATA.categories[i])),
  sweets: () => DATA.names.filter((_,i) => ['Sweetener','Chocolate','Fruit','Bakery/Dessert/Snack'].includes(DATA.categories[i])),
  nuts:   () => DATA.names.filter((_,i) => ['Tree Nut','Seed','Legume'].includes(DATA.categories[i])),
}};

// ── Dropdown ─────────────────────────────────────────────────────────────────
const searchEl = document.getElementById('search');
const dropEl   = document.getElementById('dropdown');

function renderDropdown(query) {{
  const q = query.toLowerCase();
  const hits = DATA.names.filter(n => n.toLowerCase().includes(q)).slice(0, 40);
  dropEl.innerHTML = hits.map(n => {{
    const i   = nameToIdx[n];
    const col = DATA.palette[i];
    const cat = DATA.categories[i];
    const sel = selected.includes(n);
    return `<div class="dd-item" data-name="${{n}}" style="opacity:${{sel?0.4:1}}">
      <span class="dd-dot" style="background:${{col}}"></span>
      <span>${{n}}</span>
      <span class="dd-cat" style="margin-left:auto">${{cat}}</span>
      ${{sel ? '✓' : ''}}
    </div>`;
  }}).join('');
  dropEl.classList.toggle('open', hits.length > 0 && q.length > 0);
  dropEl.querySelectorAll('.dd-item').forEach(el => {{
    el.addEventListener('mousedown', e => {{
      e.preventDefault();
      toggleIngredient(el.dataset.name);
      searchEl.value = '';
      dropEl.classList.remove('open');
    }});
  }});
}}

searchEl.addEventListener('input', () => renderDropdown(searchEl.value));
searchEl.addEventListener('focus', () => {{ if (searchEl.value) renderDropdown(searchEl.value); }});
document.addEventListener('click', e => {{
  if (!searchEl.contains(e.target) && !dropEl.contains(e.target)) dropEl.classList.remove('open');
}});

// ── Chips ────────────────────────────────────────────────────────────────────
function renderChips() {{
  const el = document.getElementById('chips');
  document.getElementById('sel-count').textContent = selected.length + ' ingredient' + (selected.length===1?'':'s');
  el.innerHTML = selected.map(n => {{
    const i   = nameToIdx[n];
    const col = DATA.palette[i];
    return `<div class="chip" data-name="${{n}}">
      <span class="dot" style="background:${{col}}"></span>
      <span>${{n}}</span>
      <span class="x">✕</span>
    </div>`;
  }}).join('');
  el.querySelectorAll('.chip').forEach(el => {{
    el.addEventListener('click', () => toggleIngredient(el.dataset.name));
  }});
}}

function toggleIngredient(name) {{
  const idx = selected.indexOf(name);
  if (idx === -1) selected.push(name);
  else            selected.splice(idx, 1);
  renderChips();
  renderHeatmap();
}}

function setSelected(arr) {{
  selected = [...new Set(arr)];
  renderChips();
  renderHeatmap();
}}

// ── Preset buttons ────────────────────────────────────────────────────────────
document.querySelectorAll('.qbtn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const preset = btn.dataset.preset;
    const hits   = PRESETS[preset]();
    const subset = preset === 'all' ? hits : hits.slice(0, 50);
    setSelected(subset);
    document.querySelectorAll('.qbtn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }});
}});

document.getElementById('clear-btn').addEventListener('click', () => {{
  setSelected([]);
  document.querySelectorAll('.qbtn').forEach(b => b.classList.remove('active'));
}});

// ── Heatmap rendering ─────────────────────────────────────────────────────────
const emptyEl = document.getElementById('empty-state');

// Category bonus label
function catLabel(v) {{
  if (v >= 1.0) return 'Same group ✓✓';
  if (v >= 0.5) return 'Same family ✓';
  return 'Different ✗';
}}

function renderHeatmap() {{
  if (selected.length < 2) {{
    emptyEl.style.display = 'flex';
    Plotly.purge('heatmap');
    return;
  }}
  emptyEl.style.display = 'none';

  const idxs   = selected.map(n => nameToIdx[n]);
  const labels = selected;
  const n      = selected.length;

  const zData = [], textData = [], customData = [];
  for (let i = 0; i < n; i++) {{
    const row = [], trow = [], crow = [];
    for (let j = 0; j < n; j++) {{
      const v   = DATA.matrix[idxs[i]][idxs[j]];
      const mol = DATA.mol_matrix[idxs[i]][idxs[j]];
      const cat = DATA.cat_matrix[idxs[i]][idxs[j]];
      const cos = (v - 0.30*mol - 0.40*cat) / 0.30;  // back-solve embedding
      row.push(v);
      trow.push((v * 10).toFixed(1) + '/10');
      crow.push([
        labels[i], labels[j],
        DATA.categories[idxs[i]], DATA.categories[idxs[j]],
        (mol*100).toFixed(1),
        catLabel(cat),
        (Math.max(0, Math.min(1, cos))*100).toFixed(1),
        (v*100).toFixed(1)
      ]);
    }}
    zData.push(row);
    textData.push(trow);
    customData.push(crow);
  }}

  const trace = {{
    type: 'heatmap',
    z: zData,
    x: labels,
    y: labels,
    text: textData,
    customdata: customData,
    texttemplate: '%{{text}}',
    hovertemplate:
      '<b>%{{customdata[0]}}</b> vs <b>%{{customdata[1]}}</b><br>' +
      'Composite Score: <b>%{{customdata[7]}}%</b> (%{{text}})<br>' +
      '<br>' +
      '🧪 Molecule overlap:  <b>%{{customdata[4]}}%</b><br>' +
      '🏷️ Category match:    <b>%{{customdata[5]}}</b><br>' +
      '🔗 Embedding sim:     <b>%{{customdata[6]}}%</b><br>' +
      '<br>' +
      'Categories: %{{customdata[2]}} · %{{customdata[3]}}<extra></extra>',
    colorscale: [
      [0,    '#fef9c3'],
      [0.25, '#fde68a'],
      [0.4,  '#34d399'],
      [0.6,  '#059669'],
      [0.8,  '#065f46'],
      [0.95, '#1e3a2f'],
      [1.0,  '#0f1117'],
    ],
    zmin: 0,
    zmax: 1.0,
    showscale: true,
    colorbar: {{
      title: {{
        text: 'Composite Score',
        font: {{color: '#e2e8f0', size: 12}},
        side: 'right'
      }},
      tickfont: {{color: '#e2e8f0'}},
      tickformat: '.0%',
      thickness: 16,
      len: 0.85,
    }},
  }};

  const fs = n <= 15 ? 11 : n <= 30 ? 9 : n <= 50 ? 7 : 5;

  const layout = {{
    paper_bgcolor: '#0f1117',
    plot_bgcolor:  '#1a1d2e',
    font: {{family: 'Inter, Segoe UI, sans-serif', color: '#e2e8f0', size: fs}},
    margin: {{t: 30, b: 120, l: 160, r: 100}},
    xaxis: {{ tickangle: -45, tickfont: {{size: fs}}, gridcolor: '#2d3748', zeroline: false }},
    yaxis: {{ autorange: 'reversed', tickfont: {{size: fs}}, gridcolor: '#2d3748', zeroline: false }},
    hoverlabel: {{ bgcolor: '#1e2235', bordercolor: '#10b981', font: {{size: 13, color: '#e2e8f0'}} }},
    dragmode: 'pan',
  }};

  const config = {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['select2d','lasso2d'],
  }};

  Plotly.react('heatmap', [trace], layout, config);
}}

// ── No auto-boot: user selects ingredients manually ──────────────────────
</script>
</body>
</html>"""

    os.makedirs(VIZ_INTER, exist_ok=True)
    out = os.path.join(VIZ_INTER, 'substitute_score.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓  Saved → {out}  ({N} ingredients, {N*N:,} cells pre-computed)")
    return out


def main():

    parser = argparse.ArgumentParser(description='GastroGraph Visualization Suite')
    parser.add_argument('--ingredient', type=str, default='apple',
                        help='Ingredient to draw neighbourhood graph for (default: apple)')
    parser.add_argument('--skip-tsne', action='store_true',
                        help='Use fast PCA instead of slow t-SNE for 2-D reduction')
    args = parser.parse_args()

    use_tsne = not args.skip_tsne

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    G, embeddings, node2idx = load_resources()
    rec = get_recommender()

    outputs = []

    outputs.append(plot_tsne_universe(G, embeddings, node2idx, rec, use_tsne))
    outputs.append(plot_category_distribution(G, rec))
    outputs.append(plot_molecule_heatmap(G, rec))
    outputs.append(plot_neighbourhood(G, args.ingredient))
    outputs.append(plot_interactive_explorer(G, embeddings, node2idx, rec, use_tsne))
    outputs.append(plot_interactive_network(G, embeddings, node2idx, rec, use_tsne))
    outputs.append(plot_interactive_heatmap(G, rec))
    outputs.append(plot_substitute_score_heatmap(G, embeddings, node2idx, rec))


    print(f"\n{'='*60}")
    print("  ALL VISUALIZATIONS COMPLETE")
    print(f"{'='*60}")
    for p in outputs:
        if p:
            print(f"  → {p}")
    print()


if __name__ == '__main__':
    main()
