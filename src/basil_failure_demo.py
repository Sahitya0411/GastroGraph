import os
import sys
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR   = os.path.join(PROJECT_ROOT, "config")

embeddings = np.load(os.path.join(MODEL_DIR, "node_embeddings.npy"))

with open(os.path.join(MODEL_DIR, "node2idx.pkl"), "rb") as f:
    data = pickle.load(f)
node2idx = data["node2idx"]
idx2node = {v: k for k, v in node2idx.items()}

with open(os.path.join(MODEL_DIR, "gastro_graph.gpickle"), "rb") as f:
    G = pickle.load(f)

def find_node(name):
    for n, d in G.nodes(data=True):
        nm = d.get("name", "")
        if isinstance(nm, str) and nm.lower() == name.lower():
            return n
    return None

QUERY      = "basil"
basil_node = find_node(QUERY)
basil_vec  = embeddings[node2idx[basil_node]].reshape(1, -1)

ingredient_nodes = [n for n, d in G.nodes(data=True)
                    if d.get("type") == "ingredient" and n != basil_node]
ing_idxs  = [node2idx[n] for n in ingredient_nodes]
ing_names = [G.nodes[n].get("name", "") for n in ingredient_nodes]

raw_cosines  = cosine_similarity(basil_vec, embeddings[ing_idxs])[0]
sorted_order = np.argsort(raw_cosines)[::-1]

SEAFOOD_KW = {"caviar","salmon","anchovy","tuna","trout","herring","shrimp",
              "crab","lobster","squid","oyster","clam","mussel","sardine",
              "cod","bass","fish","seafood"}
MEAT_KW    = {"beef","pork","chicken","lamb","bacon","ham","turkey",
              "meat","steak","sausage"}
CHEESE_KW  = {"cheese"}
GRAIN_KW   = {"rice","wheat","corn","grain","oat","barley","rye"}
VEG_KW     = {"asparagus","pea","sweetcorn","carrot","spinach","broccoli"}
ALCOHOL_KW = {"wine","beer","whiskey","rum","vodka","gin","brandy","sake"}

def verdict(name):
    nl = name.lower()
    if any(k in nl for k in SEAFOOD_KW): return "Wrong (Seafood)"
    if any(k in nl for k in MEAT_KW):    return "Wrong (Meat)"
    if any(k in nl for k in CHEESE_KW):  return "Wrong (Cheese)"
    if any(k in nl for k in GRAIN_KW):   return "Wrong (Grain)"
    if any(k in nl for k in VEG_KW):     return "Wrong (Veg)"
    if any(k in nl for k in ALCOHOL_KW): return "Wrong (Alcohol)"
    return "ok"

print()
print("  BEFORE — Naive Cosine Similarity (top 10) for 'basil'")
print(f"  {'#':<4} {'Ingredient':<30} {'Cosine':<8} {'Verdict'}")
print("  " + "-"*60)
for rank, idx in enumerate(sorted_order[:10], 1):
    name  = ing_names[idx]
    score = raw_cosines[idx]
    v     = verdict(name)
    flag  = "⚠ " if v.startswith("Wrong") else "✓ "
    print(f"  {rank:<4} {name:<30} {score:.4f}   {flag}{v}")

import warnings, io, contextlib
warnings.filterwarnings("ignore")

from disjoint_context import DisjointContextRecommender

with contextlib.redirect_stdout(io.StringIO()):
    rec = DisjointContextRecommender(output_dir=MODEL_DIR,
                                     input_dir=DATA_DIR,
                                     config_dir=CONFIG_DIR)
    results = rec.get_substitutes(QUERY, top_k=5, show_complements=False)

func_subs = results.get("functional", [])

print()
print("  AFTER — GastroGraph Functional Substitutes for 'basil'")
print(f"  {'#':<4} {'Ingredient':<30} {'Score':>7} {'Mol%':>6} {'Embed':>7} {'Group'}")
print("  " + "-"*60)
for i, r in enumerate(func_subs, 1):
    print(f"  {i:<4} {r['name']:<30} {r['_func_score']*10:>6.1f}/10 "
          f"{(r['_mol_score'] or 0)*100:>5.0f}%  {r['cos_sim']:>6.3f}  {r.get('label','')}")
print()
