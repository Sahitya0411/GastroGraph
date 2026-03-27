import numpy as np
import networkx as nx
import pickle
import os
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class DisjointContextRecommender:

    FINE_CATEGORIES = []

    @staticmethod
    def _load_fine_categories_from_csv(csv_path):
        import csv as _csv
        from collections import OrderedDict

        if not os.path.exists(csv_path):
            print(f"[WARNING] fine_categories.csv not found at {csv_path}. "
                  f"FINE_CATEGORIES will be empty — all ingredients will be Unknown.")
            return []

        groups = OrderedDict()
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    pri = int(row['priority'])
                except (KeyError, ValueError):
                    continue
                kw   = row['keyword'].strip()
                cat  = row['category'].strip()
                fgrp = row['functional_group'].strip()
                if pri not in groups:
                    groups[pri] = ([], cat, fgrp)
                groups[pri][0].append(kw)

        result = [(kws, cat, fgrp) for (kws, cat, fgrp) in groups.values()]
        return result

    AROMA_INCOMPATIBLE = [
        ({'SEAFOOD', 'MEAT', 'EGG'}, {'FRUIT', 'CITRUS', 'SWEETENER', 'CHOCOLATE', 'COFFEE_TEA'}),
        ({'DAIRY_LIQUID', 'DAIRY_SOFT', 'CHEESE'}, {'SEAFOOD', 'MEAT'}),
        ({'SOLID_FAT', 'LIQUID_OIL'}, {'FRUIT', 'CITRUS', 'SEAFOOD'}),
        ({'HERB'}, {'SEAFOOD', 'MEAT', 'CHEESE', 'SOLID_FAT', 'DAIRY_LIQUID', 'SWEETENER', 'CHOCOLATE'}),
        ({'SPICE'}, {'SEAFOOD', 'MEAT', 'SOLID_FAT', 'DAIRY_LIQUID', 'SWEETENER', 'CHOCOLATE'}),
        ({'ALLIUM'}, {'FRUIT', 'CITRUS', 'SWEETENER', 'CHOCOLATE', 'COFFEE_TEA', 'DAIRY_LIQUID'}),
        ({'BREAD', 'STARCH_FLOUR', 'WHOLE_GRAIN'}, {'SEAFOOD', 'MEAT', 'DAIRY_LIQUID', 'FRUIT', 'CITRUS'}),
        ({'SWEETENER'}, {'SEAFOOD', 'MEAT', 'ALLIUM', 'SPICE', 'ACID'}),
    ]

    PRODUCT_KEYWORDS = [
        'pudding', 'mix', 'filling', 'soup', 'spread', 'flavoring', 'extract',
        'essence', 'powder', 'dried', 'canned', 'frozen', 'processed', 'prepared',
        'sandwich', 'pastry', 'pie_crust', 'topping', 'juice', 'beverage', 'drink',
        'chip', 'cracker', 'snack', 'pancakes', 'waffle', 'custard', 'muffin',
        'dressing', 'marinade', 'cereal', 'cooky', 'cookie', 'cake'
    ]

    def __init__(self, output_dir='models', input_dir='data', config_dir=None):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.config_dir = config_dir if config_dir is not None else input_dir
        self.embeddings_path = os.path.join(output_dir, 'node_embeddings.npy')
        self.mapping_path = os.path.join(output_dir, 'node2idx.pkl')
        self.graph_path = os.path.join(output_dir, 'gastro_graph.gpickle')
        self.external_cats_path = os.path.join(input_dir, 'dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv')

        self.external_mapping = {}
        self._sub_cat_cache = {}
        self._mol_cache = {}

        csv_path = os.path.join(self.config_dir, 'fine_categories.csv')
        self.fine_categories = self._load_fine_categories_from_csv(csv_path)
        self.load_resources()

    def load_resources(self):
        print("Loading embeddings and graph...")
        self.embeddings = np.load(self.embeddings_path)

        with open(self.mapping_path, 'rb') as f:
            data = pickle.load(f)
            self.node2idx = data['node2idx']
            self.idx2node = {v: k for k, v in self.node2idx.items()}

        with open(self.graph_path, 'rb') as f:
            self.G = pickle.load(f)

        if os.path.exists(self.external_cats_path):
            print(f"Loading external category mapping from {self.external_cats_path}...")
            df = pd.read_csv(self.external_cats_path)
            for _, row in df.iterrows():
                self.external_mapping[row['ingredient'].lower().strip()] = row['category']
            print(f"Loaded {len(self.external_mapping)} ingredient mappings.")

        print("Building molecule index...")
        self._build_molecule_index()
        print("Resources loaded.")

    def _build_molecule_index(self):
        self._mol_neighbors = {}
        for n, d in self.G.nodes(data=True):
            if d['type'] == 'ingredient':
                mols = set()
                for nbr in self.G.neighbors(n):
                    if self.G.nodes[nbr]['type'] == 'molecule':
                        mols.add(nbr)
                self._mol_neighbors[n] = mols

    def _get_sub_category(self, name):
        if not isinstance(name, str):
            return None, None
        if name in self._sub_cat_cache:
            return self._sub_cat_cache[name]

        name_lower = name.lower().strip()
        parts = set(name_lower.replace('-', '_').split('_'))

        ext_cat = self.external_mapping.get(name_lower)
        if not ext_cat:
            for w in name_lower.split('_'):
                if w in self.external_mapping:
                    ext_cat = self.external_mapping[w]
                    break

        name_tokens = set(name_lower.replace('-', '_').split('_'))
        for keywords, label, group in self.fine_categories:
            matched = False
            for kw in keywords:
                kw_tokens = set(kw.replace('-', '_').split('_'))
                if len(kw_tokens) == 1:
                    if kw in name_tokens:
                        matched = True
                        break
                else:
                    if kw_tokens.issubset(name_tokens):
                        matched = True
                        break
            if matched:
                result = (label, group)
                self._sub_cat_cache[name] = result
                return result

        if ext_cat:
            result = (ext_cat, ext_cat.upper().replace('/', '_').replace(' ', '_'))
            self._sub_cat_cache[name] = result
            return result

        self._sub_cat_cache[name] = (None, None)
        return None, None

    def _is_product(self, name):
        if not isinstance(name, str):
            return False
        return any(pk in name.lower() for pk in self.PRODUCT_KEYWORDS)

    def _is_aromatic_compatible(self, q_group, c_group):
        if q_group is None or c_group is None:
            return True
        for set_a, set_b in self.AROMA_INCOMPATIBLE:
            if (q_group in set_a and c_group in set_b) or \
               (q_group in set_b and c_group in set_a):
                return False
        return True

    def _molecule_overlap_score(self, node_a, node_b):
        mols_a = self._mol_neighbors.get(node_a, set())
        mols_b = self._mol_neighbors.get(node_b, set())
        if not mols_a or not mols_b:
            return 0.0

        shared = mols_a & mols_b
        if not shared:
            return 0.0

        def edge_w(node, mol):
            d = self.G.get_edge_data(node, mol)
            return d.get('weight', 1.0) if d else 0.0

        num = 0.0
        den = 0.0
        all_mols = mols_a | mols_b
        for m in all_mols:
            wa = edge_w(node_a, m)
            wb = edge_w(node_b, m)
            num += min(wa, wb)
            den += max(wa, wb)

        return num / den if den > 0 else 0.0

    def _cooccur_similarity(self, idx_a, idx_b):
        va = self.embeddings[idx_a].reshape(1, -1)
        vb = self.embeddings[idx_b].reshape(1, -1)
        return float(cosine_similarity(va, vb)[0][0])

    _BROAD_FAMILIES = [
        {'WHOLE_GRAIN', 'STARCH_FLOUR', 'BREAD', 'CEREAL_CROP_BEAN', 'BAKERY'},
        {'DAIRY_LIQUID', 'DAIRY_SOFT', 'CHEESE', 'DAIRY'},
        {'MEAT', 'SEAFOOD', 'EGG', 'PLANT_PROTEIN'},
        {'HERB', 'SPICE', 'ALLIUM', 'FLOWER'},
        {'FRUIT', 'CITRUS'},
        {'TREE_NUT', 'SEED', 'LEGUME'},
        {'SWEETENER', 'CHOCOLATE'},
        {'CONDIMENT', 'ACID', 'SAUCE_POWDER_DRESSING'},
        {'LIQUID_OIL', 'SOLID_FAT'},
        {'ROOT_VEG', 'LEAFY_GREEN', 'FRUITING_VEG',
         'BRASSICA', 'MUSHROOM', 'PLANT_VEG'},
        {'ALCOHOL', 'BEVERAGE', 'COFFEE_TEA'},
    ]

    def _category_bonus(self, group_a, group_b):
        if group_a is None or group_b is None:
            return 0.0
        if group_a == group_b:
            return 1.0
        for family in self._BROAD_FAMILIES:
            if group_a in family and group_b in family:
                return 0.5
        return 0.0

    def get_substitutes(self, query, top_k=5, show_complements=True):
        query_node = None
        for n, d in self.G.nodes(data=True):
            nm = d.get('name', '')
            if isinstance(nm, str) and nm.lower() == query.lower():
                query_node = n
                break

        if query_node is None:
            matches = [
                n for n, d in self.G.nodes(data=True)
                if query.lower() in str(d.get('name', '')).lower()
                and d['type'] == 'ingredient'
            ]
            if not matches:
                print(f"\n  Ingredient '{query}' not found in the graph.")
                print("  Tip: use underscores for multi-word names, e.g. 'olive_oil', 'soy_sauce'")
                return []
            matches.sort(key=lambda x: (self._is_product(self.G.nodes[x].get('name', '')),
                                         len(str(self.G.nodes[x].get('name', '')))))
            query_node = matches[0]
            resolved = self.G.nodes[query_node]['name']
            print(f"  Resolved '{query}' → '{resolved}'")

        query_name   = self.G.nodes[query_node]['name']
        query_idx    = self.node2idx[query_node]
        q_label, q_group = self._get_sub_category(query_name)
        q_is_product = self._is_product(query_name)

        ingredient_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d['type'] == 'ingredient' and n != query_node
        ]

        all_ingredient_idxs = [self.node2idx[n] for n in ingredient_nodes]
        q_vec         = self.embeddings[query_idx].reshape(1, -1)
        cand_matrix   = self.embeddings[all_ingredient_idxs]
        cosine_scores = cosine_similarity(q_vec, cand_matrix)[0]

        results = []
        for i, node in enumerate(ingredient_nodes):
            cname        = self.G.nodes[node]['name']
            c_label, c_group = self._get_sub_category(cname)
            c_is_product = self._is_product(cname)
            cos_sim      = float(cosine_scores[i])
            is_cooccur   = self.G.has_edge(query_node, node)

            results.append({
                'node': node,
                'name': cname,
                'label': c_label,
                'group': c_group,
                'is_product': c_is_product,
                'is_cooccur': is_cooccur,
                'cos_sim': cos_sim,
                '_mol_score': None,
            })

        func_subs = []
        if q_group:
            same_group = [r for r in results if r['group'] == q_group and not r['is_cooccur']]
            if not q_is_product:
                same_group = [r for r in same_group if not r['is_product']]

            for r in same_group:
                r['_mol_score'] = self._molecule_overlap_score(query_node, r['node'])
                cos_norm        = (r['cos_sim'] + 1) / 2
                cat_bonus       = self._category_bonus(q_group, r['group'])
                r['_func_score'] = 0.30 * r['_mol_score'] + 0.40 * cat_bonus + 0.30 * cos_norm

            same_group = [r for r in same_group if r['_func_score'] > 0.35]
            same_group.sort(key=lambda x: x['_func_score'], reverse=True)
            func_subs = same_group[:top_k]

        if not q_is_product:
            aromatic_pool = [r for r in results if not r['is_product']]
        else:
            aromatic_pool = results[:]

        aromatic_pool = [
            r for r in aromatic_pool
            if not r['is_cooccur']
            and r['group'] != q_group
            and self._is_aromatic_compatible(q_group, r['group'])
        ]

        aromatic_pool.sort(key=lambda x: x['cos_sim'], reverse=True)
        top_aromatic_pool = aromatic_pool[:3000]

        for r in top_aromatic_pool:
            r['_mol_score'] = self._molecule_overlap_score(query_node, r['node'])

        for r in top_aromatic_pool:
            r['_cos_norm']    = (r['cos_sim'] + 1) / 2
            cat_bonus         = self._category_bonus(q_group, r['group'])
            r['_aroma_final'] = (0.50 * r['_mol_score']
                                 + 0.20 * cat_bonus
                                 + 0.30 * r['_cos_norm'])

        aromatic_subs = [r for r in top_aromatic_pool if r['_mol_score'] > 0.01]
        aromatic_subs.sort(key=lambda x: x['_aroma_final'], reverse=True)
        aromatic_subs = aromatic_subs[:top_k]

        complements = [r for r in results if r['is_cooccur']]
        complements.sort(key=lambda x: x['cos_sim'], reverse=True)

        print(f"\n{'='*55}")
        print(f"  GastroGraph Recommender: {query_name}")
        if q_label:
            print(f"  Category: {q_label}  |  Group: {q_group}")
        else:
            print(f"  Category: Unknown")
        print(f"{'='*55}")

        print(f"\n[Functional / Practical Substitutes]")
        print(f"  (Same culinary category — drop-in role replacement)")
        if func_subs:
            for i, r in enumerate(func_subs):
                score_display = r['_func_score'] * 10
                cat_tag = f" [{r['label']}]" if r['label'] else ""
                mol_pct = r['_mol_score'] * 100
                print(f"  {i+1:2}. {r['name']:<30}  Score: {score_display:.1f}/10  (mol:{mol_pct:.0f}% + cat:✓ + embed:{r['cos_sim']:.2f}){cat_tag}")
        else:
            print("  ⚠  No high-confidence functional substitutes found for this category.")
            print(f"     (Resolved group: {q_group})")

        print(f"\n[Aromatic / Flavor-Profile Matches]")
        print(f"  (Shared flavour molecules from FlavorDB — cross-category pairings)")
        if aromatic_subs:
            for i, r in enumerate(aromatic_subs):
                mol_pct = r['_mol_score'] * 100
                cat_tag = f" [{r['label']}]" if r['label'] else ""
                print(f"  {i+1:2}. {r['name']:<30}  Molecule overlap: {mol_pct:.1f}%{cat_tag}")
        else:
            print("  ⚠  No meaningful aromatic matches found (this ingredient may have few recorded molecules).")

        if show_complements and complements:
            print(f"\n[Common Pairings / Complements]")
            print(f"  (Ingredients that frequently appear together in recipes)")
            for i, r in enumerate(complements[:top_k]):
                cat_tag = f" [{r['label']}]" if r['label'] else ""
                print(f"  {i+1:2}. {r['name']:<30}  Co-occurrence score: {r['cos_sim']:.3f}{cat_tag}")

        print(f"\n{'─'*55}")
        print("  NOTE: Functional = same role in recipe (hot/cold swap).")
        print("  Aromatic = shares flavour molecules (creative pairings).")
        print(f"{'─'*55}\n")

        return {
            'functional': func_subs,
            'aromatic': aromatic_subs,
            'complements': complements[:top_k]
        }

    def check_pair(self, ing1, ing2):
        print(f"\n--- Checking Pair: {ing1} vs {ing2} ---")
        n1, n2 = None, None
        for n, d in self.G.nodes(data=True):
            nm = d.get('name', '')
            if not isinstance(nm, str):
                continue
            nm_lower = nm.lower()
            if nm_lower == ing1.lower():
                n1 = n
            if nm_lower == ing2.lower():
                n2 = n
        if not n1 or not n2:
            print("  One or both ingredients not found.")
            return

        idx1, idx2 = self.node2idx[n1], self.node2idx[n2]
        sim       = self._cooccur_similarity(idx1, idx2)
        mol_score = self._molecule_overlap_score(n1, n2)

        l1, g1 = self._get_sub_category(self.G.nodes[n1]['name'])
        l2, g2 = self._get_sub_category(self.G.nodes[n2]['name'])

        print(f"  Embedding Similarity:  {sim * 10:.2f}/10")
        print(f"  Molecule Overlap:      {mol_score * 100:.1f}%")
        print(f"  Categories:  {l1} ({g1})  vs  {l2} ({g2})")

        if self.G.has_edge(n1, n2):
            data = self.G.get_edge_data(n1, n2)
            print(f"  Direct Edge: Type={data.get('type')}, Weight={data.get('weight', 0):.4f}")
            print(f"  Status: ✓ COMPLEMENTS (frequently used together)")
        else:
            print("  No direct co-occurrence edge.")
            if g1 == g2 and sim > 0.7:
                print(f"  Status: ✓ PRACTICAL SUBSTITUTE (same category, high similarity)")
            elif mol_score > 0.05 and sim > 0.5:
                print(f"  Status: ~ FLAVOR SUBSTITUTE (shared molecules + moderate similarity)")
            elif mol_score > 0.05:
                print(f"  Status: ~ AROMATIC BRIDGE (shares flavour compounds)")
            else:
                print(f"  Status: ✗ UNRELATED")


if __name__ == "__main__":
    recommender = DisjointContextRecommender()

    if len(sys.argv) > 2:
        recommender.check_pair(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        recommender.get_substitutes(sys.argv[1], show_complements=True)
    else:
        for demo in ['apple', 'basil', 'milk', 'lemon']:
            recommender.get_substitutes(demo, show_complements=True)
