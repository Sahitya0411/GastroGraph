
import pandas as pd
import networkx as nx
import os
import sys

class GastroGraphBuilder:
    def __init__(self, input_dir='data'):
        self.input_dir = input_dir
        self.nodes_path = os.path.join(input_dir, 'nodes_191120.csv')
        self.edges_path = os.path.join(input_dir, 'edges_191120.csv')
        self.graph = nx.Graph()

    def load_data(self):
        if not os.path.exists(self.nodes_path) or not os.path.exists(self.edges_path):
            print(f"Error: Input files not found in {self.input_dir}")
            sys.exit(1)

        print("Loading nodes...")
        self.nodes_df = pd.read_csv(self.nodes_path)

        print("Loading edges...")
        self.edges_df = pd.read_csv(self.edges_path)

        print(f"Loaded {len(self.nodes_df)} nodes and {len(self.edges_df)} edges.")

    def filter_data(self):
        valid_types = ['ingredient', 'compound']
        self.nodes_df = self.nodes_df[self.nodes_df['node_type'].isin(valid_types)]

        valid_edge_types = ['ingr-ingr', 'ingr-dcomp', 'ingr-fcomp']
        self.edges_df = self.edges_df[self.edges_df['edge_type'].isin(valid_edge_types)]

        print(f"Filtered to {len(self.nodes_df)} nodes and {len(self.edges_df)} edges.")

    def build_graph(self):
        print("Building the graph...")

        for _, row in self.nodes_df.iterrows():
            self.graph.add_node(
                row['node_id'],
                name=row['name'],
                type='ingredient' if row['node_type'] == 'ingredient' else 'molecule',
                original_type=row['node_type'],
                is_hub=row['is_hub']
            )

        for _, row in self.edges_df.iterrows():
            source = row['id_1']
            target = row['id_2']
            edge_type = row['edge_type']
            weight = row['score']

            if not self.graph.has_node(source) or not self.graph.has_node(target):
                continue

            relationship = 'CO_OCCURS' if edge_type == 'ingr-ingr' else 'CONTAINS'

            if pd.isna(weight):
                weight = 1.0

            self.graph.add_edge(
                source,
                target,
                weight=weight,
                type=relationship,
                original_edge_type=edge_type
            )

        print(f"Graph constructed with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        self.apply_tfidf_weights()

    def apply_tfidf_weights(self):
        print("Calculating TF-IDF weights for Chemistry edges...")
        import math

        ingredients = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'ingredient']
        total_docs = len(ingredients)

        molecule_counts = {}

        for ing in ingredients:
            neighbors = self.graph.neighbors(ing)
            for neighbor in neighbors:
                if self.graph.nodes[neighbor]['type'] == 'molecule':
                    molecule_counts[neighbor] = molecule_counts.get(neighbor, 0) + 1

        print(f"Computed frequencies for {len(molecule_counts)} molecules.")

        count_updates = 0
        for u, v, data in self.graph.edges(data=True):
            if data['type'] == 'CONTAINS':
                node_u = self.graph.nodes[u]
                molecule = v if node_u['type'] == 'ingredient' else u

                doc_freq = molecule_counts.get(molecule, 0)
                if doc_freq > 0:
                    idf = math.log(total_docs / doc_freq)
                    data['weight'] = idf
                    count_updates += 1

        print(f"Updated weights for {count_updates} chemistry edges using TF-IDF.")

    def get_stats(self):
        ingredients = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'ingredient']
        molecules = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'molecule']

        print("\n--- Graph Statistics ---")
        print(f"Total Nodes: {self.graph.number_of_nodes()}")
        print(f"  - Ingredients: {len(ingredients)}")
        print(f"  - Molecules: {len(molecules)}")
        print(f"Total Edges: {self.graph.number_of_edges()}")

        co_occurs = sum(1 for _, _, d in self.graph.edges(data=True) if d['type'] == 'CO_OCCURS')
        contains = sum(1 for _, _, d in self.graph.edges(data=True) if d['type'] == 'CONTAINS')

        print(f"  - Context Edges (CO_OCCURS): {co_occurs}")
        print(f"  - Chemistry Edges (CONTAINS): {contains}")

    def save_graph(self, output_path='output/gastro_graph.gpickle'):
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    builder = GastroGraphBuilder(input_dir='data')
    builder.load_data()
    builder.filter_data()
    builder.build_graph()
    builder.get_stats()
    builder.save_graph(output_path='models/gastro_graph.gpickle')
