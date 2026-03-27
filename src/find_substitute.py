import sys
import os
from disjoint_context import DisjointContextRecommender

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')
    data_dir   = os.path.join(project_root, 'data')
    config_dir = os.path.join(project_root, 'config')

    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        return

    try:
        recommender = DisjointContextRecommender(
            output_dir=models_dir, input_dir=data_dir, config_dir=config_dir)
    except FileNotFoundError as e:
        print(f"Error loading resources: {e}")
        print("Please ensure you have run the training process first.")
        return

    print("\n" + "="*50)
    print("Welcome to the GastroGraph Substitute Finder!")
    print("="*50)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        recommender.get_substitutes(query, top_k=5, show_complements=True)
    else:
        while True:
            query = input("\nEnter an ingredient name (or 'q' to quit): ").strip()
            if query.lower() == 'q':
                break
            if not query:
                continue
            recommender.get_substitutes(query, top_k=5, show_complements=True)

if __name__ == "__main__":
    main()
