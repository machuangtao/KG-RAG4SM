import json
import os
from tqdm import tqdm
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathRanking:
    def __init__(self, input_file, top_k_relationships, output_file_prefix):
        self.input_file = input_file
        self.top_k_relationships = top_k_relationships
        self.output_file_prefix = output_file_prefix

    def load_data(self):
        """Load the input JSON file containing paths."""
        logger.info(f"Loading data from {self.input_file}...")
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Successfully loaded data.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def rank_paths(self):
        """Rank paths based on normalized relationship-based scoring."""
        self.ranked_results = []
        for question in tqdm(self.data['results'], desc="Processing questions"):
            question_id = question['question_id']
            question_text = question['question_text']
            paths = question['paths']

            scored_paths = []
            for path_set in paths:
                entity_pair = path_set['entity_pair']
                path_data = path_set['paths']

                for path in path_data:
                    # Check if the path predicates overlap with the top_k_relationships
                    predicate_overlap = set(path['predicates']) & set(self.top_k_relationships)
                    overlap_count = len(predicate_overlap)
                    path_length = len(path['predicates'])  # Number of predicates in the path

                    # Normalize the score by path length
                    normalized_score = overlap_count / path_length if path_length > 0 else 0

                    # Add the score and path to the scored paths
                    scored_paths.append({
                        'score': normalized_score,
                        'entity_pair': entity_pair,
                        'path': path
                    })

            # Sort all paths for the question by normalized score in descending order
            scored_paths.sort(key=lambda x: x['score'], reverse=True)
            self.ranked_results.append({
                'question_id': question_id,
                'question_text': question_text,
                'scored_paths': scored_paths
            })

    def save_results(self):
        """Save ranked paths in text files."""
        # File paths
        full_path_file = f"{self.output_file_prefix}_ranking10.txt"
        top2_path_file = f"{self.output_file_prefix}_ranking_top2.txt"
        top1_path_file = f"{self.output_file_prefix}_ranking_top1.txt"

        # Save all ranked paths
        with open(full_path_file, 'w', encoding='utf-8') as f:
            for question in self.ranked_results:
                f.write(f"Question ID: {question['question_id']}\n")
                f.write(f"Question Text: {question['question_text']}\n")
                for scored_path in question['scored_paths']:
                    entity_pair = scored_path['entity_pair']
                    f.write(f"Entity Pair: {entity_pair}\n")
                    f.write(f"  Path: {scored_path['path']['path_text']}\n")
                f.write("\n")
        logger.info(f"Saved full path rankings to {full_path_file}")

        # Save top-2 paths
        with open(top2_path_file, 'w', encoding='utf-8') as f:
            for question in self.ranked_results:
                f.write(f"Question ID: {question['question_id']}\n")
                f.write(f"Question Text: {question['question_text']}\n")
                for scored_path in question['scored_paths'][:2]:  # Top-2 paths
                    entity_pair = scored_path['entity_pair']
                    f.write(f"Entity Pair: {entity_pair}\n")
                    f.write(f"  Path: {scored_path['path']['path_text']}\n")
                f.write("\n")
        logger.info(f"Saved top-2 path rankings to {top2_path_file}")

        # Save top-1 paths
        with open(top1_path_file, 'w', encoding='utf-8') as f:
            for question in self.ranked_results:
                f.write(f"Question ID: {question['question_id']}\n")
                f.write(f"Question Text: {question['question_text']}\n")
                if question['scored_paths']:  # Ensure at least one path exists
                    scored_path = question['scored_paths'][0]  # Top-1 path
                    entity_pair = scored_path['entity_pair']
                    f.write(f"Entity Pair: {entity_pair}\n")
                    f.write(f"  Path: {scored_path['path']['path_text']}\n")
                f.write("\n")
        logger.info(f"Saved top-1 path rankings to {top1_path_file}")


def main():
    # Configuration
    input_file = 'cms_wikidata_path.json'
    output_file_prefix = 'cms_wikidata_path'
    top_k_relationships = ['P780', 'P828', 'P1535', 'P425']  # Example relationships

    try:
        ranking = PathRanking(input_file, top_k_relationships, output_file_prefix)
        ranking.load_data()
        ranking.rank_paths()
        ranking.save_results()
        logger.info("Ranking process completed successfully.")
    except Exception as e:
        logger.error(f"Fatal error during ranking process: {e}")


if __name__ == "__main__":
    main()
