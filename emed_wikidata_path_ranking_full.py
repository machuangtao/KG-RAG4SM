import json
import pandas as pd
from tqdm import tqdm
import logging
import os

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathRankingToExcel:
    def __init__(self, input_json_path, input_excel_path, output_excel_prefix, top_k_relationships):
        self.input_json_path = input_json_path
        self.input_excel_path = input_excel_path
        self.output_excel_prefix = output_excel_prefix
        self.top_k_relationships = top_k_relationships
        self.ranked_results = None

    def load_json_data(self):
        """Load the input JSON file containing paths."""
        logger.info(f"Loading JSON data from {self.input_json_path}...")
        try:
            with open(self.input_json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            logger.info(f"Successfully loaded {len(self.json_data['results'])} questions.")
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise

    def load_excel_data(self):
        """Load the input Excel file containing the questions."""
        logger.info(f"Loading Excel data from {self.input_excel_path}...")
        try:
            self.excel_data = pd.read_excel(self.input_excel_path)
            logger.info(f"Successfully loaded Excel data with {self.excel_data.shape[0]} rows.")
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise

    def rank_and_add_paths(self):
        """Add ranked paths from JSON to Excel data."""
        logger.info("Ranking paths and adding them to the Excel data...")
        paths_dict = {q['question_id']: q['paths'] for q in self.json_data['results']}
        
        # Create new columns in the Excel data for ranked paths
        self.excel_data['full_paths'] = ""
        self.excel_data['top2_paths'] = ""
        self.excel_data['top1_paths'] = ""

        for idx, row in tqdm(self.excel_data.iterrows(), total=self.excel_data.shape[0], desc="Processing rows"):
            question_id = row.get('question')
            paths = paths_dict.get(question_id, [])

            # Prepare ranked paths
            ranked_paths = self.rank_paths(paths)
            full_paths = " | ".join([p['path_text'] for p in ranked_paths])
            top2_paths = " | ".join([p['path_text'] for p in ranked_paths[:2]])
            top1_paths = ranked_paths[0]['path_text'] if ranked_paths else ""

            # Add the paths to the Excel dataframe
            self.excel_data.at[idx, 'full_paths'] = full_paths
            self.excel_data.at[idx, 'top2_paths'] = top2_paths
            self.excel_data.at[idx, 'top1_paths'] = top1_paths

    def rank_paths(self, paths):
        """Rank paths based on relationship overlap with the top_k_relationships."""
        ranked_paths = []
        for path_text in paths:
            predicates = self.extract_predicates(path_text)
            overlap_count = len(set(predicates) & set(self.top_k_relationships))
            normalized_score = overlap_count / len(predicates) if predicates else 0
            ranked_paths.append({
                'path_text': path_text,
                'score': normalized_score
            })
        # Sort paths by score in descending order
        ranked_paths.sort(key=lambda x: x['score'], reverse=True)
        return ranked_paths

    @staticmethod
    def extract_predicates(path_text):
        """Extract predicates (relationships) from the path text."""
        predicates = []
        try:
            parts = path_text.split(" --> ")
            for part in parts:
                if "[" in part and "]" in part:
                    predicate = part.split("[")[1].split("]")[0]
                    predicates.append(predicate)
        except Exception as e:
            logger.warning(f"Error extracting predicates from path: {path_text}, error: {e}")
        return predicates

    def save_to_excel(self):
        """Save the updated Excel data to files."""
        output_full = f"{self.output_excel_prefix}_full_paths.xlsx"
        output_top2 = f"{self.output_excel_prefix}_top2_paths.xlsx"
        output_top1 = f"{self.output_excel_prefix}_top1_paths.xlsx"

        # Save full paths
        self.excel_data.to_excel(output_full, index=False)
        logger.info(f"Saved full paths to {output_full}")

        # Save top-2 paths
        self.excel_data[['question', 'top2_paths']].to_excel(output_top2, index=False)
        logger.info(f"Saved top-2 paths to {output_top2}")

        # Save top-1 paths
        self.excel_data[['question', 'top1_paths']].to_excel(output_top1, index=False)
        logger.info(f"Saved top-1 paths to {output_top1}")


def main():
    # File paths (absolute paths)
    input_json_path = "emed_wikidata_full_path.json"  # JSON file with paths
    input_excel_path = "/app/datafinal/test_emed_q.xlsx"  # Excel file with questions
    output_excel_prefix = "/app/datafinal/test_emed_q"  # Prefix for output Excel files
    top_k_relationships = ['risk factor', 'handled, mitigated, or managed by', 'subclass of', 'founded by']

    try:
        path_ranking = PathRankingToExcel(
            input_json_path=input_json_path,
            input_excel_path=input_excel_path,
            output_excel_prefix=output_excel_prefix,
            top_k_relationships=top_k_relationships
        )
        path_ranking.load_json_data()
        path_ranking.load_excel_data()
        path_ranking.rank_and_add_paths()
        path_ranking.save_to_excel()
        logger.info("Process completed successfully.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
