import json
import os
import pandas as pd
from tqdm import tqdm
import logging
import re

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathRanking:
    def __init__(self, input_file, excel_input_file, top_k_relationships, output_file_prefix):
        self.input_file = input_file
        self.excel_input_file = excel_input_file
        self.top_k_relationships = top_k_relationships
        self.output_file_prefix = output_file_prefix
        self.excel_data = None

    def load_data(self):
        """Load the input JSON file containing paths."""
        logger.info(f"Loading data from {self.input_file}...")
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Successfully loaded {len(self.data['results'])} questions.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def load_excel_data(self):
        """Load the Excel file containing questions."""
        logger.info(f"Loading Excel data from {self.excel_input_file}...")
        try:
            self.excel_data = pd.read_excel(self.excel_input_file)
            logger.info(f"Successfully loaded {len(self.excel_data)} rows from Excel.")
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise

    def parse_path(self, path_text):
        """Parse a textual path into a structured format."""
        pattern = r"(.*?) \((Q\d+)\) --> \[(.*?)\] --> (.*?) \((Q\d+)\)"
        matches = re.findall(pattern, path_text)
        predicates = []
        entities = []

        for match in matches:
            entity1_label, entity1_id, predicate, entity2_label, entity2_id = match
            entities.append((entity1_label.strip(), entity1_id.strip()))
            predicates.append(predicate.strip())

        return {
            "path_text": path_text,
            "entities": entities,
            "predicates": predicates
        }

    def rank_paths(self):
        """Rank paths based on normalized relationship-based scoring."""
        self.ranked_results = []
        for question in tqdm(self.data['results'], desc="Processing questions"):
            question_id = question['question_id']
            question_text = question['question_text']
            paths = question.get('paths', [])

            scored_paths = []
            for path_text in paths:
                # Parse the textual path into a structured format
                parsed_path = self.parse_path(path_text)

                # Check if the path predicates overlap with the top_k_relationships
                predicate_overlap = set(parsed_path['predicates']) & set(self.top_k_relationships)
                overlap_count = len(predicate_overlap)
                path_length = len(parsed_path['predicates'])  # Number of predicates in the path

                # Normalize the score by path length
                normalized_score = overlap_count / path_length if path_length > 0 else 0

                # Add the score and path to the scored paths
                scored_paths.append({
                    'score': normalized_score,
                    'path': parsed_path
                })

            # Sort all paths for the question by normalized score in descending order
            scored_paths.sort(key=lambda x: x['score'], reverse=True)
            self.ranked_results.append({
                'question_id': question_id,
                'question_text': question_text,
                'scored_paths': scored_paths
            })

    def save_to_excel(self):
        """Save ranked paths to Excel."""
        # Copy the Excel data
        excel_data_with_paths = self.excel_data.copy()
        paths_column = []

        # Map paths to questions in the Excel file
        for index, row in self.excel_data.iterrows():
            question_id = f"question_{index}"
            matching_result = next(
                (q for q in self.ranked_results if q['question_id'] == question_id),
                None
            )
            if matching_result:
                # All paths for this question
                full_paths = " | ".join(
                    [path['path']['path_text'] for path in matching_result['scored_paths']]
                )
                # Top-2 paths for this question
                top2_paths = " | ".join(
                    [path['path']['path_text'] for path in matching_result['scored_paths'][:2]]
                )
                # Top-1 path for this question
                top1_path = (
                    matching_result['scored_paths'][0]['path']['path_text']
                    if matching_result['scored_paths'] else ""
                )
            else:
                full_paths, top2_paths, top1_path = "", "", ""

            # Add paths to the column
            paths_column.append((full_paths, top2_paths, top1_path))

        # Split paths into separate columns
        excel_data_with_paths['Full Paths'] = [x[0] for x in paths_column]
        excel_data_with_paths['Top-2 Paths'] = [x[1] for x in paths_column]
        excel_data_with_paths['Top-1 Path'] = [x[2] for x in paths_column]

        # Save to new Excel files
        full_output_file = os.path.join(self.output_file_prefix + "_full_paths.xlsx")
        top2_output_file = os.path.join(self.output_file_prefix + "_top2_paths.xlsx")
        top1_output_file = os.path.join(self.output_file_prefix + "_top1_path.xlsx")

        excel_data_with_paths.to_excel(full_output_file, index=False)
        logger.info(f"Saved full paths to {full_output_file}")

        excel_data_with_paths[['omop', 'table', 'question', 'Top-2 Paths']].to_excel(
            top2_output_file, index=False
        )
        logger.info(f"Saved top-2 paths to {top2_output_file}")

        excel_data_with_paths[['omop', 'table', 'question', 'Top-1 Path']].to_excel(
            top1_output_file, index=False
        )
        logger.info(f"Saved top-1 paths to {top1_output_file}")


def main():
    # Configuration
    input_file = '/app/layers/cms_wikidata_full_path.json'  # Full dataset
    excel_input_file = '/app/datafinal/test_cms_q.xlsx'  # Absolute path to the Excel file
    output_file_prefix = '/app/datafinal/test_cms_q'
    top_k_relationships = ['handled, mitigated, or managed by', 'facet of', 'subclass of', 'instance of']  # Example relationships

    try:
        ranking = PathRanking(input_file, excel_input_file, top_k_relationships, output_file_prefix)
        ranking.load_data()
        ranking.load_excel_data()
        ranking.rank_paths()
        ranking.save_to_excel()
        logger.info("Ranking process completed successfully.")
    except Exception as e:
        logger.error(f"Fatal error during ranking process: {e}")


if __name__ == "__main__":
    main()
