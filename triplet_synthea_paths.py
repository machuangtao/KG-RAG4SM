import os
import json
import pandas as pd

# Define directories and file paths (adjust as needed)
base_dir = "/app"
layers_dir = os.path.join(base_dir, "layers")
datafinal_dir = os.path.join(base_dir, "datafinal")

# Define the paths to input and output files
input_xlsx = os.path.join(datafinal_dir, "test_synthea_q.xlsx")  # Ensure this file exists and has a 'question_id' column
output_xlsx = os.path.join(datafinal_dir, "test_synthea_q_method2.xlsx")
json_file = os.path.join(layers_dir, "synthea_top10_similar2.json")  # Your Synthea JSON results

# Check file existence
if not os.path.exists(json_file):
    raise FileNotFoundError(f"JSON file not found: {json_file}")

if not os.path.exists(input_xlsx):
    raise FileNotFoundError(f"Excel file not found: {input_xlsx}")

# Load the JSON data
with open(json_file, "r", encoding="utf-8") as f:
    results_data = json.load(f)

def extract_path(result):
    """
    Extract and format the path with QIDs/PIDs and English labels.
    Format: head_entity (head_id) --> [relation](relation_id) --> tail_entity (tail_id)
    This function assumes that each 'result' has a 'metadata' field with 'head_id', 'head_entity',
    'relation_id', 'relation', 'tail_id', and 'tail_entity'.
    Adjust if your Synthea dataset has different fields or naming conventions.
    """
    doc_id = result["id"]
    meta = result["metadata"]
    
    # Extract fields
    head_id = meta.get("head_id", "")
    head_entity = meta.get("head_entity", "")
    relation_id = meta.get("relation_id", "")
    relation = meta.get("relation", "")
    tail_id = meta.get("tail_id", "")
    tail_entity = meta.get("tail_entity", "")
    
    # Ensure required fields are present
    if head_id and head_entity and relation_id and relation and tail_id and tail_entity:
        return f"{head_entity} ({head_id}) --> [{relation}]({relation_id}) --> {tail_entity} ({tail_id})"
    return ""

# Create a dictionary mapping question_index to concatenated paths
question_paths = {}
for q in results_data:
    q_idx = q["question_index"]
    res = q["results"]
    # Extract formatted paths
    paths = [extract_path(r) for r in res]
    # Filter out empty paths
    paths = [p for p in paths if p]
    # Join all paths by '|'
    joined_paths = "|".join(paths)
    question_paths[q_idx] = joined_paths

# Load Excel data
df = pd.read_excel(input_xlsx)

def get_question_index(qid):
    """
    Extract the numerical question index from the 'question_id' column.
    Assumes 'question_id' is in the format 'question_<index>'.
    Adjust as needed if your Synthea question IDs differ.
    """
    if isinstance(qid, str) and qid.startswith("question_"):
        try:
            return int(qid.split("_")[-1])
        except ValueError:
            return None
    return None

# Create a 'question_index' column in the DataFrame
df["question_index"] = df["question_id"].apply(get_question_index)

# Prepare new columns
all_paths_list = []
top2_paths_list = []
top1_path_list = []

# Iterate over rows and assign paths
for idx, row in df.iterrows():
    q_idx = row["question_index"]
    if q_idx in question_paths:
        paths_str = question_paths[q_idx]
        paths = paths_str.split("|")
        
        all_paths = paths_str
        top2 = "|".join(paths[:2]) if len(paths) >= 2 else paths_str
        top1 = paths[0] if paths else ""
    else:
        all_paths = ""
        top2 = ""
        top1 = ""
    
    all_paths_list.append(all_paths)
    top2_paths_list.append(top2)
    top1_path_list.append(top1)

# Add new columns to DataFrame
df["all paths"] = all_paths_list
df["top 2 paths"] = top2_paths_list
df["top1 path"] = top1_path_list

# Save the updated DataFrame
df.to_excel(output_xlsx, index=False)
print(f"Successfully created {output_xlsx} with new columns.")
