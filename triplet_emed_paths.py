import os
import json
import pandas as pd

# Define the base directory and relevant subdirectories
base_dir = "/app"
layers_dir = os.path.join(base_dir, "layers")
datafinal_dir = os.path.join(base_dir, "datafinal")

# Define the paths to input and output files
input_xlsx = os.path.join(datafinal_dir, "test_emed_q.xlsx")
output_xlsx = os.path.join(datafinal_dir, "test_emed_q_method2.xlsx")
json_file = os.path.join(layers_dir, "emed_top10_similar2.json")

# Check if the JSON and Excel files exist
if not os.path.exists(json_file):
    raise FileNotFoundError(f"JSON file not found: {json_file}")

if not os.path.exists(input_xlsx):
    raise FileNotFoundError(f"Excel file not found: {input_xlsx}")

# Load the JSON data
with open(json_file, "r") as f:
    results_data = json.load(f)

def extract_path(result):
    """
    Extract and format the path with QIDs/PIDs and English labels.
    Format: head_entity (head_id) --> [relation](relation_id) --> tail_entity (tail_id)
    """
    doc_id = result["id"]
    meta = result["metadata"]
    
    # Extract fields based on triplet type
    if doc_id.startswith("triplet2_") or doc_id.startswith("triplet3_"):
        head_id = meta.get("head_id", "")
        head_entity = meta.get("head_entity", "")
        relation_id = meta.get("relation_id", "")
        relation = meta.get("relation", "")
        tail_id = meta.get("tail_id", "")
        tail_entity = meta.get("tail_entity", "")
        
        # Check if all necessary fields are present
        if head_id and head_entity and relation_id and relation and tail_id and tail_entity:
            return f"{head_entity} ({head_id}) --> [{relation}]({relation_id}) --> {tail_entity} ({tail_id})"
    
    # If fields are missing, return an empty string or a placeholder
    return ""

# Create a dictionary to map question_index to concatenated paths
question_paths = {}
for q in results_data:
    q_idx = q["question_index"]
    res = q["results"]
    # Extract formatted paths for all results
    paths = [extract_path(r) for r in res]
    # Filter out any empty paths
    paths = [path for path in paths if path]
    # Join all paths by '|'
    joined_paths = "|".join(paths)
    question_paths[q_idx] = joined_paths

# Load the Excel data into a DataFrame
df = pd.read_excel(input_xlsx)

def get_question_index(qid):
    """
    Extract the numerical question index from the 'question_id' column.
    Assumes 'question_id' is in the format 'question_<index>'.
    """
    if isinstance(qid, str) and qid.startswith("question_"):
        try:
            return int(qid.split("_")[-1])
        except ValueError:
            return None
    return None

# Apply the function to create a 'question_index' column
df["question_index"] = df["question_id"].apply(get_question_index)

# Initialize lists to store the new columns
all_paths_list = []
top2_paths_list = []
top1_path_list = []

# Iterate over each row in the DataFrame
for idx, row in df.iterrows():
    q_idx = row["question_index"]
    if q_idx in question_paths:
        paths_str = question_paths[q_idx]
        # Split the paths by '|'
        paths = paths_str.split("|")
        # Assign all paths
        all_paths = paths_str
        # Assign top 2 paths
        top2 = "|".join(paths[:2]) if len(paths) >= 2 else paths_str
        # Assign top 1 path
        top1 = paths[0] if len(paths) >= 1 else ""
    else:
        all_paths = ""
        top2 = ""
        top1 = ""
    
    # Append the paths to the respective lists
    all_paths_list.append(all_paths)
    top2_paths_list.append(top2)
    top1_path_list.append(top1)

# Add the new columns to the DataFrame
df["all paths"] = all_paths_list
df["top 2 paths"] = top2_paths_list
df["top1 path"] = top1_path_list

# Save the updated DataFrame to a new Excel file
df.to_excel(output_xlsx, index=False)
print(f"Successfully created {output_xlsx} with new columns.")
