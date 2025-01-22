import json
import pandas as pd
import re
import os

# Debug info
print("Current working directory:", os.getcwd())
print("Listing current directory contents:", os.listdir('.'))

# File paths relative to current directory (/app/layers)
json_file = "emed_wikidata_full_path.json"
excel_file = "../datafinal/test_emed_q.xlsx"
output_excel_file = "../datafinal/test_emed_q_with_paths.xlsx"

# Check if files exist
if not os.path.exists(json_file):
    raise FileNotFoundError(f"JSON file not found at {json_file}")
if not os.path.exists(excel_file):
    raise FileNotFoundError(f"Excel file not found at {excel_file}")

# Step 1: Read the JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

results = data.get("results", [])
if not results:
    print("No results found in the JSON file.")

# Step 2: Sort results by question_id
def extract_question_number(q_id):
    # q_id is in the form "question_XXXX"
    match = re.search(r'question_(\d+)', q_id)
    if match:
        return int(match.group(1))
    return float('inf')  # fallback if not found

results_sorted = sorted(results, key=lambda x: extract_question_number(x.get("question_id", "")))
results_dict = {r["question_id"]: r for r in results_sorted if "question_id" in r}

# Step 3: Read Excel file
df = pd.read_excel(excel_file)
if 'question_id' not in df.columns:
    raise ValueError("The 'question_id' column is not found in the input Excel file.")

print(f"Number of rows in Excel: {len(df)}")

def extract_relations_from_path(path):
    # Extract relations appearing between '--> [relation] -->'
    pattern = r'-->\s*\[(.*?)\]\s*-->'
    return re.findall(pattern, path)

def score_paths(paths):
    # Score paths based on frequency of relations
    # 1. Extract all relations from each path
    # 2. Count frequency of each relation in all paths
    # 3. Score each path: (sum of frequencies of its relations) / segments
    from collections import Counter
    all_relations = []
    paths_relations = []
    for p in paths:
        rels = extract_relations_from_path(p)
        paths_relations.append(rels)
        all_relations.extend(rels)
    
    freq = Counter(all_relations)
    scored_paths = []
    for p, rels in zip(paths, paths_relations):
        segments = len(p.split('-->'))
        if segments == 0:
            score = 0
        else:
            rel_score_sum = sum(freq[r] for r in rels)
            score = rel_score_sum / segments
        scored_paths.append((p, score))
    
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    return scored_paths

paths_all_list = []
top1_path_list = []
top2_path_list = []

for idx, row in df.iterrows():
    q_id = row['question_id']
    entry = results_dict.get(q_id, None)
    if entry and "paths" in entry:
        paths = entry["paths"]
        paths_all = "|".join(paths)
        num_paths = len(paths)

        if num_paths == 0:
            # No paths
            top1_path = ""
            top2_path = ""
        elif num_paths == 1:
            # One path: use the same for top1 and top2
            top1_path = paths[0]
            top2_path = paths[0]
        else:
            # Two or more paths, do ranking
            scored_paths = score_paths(paths)
            # top1_path: best ranked
            top1_path = scored_paths[0][0] if scored_paths else ""
            # top2_path: top two ranked paths separated by '|'
            # If there's only two paths, this will place both in top2_path
            # If more than two, still just take top two.
            top_two = [sp[0] for sp in scored_paths[:2]]  # top 2 paths
            top2_path = "|".join(top_two)

        paths_all_list.append(paths_all)
        top1_path_list.append(top1_path)
        top2_path_list.append(top2_path)
    else:
        # No matching entry found
        paths_all_list.append("")
        top1_path_list.append("")
        top2_path_list.append("")

df['paths_all'] = paths_all_list
df['top1_path'] = top1_path_list
df['top2_path'] = top2_path_list

df.to_excel(output_excel_file, index=False)
print(f"File saved to {output_excel_file}")
