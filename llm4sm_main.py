import pandas as pd
import sys
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

from src.utils import get_devices, setup_llm_logging, extract_label, calculate_metrics
from src.llm import initialize_llm_model
from src.llm4sm import LLM_for_Schema_Matching

def get_data_file(dataset: str) -> str:
    dataset_mapping = {
        "cms": "datasets/reproduce/test_cms_q_with_paths.xlsx",
        "mimic": "datasets/reproduce/test_mimic_q_with_paths.xlsx",
        "synthea": "datasets/reproduce/test_synthea_q_with_paths.xlsx",
        "emed": "datasets/reproduce/test_emed_q_with_paths.xlsx"
    }
    return dataset_mapping.get(dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cms', help='dataset: mimic, mimic, synthea, emed')
    parser.add_argument('--backbone_llm_model', type=str, default='gpt-4o-mini',
                      help='Model name (e.g., gpt-4o-mini, jellyfish-8b, jellyfish-7b, mistral-7b)')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for storing logs')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    devices = get_devices(args.device)
    
    # Setup logging
    log_filename = setup_llm_logging(args.log_dir, args.dataset, args.backbone_llm_model)
    
    # Initialize llm model
    try:
        model = initialize_llm_model(args.backbone_llm_model, devices)
    except Exception as e:
        logging.error(f"Error initializing llm model: {e}")
        sys.exit(1)
    
    # Load data
    try:
        data_file = get_data_file(args.dataset)
        if not data_file:
            raise ValueError(f"Invalid dataset: {args.dataset}")
        reader = pd.read_excel(data_file)
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        sys.exit(1)
    
    
    llm4sm = LLM_for_Schema_Matching()
    y_true = []
    y_pred = []
    
    for i in tqdm(range(reader.shape[0]), desc="Processing questions"):
        try:
            question = reader.iloc[i, 9]
            ground_truth = reader.iloc[i, 4]
            
            _, _, response = llm4sm.llm_for_schema_matching(question, model)
            label = extract_label(response)
            
            if label != -1 and not pd.isna(ground_truth):
                y_true.append(int(ground_truth))
                y_pred.append(label)
            
            logging.info(f"\nResults for row {i+1}:")
            logging.info(f"Response: {response}")
            logging.info(f"Extracted Label: {label}")
            logging.info(f"Ground Truth: {ground_truth}")
            
        except Exception as e:
            logging.error(f"Error processing row {i+1}: {e}")
            continue
    
    # Calculate and log metrics
    precision, recall, f1, accuracy = calculate_metrics(y_true, y_pred)
    
    logging.info("\nFinal Metrics:")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("\nExecution Time Summary:")
    logging.info(f"Start Time: {start_time}")
    logging.info(f"End Time: {end_time}")
    logging.info(f"Total Duration: {duration}")

    print(f"Resutls are recorded in the following log file:{log_filename}")

if __name__ == "__main__":
    main()