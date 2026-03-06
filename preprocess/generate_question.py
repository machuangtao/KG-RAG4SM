import pandas as pd
from tqdm import tqdm

def main():
    fnames = ['datasets/original/test_cms_q.xlsx', 'datasets/original/test_emed_q.xlsx', 'datasets/original/test_mimic_q.xlsx','datasets/original/test_synthea_q.xlsx', 'datasets/original/test_bank_q.xlsx', 'datasets/original/test_imsa_q.xlsx']

    for fname in fnames:
        reader = pd.read_excel(fname)
        questions = []
        for i in tqdm(range(reader.shape[0]), desc=f"Processing {fname}"):
            column_name1 = reader.iloc[i, 0]
            text_raw1 = reader.iloc[i, 2]
            column_name2 = reader.iloc[i, 1]
            text_raw2 = reader.iloc[i, 3]
            question = f"Attribute 1 {column_name1} and its description 1 {text_raw1} \nAttribute 2 {column_name2} and its description 2 {text_raw2} \nDo attribute 1 and attribute 2 are semantically matched with each other?" 
            questions.append(question)
        reader['question'] = questions
        reader.to_excel(fname, index=False)
        print('Generating questions...')



if __name__ == '__main__':
    main()
