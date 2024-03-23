import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Script to run predictions with a specified GPU and data file.")
    parser.add_argument('--gpu', required=True, help="GPU number to use")
    parser.add_argument('--data_path', required=True, help="Path to the input data file")
    return parser.parse_args()


args = parse_args()

# Set CUDA GPU before importing torch and transformers
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Now it's safe to import heavy libraries
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = '/shared/4/projects/research-jam-2024/models/fine-tuned/roberta-base/checkpoint-864'
TOKENIZER = 'roberta-base'


def main(data_fp):
    out_dir = '/shared/4/projects/research-jam-2024/working-dir/model-predictions/'

    data_file_name = os.path.basename(data_fp).split('/')[-1]  # Name of the month e.g. en.2009-11.tsv
    output_csv_fp = os.path.join(out_dir, f'{data_file_name}')

    df = pd.read_csv(data_fp, sep='\t')[['message_id', 'message_body_clean']]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to('cuda')
    model.eval()

    labels = [
        'label_Act:::Thank you/Welcome', 'label_Act:::Commit/Agree', 'label_Act:::Request',
        'label_Act:::Deliver/Informative', 'label_Act:::Other_merged',
        'label_spam', 'label_Sender Expectation'
    ]

    predictions_list = []

    for index, text in enumerate(tqdm(df['message_body_clean'], total=len(df))):
        pred_dict = run_predictions(tokenizer, model, [text], labels)[0]
        predictions_list.append(pred_dict)

        if (index + 1) % 25000 == 0:
            save_intermediate_results(predictions_list, df, index, output_csv_fp)

    save_final_results(predictions_list, df, output_csv_fp)


def run_predictions(tokenizer, model, texts, labels):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)
    predictions = predictions.cpu().numpy()

    predictions_dicts = []
    for pred in predictions:
        pred_dict = {label: float(pred_val) for label, pred_val in zip(labels, pred)}
        predictions_dicts.append(pred_dict)

    return predictions_dicts


def save_intermediate_results(predictions_list, df, index, output_csv_fp):
    temp_df = pd.DataFrame(predictions_list)
    temp_final_df = pd.concat([df.iloc[:index + 1]['message_id'], temp_df], axis=1)
    temp_final_df.to_csv(output_csv_fp, sep='\t', index=False)
    print(f"Saved {index + 1} rows to {output_csv_fp}")


def save_final_results(predictions_list, df, output_csv_fp):
    predictions_df = pd.DataFrame(predictions_list)
    final_df = pd.concat([df['message_id'], predictions_df], axis=1)
    final_df.to_csv(output_csv_fp, sep='\t', index=False)
    print(f"Final predictions saved to {output_csv_fp}")


if __name__ == '__main__':
    main(args.data_path)
