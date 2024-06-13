import argparse
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LABELS = ['Sharing', 'Requesting', 'Promising', 'Personal', 'Pleasantries', 'Spam', 'ResponseExpected']


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run model predictions on text data.")
    parser.add_argument('--model_path', type=str,
                        default="/shared/4/projects/research-jam-2024/models/fine-tuned/roberta-base/checkpoint-181")
    parser.add_argument('--data_fp', type=str, required=True, help='File path for the data to be processed.')
    parser.add_argument('--out_dir', type=str,
                        default="/shared/4/projects/research-jam-2024/working-dir/model-predictions/")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing the data.')
    return parser.parse_args()


def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    df = df[['message_id', 'message_body_clean']].rename(columns={'message_id': 'id', 'message_body_clean': 'text'})
    return df


def setup_model(model_path):
    logging.info("Setting up the model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=7,
                                                               problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = model.to('cuda')
    return model, tokenizer


def run_predictions(model, tokenizer, texts, labels):
    logging.info(f"Running predictions on {len(texts)} texts")
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=350, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)
    predictions = predictions.cpu().numpy()

    predictions_dicts = [{label: float(pred_val) for label, pred_val in zip(labels, pred)} for pred in predictions]
    return predictions_dicts


def main():
    args = parse_arguments()
    df = load_data(args.data_fp)
    model, tokenizer = setup_model(args.model_path)
    predictions_list = []

    model.eval()

    logging.info(f"Starting prediction batches of size {args.batch_size}")

    file_name = os.path.basename(args.data_fp)
    output_csv_fp = os.path.join(args.out_dir, file_name)

    for i in tqdm(range(0, len(df), args.batch_size), desc="Processing text batches"):
        batch_texts = df['text'][i:i + args.batch_size].tolist()
        batch_predictions = run_predictions(model, tokenizer, batch_texts, LABELS)
        predictions_list.extend(batch_predictions)

        # Save complete results every 100 batches
        if (i // args.batch_size) % 100 == 99:
            complete_df = pd.DataFrame(predictions_list)
            complete_df = pd.concat([df['id'].iloc[:len(complete_df)], complete_df], axis=1)
            complete_csv_fp = output_csv_fp.replace('.csv', f'_up_to_batch_{i // args.batch_size + 1}.csv')
            complete_df.to_csv(complete_csv_fp, index=False)
            logging.info(f"Saved complete results up to batch {i // args.batch_size + 1} to {complete_csv_fp}")

    # Save the final results
    final_df = pd.DataFrame(predictions_list)
    final_df = pd.concat([df['id'], final_df], axis=1)
    final_df.to_csv(output_csv_fp, index=False)
    logging.info(f"Final output saved to {output_csv_fp}")


if __name__ == "__main__":
    main()
