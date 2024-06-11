from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
data_dir = "/shared/4/projects/research-jam-2024/data/final_test_data.tsv"
cache_dir = "/shared/4/models/"

llama_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

For the following email:

Subject: {subject}
Body: {body}

{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer:'''

user_prompt_dict = {
    "Sharing": '''Does the email's intent align with "Sharing"? "Sharing" means the sender is sharing information, opinions, updates, or other content with the recipient(s). This includes (but is not limited to) answering a question posed by the recipient; sharing background information to ask a question; stating opinions or takes; sharing documents/external links/resources/references/sample code; or proactively sharing FYI messages and updates. Respond with only yes or no.''',
    "Requesting": '''Does the email's intent align with "Requesting"? "Requesting" means the sender is asking the recipient(s) for something. This includes (but is not limited to) asking a question, asking for more information, or requesting a task or action be completed. Only EXPLICIT requests should be considered. Respond with only yes or no.''',
    "Promising": '''Does the email's intent align with "Promising"? "Promising" means the sender writes that they will perform some future action. This includes (but is not limited to) saying they will look into a problem, rewrite some code, fix a bug, give an updated in the future, or do some joint activity with the recipient. Respond with only yes or no.''',
    "Personal": '''Does the email's intent align with "Personal Communication"? "Personal Communication" means the sender makes some interpersonal investment in the recipient(s), beyond the subject of the email. Includes indicators that they have a relationship outside the email thread, personal self-disclosures, engagement with the recipient's personal feelings/worldviews, or other communication that goes beyond the subject of the email and addresses the recipient as a person. Does not include pleasantries like please, thank you, expressions of gratitude, etc. Respond with only yes or no.''',
    "Pleasantries": '''Does the email's intent align with "Pleasantries or Acknowledgement Only"? "Pleasantries or Acknowledgement Only" means the email contains only pleasantries, formalities, or acknowledgement of the prior email (e.g., the whole email is a 'Thank you very much' or a 'Will try that!' or 'Got it!') -- or if pleasantries/formalities/acknowledgements like these are the main point of the message and all other content is unimportant or secondary. Respond with only yes or no.''',
    "Spam": '''Does the email's intent align with "Auto-generated or Spam"? For example, email digests, auto-reminders, github or auto-change notification messages, do not reply emails, scam emails, etc. Respond with only yes or no.''',
    "ResponseExpected": "Does email sender explicitly expects an email in reply? E.g., to answer a question, engage with content/attachments and send a follow up email, take some action outside the email chain and send a confirmation that the action was taken, etc. Only annotate if the sender explicitly expects a response. Please respond with only yes or no.",
}

llm = LLM(model=model_dir, download_dir=cache_dir)  # Create an LLM.
data_df = pd.read_csv(data_dir, sep='\t')
#data_df = data_df[data_df["split"] == "test"]
subject_list = data_df["subject"]
body_list = data_df["message_body_clean"]

prediction_df_list = {}
errors = 0

for intent_name in tqdm(user_prompt_dict):
    prediction_df_list[intent_name+"_pred"] = []

    answer_prompts = []
    for idx in range(len(subject_list)):
        answer_prompts.append(
            llama_template.format(
                user_prompt=user_prompt_dict[intent_name], subject=subject_list[idx], body=body_list[idx]
            )
        )
    outputs = llm.generate(answer_prompts)

    for i, output in enumerate(outputs):
        text = output.outputs[0].text.replace("\n", "").strip().lower()
        if "yes" in text.lower():
            prediction_df_list[intent_name+"_pred"].append(1)
        else:
            prediction_df_list[intent_name+"_pred"].append(0)
            if "no" not in text.lower():
                errors += 1

data_df = data_df.assign(**prediction_df_list)

metric_true_df_list = {}
metric_pred_df_list = {}

for intent_name in user_prompt_dict:
    metric_true_df_list[intent_name] = data_df[intent_name]
    metric_pred_df_list[intent_name] = data_df[intent_name+"_pred"]

print("Llama-3 Zero-shot Single-Choice Results")
print(
    classification_report(
        pd.DataFrame.from_dict(metric_true_df_list),
        pd.DataFrame.from_dict(metric_pred_df_list),
        target_names=metric_true_df_list.keys(),
        zero_division=0.0
    )
)

print("Errors: " + str(errors))
data_df.to_csv("llama_zeroshot_single.csv", index=False)
