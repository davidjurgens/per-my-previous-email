from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
data_dir = "/shared/4/projects/research-jam-2024/data/final_test_data.tsv"
cache_dir = "/shared/4/models/"

llama_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

For the following email:

Subject: {subject}
Body: {body}

Please classify the email's intent into the following catagories:
1. Sharing: The sender is sharing information, opinions, updates, or other content with the recipient(s). This includes (but is not limited to) answering a question posed by the recipient; sharing background information to ask a question; stating opinions or takes; sharing documents/external links/resources/references/sample code; or proactively sharing FYI messages and updates.
2. Requesting: The sender is asking the recipient(s) for something. This includes (but is not limited to) asking a question, asking for more information, or requesting a task or action be completed. Only EXPLICIT requests should be annotated.
3. Promising: The sender writes that they will perform some future action. This includes (but is not limited to) saying they will look into a problem, rewrite some code, fix a bug, give an updated in the future, or do some joint activity with the recipient.
4. Personal Communication: The sender makes some interpersonal investment in the recipient(s), beyond the subject of the email. Includes indicators that they have a relationship outside the email thread, personal self-disclosures, engagement with the recipient's personal feelings/worldviews, or other communication that goes beyond the subject of the email and addresses the recipient as a person. Does not include pleasantries like please, thank you, expressions of gratitude, etc.
5. Pleasantries or Acknowledgement Only: The email contains only pleasantries, formalities, or acknowledgement of the prior email (e.g., the whole email is a 'Thank you very much' or a 'Will try that!' or 'Got it!') -- or if pleasantries/formalities/acknowledgements like these are the main point of the message and all other content is unimportant or secondary. This is mutually exclusive from the rest of options.
6. Auto-generated or Spam: Email digests, auto-reminders, github or auto-change notification messages, do not reply emails, scam emails, etc. This is mutually exclusive from the rest of options.
Respond with all the applicable categories without explanation.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer:'''

llm = LLM(model=model_dir, download_dir=cache_dir)  # Create an LLM.
data_df = pd.read_csv(data_dir, sep='\t')
#data_df = data_df[data_df["split"] == "test"]
subject_list = data_df["subject"]
body_list = data_df["message_body_clean"]

answer_prompts = []
for idx in range(len(subject_list)):
    answer_prompts.append(llama_template.format(subject=subject_list[idx], body=body_list[idx]))

# Output processing
output_text = []
prediction_df_list = {
    "Sharing_pred": [],
    "Requesting_pred": [],
    "Promising_pred": [],
    "Personal_pred": [],
    "Pleasantries_pred": [],
    "Spam_pred": [],
}

outputs = llm.generate(answer_prompts, sampling_params=SamplingParams(max_tokens=64))
for i, output in enumerate(outputs):
    text = output.outputs[0].text.replace("\n", "").strip().lower()
    print(text)
    output_text.append(text)
    if "sharing" in text or "1" in text:
        prediction_df_list["Sharing_pred"].append(1)

    if "requesting" in text or "2" in text:
        prediction_df_list["Requesting_pred"].append(1)

    if "promising" in text or "3" in text:
        prediction_df_list["Promising_pred"].append(1)
    
    if "personal" in text or "4" in text:
        prediction_df_list["Personal_pred"].append(1)

    if "pleasantries" in text or "acknowledgement" in text or "5" in text:
        prediction_df_list["Pleasantries_pred"].append(1)
    
    if "spam" in text or "auto-generated" in text or "6" in text:
        prediction_df_list["Spam_pred"].append(1)

    for key in prediction_df_list:
        if len(prediction_df_list[key]) - 1 < i:
            prediction_df_list[key].append(0)


########### START OF reply ###########
llama_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

For the following email:

Subject: {subject}
Body: {body}

{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer:'''

user_prompt_dict = {
    "ResponseExpected": "Does email sender explicitly expects an email in reply? E.g., to answer a question, engage with content/attachments and send a follow up email, take some action outside the email chain and send a confirmation that the action was taken, etc. Only annotate if the sender explicitly expects a response. Please respond with only yes or no.",
}

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

########### END OF spam & reply ###########


data_df = data_df.assign(**prediction_df_list)

metric_true_df_list = {}
metric_pred_df_list = {}

for key in prediction_df_list:
    intent_name = key.split("_")[0]
    metric_true_df_list[intent_name] = data_df[intent_name]
    metric_pred_df_list[intent_name] = data_df[intent_name+"_pred"]

print("Llama-3 Zero-shot Multi-Choice Results")
print(
    classification_report(
        pd.DataFrame.from_dict(metric_true_df_list),
        pd.DataFrame.from_dict(metric_pred_df_list),
        target_names=metric_true_df_list.keys(),
        zero_division=0.0
    )
)

data_df.to_csv("llama_zeroshot_multi.csv", index=False)
