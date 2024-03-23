from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_dir = "/shared/4/models/llama2/pytorch-versions/llama-2-7b-chat/"
data_dir = "./annotation/email_understanding/annotation_output/full/df4model.csv"
cache_dir= "/shared/4/models/"

llama_template = '''[INST]
{user_prompt}
[/INST]Answer:'''

user_prompt = '''You will be given an Email text. Please classify the intent of it into the following categories: Request, Propose, Commit/Agree, Deliver/Informative, Amend, Refuse, Introduction, Remind, Thank you/Welcome. Respond with only the category names without explanation.
Email: {email_text}'''

llm = LLM(model=model_dir)  # Create an LLM.
data_df = pd.read_csv(data_dir)
email_text_list = data_df["cleaned_text"]

answer_prompts = []
for email_text in email_text_list:
    answer_prompts.append(llama_template.format(user_prompt=user_prompt.format(email_text=email_text)))

# Output processing
output_text = []
prediction_df_list = {"pred_Act:::Request":[], 
                      "pred_Act:::Propose":[], 
                      "pred_Act:::Commit/Agree":[],
                      "pred_Act:::Deliver/Informative":[],
                      "pred_Act:::Amend":[],
                      "pred_Act:::Refuse":[],
                      "pred_Act:::Introduction":[],
                      "pred_Act:::Remind":[],
                      "pred_Act:::Thank you/Welcome":[]
                      }

outputs = llm.generate(answer_prompts, sampling_params=SamplingParams(max_tokens=32))
for i, output in enumerate(outputs):
    text = output.outputs[0].text.replace("\n", "").strip().lower()
    output_text.append(text)
    if "request" in text:
        prediction_df_list["pred_Act:::Request"].append(1)
    
    if "propose" in text:
        prediction_df_list["pred_Act:::Propose"].append(1)

    if "commit" in text or "agree" in text:
        prediction_df_list["pred_Act:::Commit/Agree"].append(1)
    
    if "deliver" in text or "informative" in text:
        prediction_df_list["pred_Act:::Deliver/Informative"].append(1)

    if "amend" in text:
        prediction_df_list["pred_Act:::Amend"].append(1)
    
    if "refuse" in text:
        prediction_df_list["pred_Act:::Refuse"].append(1)

    if "introduction" in text:
        prediction_df_list["pred_Act:::Introduction"].append(1)
    
    if "remind" in text:
        prediction_df_list["pred_Act:::Remind"].append(1)
    
    if "thank you" in text or "welcome" in text:
        prediction_df_list["pred_Act:::Thank you/Welcome"].append(1)

    for key in prediction_df_list:
        if len(prediction_df_list[key]) - 1 < i:
            prediction_df_list[key].append(0)

data_df = data_df.assign(**prediction_df_list)

metrics = {}
for key in prediction_df_list:
    intent_name = key.split(":::")[-1]
    metrics[intent_name] = {}
    y_test_local = data_df["label_Act:::"+intent_name]
    y_pred_local = data_df["pred_Act:::"+intent_name]
    metrics[intent_name]['precision'] = precision_score(y_test_local, y_pred_local)
    metrics[intent_name]['recall'] = recall_score(y_test_local, y_pred_local)
    metrics[intent_name]['f1'] = f1_score(y_test_local, y_pred_local)

print(pd.DataFrame.from_dict(metrics))
data_df.to_csv("llama_zeroshot_altogether.csv", index=False)