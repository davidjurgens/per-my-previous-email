from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_dir = "/shared/4/models/llama2/pytorch-versions/llama-2-7b-chat/"
data_dir = "./annotation/email_understanding/annotation_output/full/df4model.csv"
cache_dir= "/shared/4/models/"

llama_template = '''[INST]
{user_prompt}
Email: {email_text}
[/INST]Answer:'''

user_prompt_dict = {"Request": "You will be given an Email text. Answer yes if the sender is sending a request or order for the recipient to perform some activity, otherwise answer no. A question is considered a request for delivery of information. Please respond with only yes or no.", 
                    "Propose": "You will be given an Email text. Answer yes if the sender is proposing a joint activity, e.g., asks the recipient to perform some activity or suggests a joint meeting, otherwise answer no. Please respond with only yes or no.", 
                    "Commit/Agree": "You will be given an Email text. Answer yes if the email is committing to some future actions or confirming to comply with some previously described actions, e.g., confirming to do a job, otherwise answer no. Please respond with only yes or no.",
                    "Deliver/Informative": "You will be given an Email text. Answer yes if the email is delivering or providing information and opinion, otherwise answer no. Please respond with only yes or no.",
                    "Amend": "You will be given an Email text. Answer yes if the email is amending an earlier proposal, otherwise answer no. An amendment is a suggested modification of an already-proposed task. Please respond with only yes or no.",
                    "Refuse": "You will be given an Email text. Answer yes if the email rejects a meeting/action/task or declines an invitation/proposal, otherwise answer no. Please respond with only yes or no.",
                    "Introduction": "You will be given an Email text. Answer yes if the email is written by someone to introduce themselves, otherwise answer no. Please respond with only yes or no.",
                    "Remind": "You will be given an Email text. Answer yes if the email is reminding recipients of coming deadlines or to keep commitment, otherwise answer no. Please respond with only yes or no.",
                    "Thank you/Welcome": "You will be given an Email text. Answer yes if the email is intended to thank, congratulate, apologize, or welcome the recipients, otherwise answer no. Please respond with only yes or no.",
                    "spam":"You will be given an Email text. Answer yes if the email is spam or has an empty body, otherwise answer no. Please respond with only yes or no.",
                    "Sender Expectation": "You will be given an Email text. Answer yes if the email sender expects an explicit email in reply, otherwise answer no. Please respond with only yes or no."
                    }

llm = LLM(model=model_dir)  # Create an LLM.
data_df = pd.read_csv(data_dir)
data_df = data_df[data_df["split"] == "test"]
email_text_list = data_df["cleaned_text"]

prediction_df_list = {}
errors = 0

for intent_name in tqdm(user_prompt_dict):
    prediction_df_list["pred_Act:::"+intent_name] = []
    
    answer_prompts = []
    for email_text in email_text_list:
        answer_prompts.append(llama_template.format(user_prompt=user_prompt_dict[intent_name], email_text=email_text))
    outputs = llm.generate(answer_prompts)
    
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.replace("\n", "").strip().lower()
        if "yes" in text.lower():
            prediction_df_list["pred_Act:::"+intent_name].append(1)
        else:
            prediction_df_list["pred_Act:::"+intent_name].append(0)
            if "no" not in text.lower():
                print(text)
                errors += 1
    

data_df = data_df.assign(**prediction_df_list)

metrics = {}
for key in prediction_df_list:
    intent_name = key.split(":::")[-1]
    metrics[intent_name] = {}

    if "spam" in intent_name:
        y_test_local = data_df["label_spam"]
    elif "Sender Expectation" in intent_name:
        y_test_local = data_df["label_Sender Expectation"]
    else:
        y_test_local = data_df["label_Act:::"+intent_name]

    y_pred_local = data_df["pred_Act:::"+intent_name]

    metrics[intent_name]['precision'] = precision_score(y_test_local, y_pred_local)
    metrics[intent_name]['recall'] = recall_score(y_test_local, y_pred_local)
    metrics[intent_name]['f1'] = f1_score(y_test_local, y_pred_local)

print("Errors: " + str(errors))
metrics_df = pd.DataFrame.from_dict(metrics)
print(metrics_df)
metrics_df.to_csv("llama_zeroshot_independent_metrics.csv", index=True)
data_df.to_csv("llama_zeroshot_independent.csv", index=False)