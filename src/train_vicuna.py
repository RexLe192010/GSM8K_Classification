from src.local_llm import LocalLLM, Segment, TextSample
from src.data_utils import get_experience_question_list, create_sample_df
import fire

question_prompt_template = '''
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: You are given the following question:
{question_text}

You should decide how to solve the question to achieve the best accuracy. Here're two options.
Option A: Directly give the answer. 
Option B: Think it step by step. 
Now answer your selected option (A or B), without adding anything else.
ASSISTANT: Option
'''.strip()

def main(
    lora_r=32,
    learning_rate=3e-4,
    lora_modules='qkv', # something like q, k, v, qk, qv, kv, qkv.
    lora_dropout=0.05,
):
    learning_rate = float(learning_rate)
    normal_solver_name_list = [
        'direct', 
        'cot',              
    ]
    maj_solver_name_list = [
    ]
    valid_solver_names = normal_solver_name_list + maj_solver_name_list

    experience_question_list = get_experience_question_list(
        ['./labeled_dataset/mmlu_cot_or_direct.jsonl'], 
        normal_solver_name_list
    )
    sample_df = create_sample_df(experience_question_list, normal_solver_name_list, train_ratio = 0.6, dev_ratio = 0.2, test_ratio = 0.2, 
                                 maj_solver_name_list=maj_solver_name_list, average_normal_solver_metrics=True)

    text_sample_list = []
    # only select train!
    for _, row in sample_df[sample_df.split == "TRAIN"].iterrows():
        segment1 = Segment(
            text=question_prompt_template.format(question_text=row.text),
            is_skip_training=True,
        )
        cot_accuracy_list = []
        direct_accuracy_list = []
        if row['cot_accuracy'] < row['direct_accuracy']:
            segment2 = Segment(
                text='A',
                is_skip_training=False,
            )
        elif row['cot_accuracy'] > row['direct_accuracy']:
            segment2 = Segment(
                text='B',
                is_skip_training=False,
            )
        else:
            continue
        text_sample = TextSample([segment1, segment2])
        text_sample_list.append(text_sample)

    model = LocalLLM("/home/zhiyuhe/vicuna-13b/")
    print("learning_rate:", learning_rate)
    lora_target_modules = []
    for c in lora_modules.lower():
        if c == 'q':
            lora_target_modules.append('q_proj')
        elif c == 'k':
            lora_target_modules.append('k_proj')
        elif c =='v':
            lora_target_modules.append('v_proj')
        else:
            raise NotImplementedError

    model.train_and_save(
        text_sample_list,
        num_epochs=2,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        output_dir='./ret/lr-{:.6f}-lora_r-{}-lora_modules-{}-lora_dropout-{:.3f}'.format(learning_rate, lora_r, lora_modules, lora_dropout),
    )

if __name__ == '__main__':
    fire.Fire(main)