from llm import ChatGPT
from internal_llm import LLMClient
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from classify import ans2num
import random
import json
import jsonlines
import openai
import queue
import time
import os
import re



gsm8k_calculator_prompt_template = '''

# Examples

Question:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Let's think step by step
Natalia sold 48/2 = <<48/2=24>>24 clips in May. 
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. 
The answer is 72


Question:
Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Let's think step by step
In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. 
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. 
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. 
The answer is 5


Question:
Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Let's think step by step
Maila read 12 x 2 = <<12*2=24>>24 pages today. 
So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday. 
There are 120 - 36 = <<120-36=84>>84 pages left to be read. 
Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages. 
The answer is 42


Question:
Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?
Let's think step by step
There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers. 
So in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers. 
Purple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers. 
That means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers. 
So in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden. 
The answer is 35



Question:
{}
Let's think step by step
'''.strip()


gsm8k_equation_prompt_template = '''
# Examples


Question:
Jim collects model cars, and he has 301 models total. Jim has 4 times as many Buicks as Fords, and 3 more than twice the number of Fords than Chevys. How many Buicks does Jim have?
Let's think step by step
Let x represent the number of Chevys 
Fords:3+2x 
Buicks:4(3+2x)=12+8x 
Total:x+3+2x+12+8x=301 
11x+15=301 
11x=286 
x=<<26=26>>26 
Buicks:12+8(26)=220 
The answer is 220


Question:
Inez has $150. She spends one-half on hockey skates and a certain amount on hockey pads. If Inez has $25 remaining, how much did the hockey pads cost, together, in dollars?
Let's think step by step
She spent 150/2=$<<150/2=75>>75 on hockey skates. Let X be the amount Inez spent on hockey pads. 
150-75-X=25
75-X=25
X=<<50=50>>50
The hockey pads cost $<<50=50>>50.
The answer is 50


Question:
Becca, Smendrick, and PJ have collections of Magic Cards.  There is a total of 341 cards.  Becca has 12 more than Smendrick, and Smendrick has 3 times the amount of cards that PJ has.  How many cards does Becca have?
Let's think step by step
Let x represent the number of cards PJ has.
Smendrick:3x
Becca:3x+12
Total:x+3x+3x+12=341
7x+12=341
7x=329
x=<<47=47>>47 cards
Becca:3(47)+12=153 cards.
The answer is 153


Question:
Jean has three times as much money as Jane. They have a combined total of $76. How much money does Jean have?
Let's think step by step
Let's assume the total amount of money that Jane has is m. 
Since Jean has 3 times as much money as Jane, he has 3*m=3m Combined. 
Jean and Jane have a total of m+3m = $76. 
This evaluates to 4m=$76. 
The total amount of money that Jane has, represented by m, is m=$76/4. 
Jane has m=$<<19=19>>19. 
Since Jean has three times as much money as Jane, Jean has 3*$19=$57. 
The answer is 57



Question:
{}
Let's think step by step
'''.strip()



def ans_by_azurellm(question, template, llm_client):
    start_time = time.time()
    prompt = template.format(question)
    request_data = {
        "prompt": prompt,
        "max_tokens": 1536,
        "temperature": 0.6,
        "top_p":1,
        "n":25,
        "stream": False,
        "stop": None,
    }
    generated_text = llm_client.send_request('dev-gpt-35-turbo', request_data)
    end_time = time.time()
    if 'error' in generated_text:
    	time.sleep(61)
    	start_time = time.time()
    	generated_text = llm_client.send_request('dev-gpt-35-turbo', request_data)
    	end_time = time.time()
    print(generated_text)
    answers = []
    try:
    	text = generated_text['choices']
    	for choice in text:
    		answers.append(choice['text'])
    except KeyError:
    	text = "No result"
    time_cost = end_time - start_time
    return answers, time_cost



def get_final_results(filepath, template):
	log_list = []
	if os.path.exists(filepath):
		with jsonlines.open(filepath, 'r') as reader:
			for log in reader:
				log_list.append(log)
			print("Has read {} records.".format(len(log_list)))

	q = queue.Queue()
	problemlist = []
	with jsonlines.open('/home/v-xinyile/llm-switch/raw_dataset/gsm8k_train_1000.jsonl', 'r') as reader:
		for data in reader:
			problemlist.append(data)
			q.put(data)
	print(len(problemlist))


	llm_client = LLMClient()
	while not(q.empty()):
		sample = q.get()
		question = sample['text']
		if question in [log['question'] for log in log_list]:
			print("This problem is solved, skip!")
			continue

		answer = sample['target']
		true_answer_string = answer.split('####')[-1].strip()
		true_answer = float(true_answer_string.replace(",", ""))
		answer_text, time_cost = ans_by_azurellm(question, template, llm_client)

		target = "The answer is"
		for ans in answer_text:
			print(ans)
			match = re.search(rf"{target}\s*(\d+(\.\d+)?)", ans)
			if match:
				number_str = match.group(1)
				try:
					result = float(number_str)
				except ValueError:
					# Find "The answer is", but cannot find the numerical answer
					result = "No result"
					print("Cannot convert the string to number")
			else:
				# Cannot find "The answer is"
				result = "No result"
				print("Cannot find target in the string")

			log_list.append({
				'question': question,
				'raw_response': ans,
				'final_ans': result,
				'true_ans': true_answer,
				'is_correct': result == true_answer,
				'time_cost': time_cost
				})


			# print("Has processed ", all_count, " problems.")
			# print("Till now, the accuracy is : ", float(correct_count / all_count))
			# It's unecessary to calculate the current accuracy, because sometimes server fails
			print("\n")
			print("One question over.\n")


		if len(log_list) % 5 == 0:
			with jsonlines.open(filepath, 'w') as writer:
				for log in log_list:
					writer.write(log)
				print("has processed {} logs success rate {:.2f}%".format(
	                   len(log_list),
	                   len([1 for log in log_list if log['is_correct']]) / len(log_list) * 100
	                   ))
				print("\n")




if __name__ == '__main__':
	get_final_results('../dataset/gsm8k_train_1000_cal_azurellm.jsonl', gsm8k_calculator_prompt_template)
	get_final_results('../dataset/gsm8k_train_1000_eq_azurellm.jsonl', gsm8k_equation_prompt_template)
	# count_true()