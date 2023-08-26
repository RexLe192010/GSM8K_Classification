from llm import ChatGPT
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from classify import ans2num
import random
import jsonlines
import openai
import time
import os



gsm8k_calculator_prompt_template = '''
# Task
Your task is to give solutions to the questions. Here are some examples.

# Examples

Question:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer:
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72


Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10


Question:
Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer:
In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5


Question:
Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer:
Maila read 12 x 2 = <<12*2=24>>24 pages today.
So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.
There are 120 - 36 = <<120-36=84>>84 pages left to be read.
Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.
#### 42


Question:
Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?
Answer:
There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.
So in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.
Purple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.
That means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.
So in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.
#### 35


# Output

Now give you the following questions:

Question:
{}

Output your Answer and your final result. Use arithmetic operations step by step to solve the problems.
Do not use equations to solve the problems.
You should include your final result after #### (Do not use comma or dollar symbol).
'''


gsm8k_equation_prompt_template = '''
# Task
Your task is to give solutions to the questions. Here are some examples.

# Examples


Question:
Kennedy's house is 600 square feet larger than 4 times Benedict's house. If Kennedy's house is 10000 square feet, how many square feet is Benedict's house?
Answer:
Let the number of square feet in Benedict's house be x. 
So, we can write the equation 4 * x + 600 = 10000. 
Subtracting 600 from both sides we get 4 * x = 9400 Dividing both sides by 4 we get x = 2350 square feet. 
#### 2350


Question:
Jim collects model cars, and he has 301 models total. Jim has 4 times as many Buicks as Fords, and 3 more than twice the number of Fords than Chevys. How many Buicks does Jim have?
Answer:
Let x represent the number of Chevys 
Fords:3+2x 
Buicks:4(3+2x)=12+8x 
Total:x+3+2x+12+8x=301 
11x+15=301 
11x=286 
x=<<26=26>>26 
Buicks:12+8(26)=220 
#### 220


Question:
Inez has $150. She spends one-half on hockey skates and a certain amount on hockey pads. If Inez has $25 remaining, how much did the hockey pads cost, together, in dollars?
</Question>
<Label>
She spent 150/2=$<<150/2=75>>75 on hockey skates.
Let X be the amount Inez spent on hockey pads.
150-75-X=25
75-X=25
X=<<50=50>>50
The hockey pads cost $<<50=50>>50.
#### 50


Question:
Becca, Smendrick, and PJ have collections of Magic Cards.  There is a total of 341 cards.  Becca has 12 more than Smendrick, and Smendrick has 3 times the amount of cards that PJ has.  How many cards does Becca have?
Answer:
Let x represent the number of cards PJ has.
Smendrick:3x
Becca:3x+12
Total:x+3x+3x+12=341
7x+12=341
7x=329
x=<<47=47>>47 cards
Becca:3(47)+12=153 cards.
#### 153


Question:
Jean has three times as much money as Jane. They have a combined total of $76. How much money does Jean have?
Answer:
Let's assume the total amount of money that Jane has is m. 
Since Jean has 3 times as much money as Jane, he has 3*m=3m Combined.
Jean and Jane have a total of m+3m = $76.
This evaluates to 4m=$76.
The total amount of money that Jane has, represented by m, is m=$76/4.
Jane has m=$<<19=19>>19.
Since Jean has three times as much money as Jane, Jean has 3*$19=$57.
#### 57


# Output

Question:
{}

Output your Answer and your final result. Set up equations to solve the problems.
You should include your final answer after #### (Do not use comma or dollar symbol).
'''

def convert_to_float(string_number):
	try:
		number = float(string_number)
		return number
	except ValueError:
		return "No result"


def construct_new_dataset(dataset):
	new_dataset = []
	calculators = []
	equations = []
	train = dataset['train']
	test = dataset['test']

	for sample in train:
		temp = []
		temp.append(sample['answer'])
		if ans2num(temp) == [1]:
			equations.append(sample)
		else:
			calculators.append(sample)

	# print(len(equations)) There're only 219 questions that belong to Equation.

	calculators = random.sample(calculators, 300)

	for sample in equations:
		new_dataset.append(sample)
	for sample in calculators:
		new_dataset.append(sample)
	for sample in test:
		new_dataset.append(sample)


	filename = 'gsm8k_new_dataset.jsonl'
	with jsonlines.open(filename, 'w') as writer:
		for data in new_dataset:
			writer.write(data)

	# print(new_dataset)
	return new_dataset


def ans_by_chatgpt(question, template):
    start_time = time.time()
    chatgpt = ChatGPT(temp=0.3)
    prompt = template.format(question)
    generated_text = chatgpt(prompt)
    print(generated_text)
    end_time = time.time()

    if '####' in generated_text:
    	answer_text = generated_text.split('####')[0].strip()
    	result_text = generated_text.split('####')[-1].strip()
    	result = convert_to_float(result_text)
    else:
    	answer_text = ""
    	result = "No result"
    time_cost = end_time - start_time
    return answer_text, result, time_cost



def get_final_results(filename, template):
	log_list = []
	if os.path.exists(filename):
		with jsonlines.open(filename, 'r') as reader:
			for log in reader:
				log_list.append(log)
			print("Has read {} record.".format(len(log_list)))

	dataset = []

	with jsonlines.open('gsm8k_new_dataset.jsonl', 'r') as reader:
		for data in reader:
			dataset.append(data)

	for data in dataset:
		question = data['question']

		if question in [log['question'] for log in log_list]:
			print("This problem is solved, skip!")
			continue

		answer = data['answer']
		true_answer_string = answer.split('####')[-1].strip()
		true_answer = float(true_answer_string.replace(",", ""))
		answer_text, result, time_cost = ans_by_chatgpt(question, template)


		# print("Has processed ", all_count, " problems.")
		# print("Till now, the accuracy is : ", float(correct_count / all_count))
		# It's unecessary to calculate the current accuracy, because sometimes server fails
		print("\n")

		log_list.append({
			'question': question,
			'raw_response': answer_text,
			'final_ans': result,
			'is_correct': result == true_answer,
			'time_cost': time_cost
			})

		if len(log_list) % 10 == 0:
			with jsonlines.open(filename, 'w') as writer:
				for log in log_list:
					writer.write(log)
				print("has processed {} logs success rate {:.2f}%".format(
                    len(log_list),
                    len([1 for log in log_list if log['is_correct']]) / len(log_list) * 100
                    ))



def count_true():
	equation = []
	calculator = []
	either_one = 0
	calculator_t_equation_f = 0
	equation_t_calculator_f = 0
	with jsonlines.open('gsm8k_equation_chatgpt', 'r') as reader:
		for data in reader:
			equation.append(data)
	with jsonlines.open('gsm8k_calculator_chatgpt', 'r') as reader:
		for data in reader:
			calculator.append(data)
	count = len(equation)
	for i in range(count):
		equation_result = (equation[i])['is_correct']
		calculator_result = (calculator[i])['is_correct']
		if equation_result == True and calculator_result == False:
			equation_t_calculator_f = equation_t_calculator_f + 1
			either_one = either_one + 1
		elif equation_result == False and calculator_result == True:
			calculator_t_equation_f = calculator_t_equation_f + 1
			either_one = either_one + 1
		elif equation_result == True and calculator_result == True:
			either_one = either_one + 1

	print("Equation true while Calculator false: {} out of {} samples".format(equation_t_calculator_f, count))
	print("Calculator true while Equation false: {} out of {} samples".format(calculator_t_equation_f, count))
	print("The accuracy of at least one of them being true: ", either_one / count)




def main():
	# dataset = load_dataset("gsm8k", "main")
	# if os.path.exists('gsm8k_new_dataset.jsonl'):
	# 	get_final_results('gsm8k_calculator_chatgpt', gsm8k_calculator_prompt_template)
	# 	get_final_results('gsm8k_equation_chatgpt', gsm8k_equation_prompt_template)
	# else:
	# 	new_dataset = construct_new_dataset(dataset)
	# 	get_final_results('gsm8k_calculator_chatgpt', gsm8k_calculator_prompt_template)
	# 	get_final_results('gsm8k_equation_chatgpt', gsm8k_equation_prompt_template)
	count_true()

if __name__ == "__main__":
	main()


