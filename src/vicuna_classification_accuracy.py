import jsonlines
import json
import hashlib
import random
from sklearn.metrics import roc_auc_score


filepath_all = '../dataset/vicuna_cal_or_eq.json'
filepath_test = '../dataset/vicuna_cal_or_eq_test.json'

def cot_1(filepath):
	cal_samples = {}
	eq_samples = {}
	# Use haslib md5 to encode every question into an id
	with jsonlines.open('../dataset/gsm8k_test_cal_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in cal_samples:
				cal_samples[qid].append(sample)
			else:
				cal_samples[qid] = []
				cal_samples[qid].append(sample)
	# print(cal_samples)
	# print(len(cal_samples)) 1318 samples in total, the last question not labelled in the file
	with jsonlines.open('../dataset/gsm8k_test_eq_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in eq_samples:
				eq_samples[qid].append(sample)
			else:
				eq_samples[qid] = []
				eq_samples[qid].append(sample)

	for qid in cal_samples:
		samples = cal_samples[qid]
		sample = random.choice(samples)
		cal_samples[qid] = sample
	# print(cal_samples)
	for qid in eq_samples:
		samples = eq_samples[qid]
		sample = random.choice(samples)
		eq_samples[qid] = sample

	with open(filepath, 'r') as file:
		class_data = json.load(file)
		# print(len(class_data))

	cal_correct_cnt = 0
	eq_correct_cnt = 0
	correct_cnt = 0
	all_cnt = 0
	for qid in class_data:
		all_cnt += 1
		if cal_samples[qid]['is_correct']:
			cal_correct_cnt += 1
		if eq_samples[qid]['is_correct']:
			eq_correct_cnt += 1

		logit_diff = class_data[qid]['logit_diff']
		if logit_diff < 0.5:
			ans = eq_samples[qid]['is_correct']
			if ans:
				correct_cnt += 1
		else:
			ans = cal_samples[qid]['is_correct']
			if ans:
				correct_cnt += 1

	cal_accuracy = cal_correct_cnt / all_cnt
	eq_accuracy = eq_correct_cnt / all_cnt
	accuracy = correct_cnt / all_cnt
	print("Cot@1:")
	print("Accuracy of pure arithmetic: {:.4f}%.".format(cal_accuracy * 100))
	print("Accuracy of pure equation: {:.4f}%.".format(eq_accuracy * 100))
	print("Accuracy after classfication: {:.4f}%.".format(accuracy * 100))
	print("\n")


def cot_majority_vote_n(filepath, threshold, n=15):
	cal_samples = {}
	eq_samples = {}
	# Use haslib md5 to encode every question into an id
	with jsonlines.open('../dataset/gsm8k_test_cal_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in cal_samples:
				cal_samples[qid].append(sample)
			else:
				cal_samples[qid] = []
				cal_samples[qid].append(sample)
	# print(cal_samples)
	with jsonlines.open('../dataset/gsm8k_test_eq_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in eq_samples:
				eq_samples[qid].append(sample)
			else:
				eq_samples[qid] = []
				eq_samples[qid].append(sample)


	# Look for cot@n majority-vote in calculator and equation samples
	for qid in cal_samples:
		samples = cal_samples[qid]
		samples = random.sample(samples, n)
		majority = None
		tmp_cnt = {}
		max_cnt = float('-inf')
		for sample in samples:
			result = sample['final_ans']
			if result in tmp_cnt:
				tmp_cnt[result].append(sample)
			else:
				tmp_cnt[result] = []
				tmp_cnt[result].append(sample)
		for result in tmp_cnt:
			cnt = len(tmp_cnt[result])
			if cnt > max_cnt:
				majority = result
				max_cnt = cnt
		choices = tmp_cnt[majority]
		choice = random.choice(choices)
		cal_samples[qid] = choice

	for qid in eq_samples:
		samples = eq_samples[qid]
		samples = random.sample(samples, n)
		majority = None
		tmp_cnt = {}
		max_cnt = float('-inf')
		for sample in samples:
			result = sample['final_ans']
			if result in tmp_cnt:
				tmp_cnt[result].append(sample)
			else:
				tmp_cnt[result] = []
				tmp_cnt[result].append(sample)
		for result in tmp_cnt:
			cnt = len(tmp_cnt[result])
			if cnt > max_cnt:
				majority = result
				max_cnt = cnt
		choices = tmp_cnt[majority]
		choice = random.choice(choices)
		eq_samples[qid] = choice


	with open(filepath, 'r') as file:
		class_data = json.load(file)


	# Initialize all the variables needed in the next stages
	cal_correct_cnt = 0
	eq_correct_cnt = 0
	correct_cnt = 0
	all_cnt = 0
	diff_cnt = 0
	chosen = 0
	actual_labels = []
	predicted_labels = []

	# Calculate AUC
	for qid in class_data:
		cal_ans = cal_samples[qid]['is_correct']
		eq_ans = eq_samples[qid]['is_correct']
		logit_diff = class_data[qid]['logit_diff']
		if cal_ans != eq_ans:
			if cal_ans:
				actual_labels.append(1)
			else:
				actual_labels.append(0)

			if logit_diff > threshold:
				predicted_labels.append(1)
			elif logit_diff < threshold:
				predicted_labels.append(0)
			else:
				predicted_labels.append(random.choice([0, 1]))
		# No need to consider the case when cal has the same answer as eq
	auc = roc_auc_score(actual_labels, predicted_labels)
	# Get the accuracy after classification
	for qid in class_data:
		all_cnt += 1
		if cal_samples[qid]['is_correct']:
			cal_correct_cnt += 1
		if eq_samples[qid]['is_correct']:
			eq_correct_cnt += 1

		if cal_samples[qid]['is_correct'] != eq_samples[qid]['is_correct']:
			diff_cnt += 1
			# print(cal_samples[qid]['question'])
			# print("Calculator: {}".format(cal_samples[qid]['is_correct']))
			# print("Equation: {}".format(eq_samples[qid]['is_correct']))
			# print("Result: {}".format(class_data[qid]['result']))
			# print("\n")

		logit_diff = class_data[qid]['logit_diff']
		if logit_diff > threshold:
			chosen += 1
			ans = cal_samples[qid]['is_correct']
			if ans:
				correct_cnt += 1
		elif logit_diff < threshold:
			chosen += 1
			ans = eq_samples[qid]['is_correct']
			if ans:
				correct_cnt += 1
		else:
			answers = [cal_samples[qid]['is_correct'], eq_samples[qid]['is_correct']]
			ans = random.choice(answers)
			if ans:
				correct_cnt += 1

	cal_accuracy = cal_correct_cnt / all_cnt
	eq_accuracy = eq_correct_cnt / all_cnt
	accuracy = correct_cnt / all_cnt
	print("Cot@{}-Majority-Vote:".format(n))
	print("AUC: {}".format(auc))
	print("There're {} different results from calculator and equation.".format(diff_cnt))
	print("Accuracy of pure arithmetic: {:.4f}%.".format(cal_accuracy * 100))
	print("Accuracy of pure equation: {:.4f}%.".format(eq_accuracy * 100))
	print("Accuracy after classfication: {:.4f}%.".format(accuracy * 100))
	print("\n")


def cot_majority_vote_25(filepath, threshold):
	cal_samples = {}
	eq_samples = {}
	# Use haslib md5 to encode every question into an id
	with jsonlines.open('../dataset/gsm8k_test_cal_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in cal_samples:
				cal_samples[qid].append(sample)
			else:
				cal_samples[qid] = []
				cal_samples[qid].append(sample)
	# print(cal_samples)
	with jsonlines.open('../dataset/gsm8k_test_eq_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in eq_samples:
				eq_samples[qid].append(sample)
			else:
				eq_samples[qid] = []
				eq_samples[qid].append(sample)


	# Look for majority-vote in calculator and equation samples
	for qid in cal_samples:
		samples = cal_samples[qid]
		majority = None
		tmp_cnt = {}
		max_cnt = float('-inf')
		for sample in samples:
			result = sample['final_ans']
			if result in tmp_cnt:
				tmp_cnt[result].append(sample)
			else:
				tmp_cnt[result] = []
				tmp_cnt[result].append(sample)
		for result in tmp_cnt:
			cnt = len(tmp_cnt[result])
			if cnt > max_cnt:
				majority = result
				max_cnt = cnt
		choices = tmp_cnt[majority]
		choice = random.choice(choices)
		cal_samples[qid] = choice

	for qid in eq_samples:
		samples = eq_samples[qid]
		majority = None
		tmp_cnt = {}
		max_cnt = float('-inf')
		for sample in samples:
			result = sample['final_ans']
			if result in tmp_cnt:
				tmp_cnt[result].append(sample)
			else:
				tmp_cnt[result] = []
				tmp_cnt[result].append(sample)
		for result in tmp_cnt:
			cnt = len(tmp_cnt[result])
			if cnt > max_cnt:
				majority = result
				max_cnt = cnt
		choices = tmp_cnt[majority]
		choice = random.choice(choices)
		eq_samples[qid] = choice


	with open(filepath, 'r') as file:
		class_data = json.load(file)


	# Initialize all the variables needed in the next stages
	cal_correct_cnt = 0
	eq_correct_cnt = 0
	correct_cnt = 0
	all_cnt = 0
	diff_cnt = 0
	eq_cnt = 0
	actual_labels = []
	predicted_labels = []

	# Calculate AUC
	for qid in class_data:
		cal_ans = cal_samples[qid]['is_correct']
		eq_ans = eq_samples[qid]['is_correct']
		logit_diff = class_data[qid]['logit_diff']
		if cal_ans != eq_ans:
			if cal_ans:
				actual_labels.append(1)
			else:
				actual_labels.append(0)

			if logit_diff > threshold:
				predicted_labels.append(1)
			elif logit_diff < threshold:
				predicted_labels.append(0)
				eq_cnt += 1
			else:
				predicted_labels.append(random.choice([0, 1]))
		# No need to consider the case when cal has the same answer as eq
	auc = roc_auc_score(actual_labels, predicted_labels)
	# Get the accuracy after classification
	for qid in class_data:
		all_cnt += 1
		if cal_samples[qid]['is_correct']:
			cal_correct_cnt += 1
		if eq_samples[qid]['is_correct']:
			eq_correct_cnt += 1

		if cal_samples[qid]['is_correct'] != eq_samples[qid]['is_correct']:
			diff_cnt += 1
			# print(cal_samples[qid]['question'])
			# print("Calculator: {}".format(cal_samples[qid]['is_correct']))
			# print("Equation: {}".format(eq_samples[qid]['is_correct']))
			# print("Result: {}".format(class_data[qid]['result']))
			# print("\n")

		logit_diff = class_data[qid]['logit_diff']
		if logit_diff > threshold:
			ans = cal_samples[qid]['is_correct']
			if ans:
				correct_cnt += 1
		elif logit_diff < threshold:
			ans = eq_samples[qid]['is_correct']
			if ans:
				correct_cnt += 1
		else:
			answers = [cal_samples[qid]['is_correct'], eq_samples[qid]['is_correct']]
			ans = random.choice(answers)
			if ans:
				correct_cnt += 1

	cal_accuracy = cal_correct_cnt / all_cnt
	eq_accuracy = eq_correct_cnt / all_cnt
	accuracy = correct_cnt / all_cnt
	print("Majority-Vote-25:")
	print("AUC: {}".format(auc))
	print(threshold)
	print(eq_cnt)
	print("There're {} different results from calculator and equation.".format(diff_cnt))
	print("Accuracy of pure arithmetic: {:.4f}%.".format(cal_accuracy * 100))
	print("Accuracy of pure equation: {:.4f}%.".format(eq_accuracy * 100))
	print("Accuracy after classfication: {:.4f}%.".format(accuracy * 100))



if __name__ == '__main__':
	# cot_1(filepath_test)
	cot_majority_vote_n(filepath_test, 0.96, 5)
	cot_majority_vote_n(filepath_test, 1.75, 10)
	cot_majority_vote_n(filepath_test, 2.37, 15)
	cot_majority_vote_n(filepath_test, 1.21, 20)
	cot_majority_vote_25(filepath_test, 1.14)