import jsonlines
import json
import hashlib
import random
from sklearn.metrics import roc_auc_score


filepath = '../dataset/vicuna_cal_or_eq_train_1000.json'



def cot_majority_vote_n(filepath, threshold, n=15):
	cal_samples = {}
	eq_samples = {}
	# Use haslib md5 to encode every question into an id
	with jsonlines.open('../dataset/gsm8k_train_1000_cal_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in cal_samples:
				cal_samples[qid].append(sample)
			else:
				cal_samples[qid] = []
				cal_samples[qid].append(sample)
	# print(cal_samples)
	with jsonlines.open('../dataset/gsm8k_train_1000_eq_azurellm.jsonl', 'r') as reader:
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
	correct_cnt = 0
	all_cnt = 0
	diff_cnt = 0
	chosen = 0

	# Get the accuracy after classification
	for qid in class_data:
		all_cnt += 1
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

	accuracy = correct_cnt / all_cnt
	# print("Cot@n-Majority-Vote:")
	# print("AUC: {}".format(auc))
	# print("There're {} different results from calculator and equation.".format(diff_cnt))
	# print("Accuracy after classfication: {:.4f}%.".format(accuracy * 100))
	# print("\n")

	return accuracy


def cot_majority_vote_25(filepath, threshold):
	cal_samples = {}
	eq_samples = {}
	# Use haslib md5 to encode every question into an id
	with jsonlines.open('../dataset/gsm8k_train_1000_cal_azurellm.jsonl', 'r') as reader:
		for sample in reader:
			question = sample['question']
			qid = hashlib.md5("{} {}".format("gsm8k", question).encode('utf-8')).hexdigest()
			if qid in cal_samples:
				cal_samples[qid].append(sample)
			else:
				cal_samples[qid] = []
				cal_samples[qid].append(sample)
	# print(cal_samples)
	with jsonlines.open('../dataset/gsm8k_train_1000_eq_azurellm.jsonl', 'r') as reader:
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
	correct_cnt = 0
	all_cnt = 0
	diff_cnt = 0

	# Get the accuracy after classification
	for qid in class_data:
		all_cnt += 1
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

	accuracy = correct_cnt / all_cnt
	# print("Majority-Vote-25:")
	# print("AUC: {}".format(auc))
	# print(threshold)
	# print(eq_cnt)
	# print("There're {} different results from calculator and equation.".format(diff_cnt))
	# print("Accuracy after classfication: {:.4f}%.".format(accuracy * 100))

	return accuracy


def get_op_threshold():
	threshold_list = [1 / 100 * i for i in range(-300, 301)]
	cot5_max_accuracy = 0.5
	cot5_op_threshold = 0
	cot10_max_accuracy = 0.5
	cot10_op_threshold = 0
	cot15_max_accuracy = 0.5
	cot15_op_threshold = 0
	cot20_max_accuracy = 0.5
	cot20_op_threshold = 0
	cot25_max_accuracy = 0.5
	cot25_op_threshold = 0
	for threshold in threshold_list:
		cot5 = cot_majority_vote_n(filepath, threshold, 5)
		cot10 = cot_majority_vote_n(filepath, threshold, 10)
		cot15 = cot_majority_vote_n(filepath, threshold, 15)
		cot20 = cot_majority_vote_n(filepath, threshold, 20)
		cot25 = cot_majority_vote_25(filepath, threshold)
		if cot5 > cot5_max_accuracy:
			cot5_max_accuracy = cot5
			cot5_op_threshold = threshold
		if cot10 > cot10_max_accuracy:
			cot10_max_accuracy = cot10
			cot10_op_threshold = threshold
		if cot15 > cot15_max_accuracy:
			cot15_max_accuracy = cot15
			cot15_op_threshold = threshold
		if cot20 > cot20_max_accuracy:
			cot20_max_accuracy = cot20
			cot20_op_threshold = threshold
		if cot25 > cot25_max_accuracy:
			cot25_max_accuracy = cot25
			cot25_op_threshold = threshold
	print("Cot5: best threshold --- {}, accuracy --- {}".format(cot5_op_threshold, cot5_max_accuracy))
	print("Cot10: best threshold --- {}, accuracy --- {}".format(cot10_op_threshold, cot10_max_accuracy))
	print("Cot15: best threshold --- {}, accuracy --- {}".format(cot15_op_threshold, cot15_max_accuracy))
	print("Cot20: best threshold --- {}, accuracy --- {}".format(cot20_op_threshold, cot20_max_accuracy))
	print("Cot25: best threshold --- {}, accuracy --- {}".format(cot25_op_threshold, cot25_max_accuracy))


if __name__ == '__main__':
	get_op_threshold()