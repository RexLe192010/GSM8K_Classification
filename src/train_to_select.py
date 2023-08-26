from sklearn.linear_model import Ridge
import jsonlines
from datasets import load_dataset
import random
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def recover_dataset():
	"""
	Recover original dataset
	"""
	questions = []
	original_dataset = []
	with jsonlines.open('../dataset/gsm8k_calculator_chatgpt', 'r') as reader:
		for sample in reader:
			questions.append(sample['question'])

	dataset = load_dataset('gsm8k', 'main')
	train_dataset = dataset['train']
	test_dataset = dataset['test']

	for question in questions:
		for sample in train_dataset:
			if sample['question'] == question:
				original_dataset.append(sample)
				continue

	for sample in test_dataset:
		original_dataset.append(sample)

	print(original_dataset)
	print(len(original_dataset))

	with jsonlines.open('../dataset/gsm8k_new_dataset.jsonl', 'w') as writer:
		for sample in original_dataset:
			writer.write(sample)


def construct_train_test():
	"""
	Construct train and test sets based on new 1830-sample-dataset
	"""
	dataset = []
	with jsonlines.open('../dataset/gsm8k_new_dataset.jsonl', 'r') as reader:
		for sample in reader:
			dataset.append(sample)
	dataset = dataset[:1830]
	random.shuffle(dataset)
	train = dataset[:915]
	test = dataset[915:]
	with jsonlines.open('../dataset/gsm8k_new_train.jsonl', 'w') as writer:
		for sample in train:
			writer.write(sample)
	with jsonlines.open('../dataset/gsm8k_new_test.jsonl', 'w') as writer:
		for sample in test:
			writer.write(sample)


def train_cal_eq_models():
	"""
	Train two models to predict the accuracy of choosing Calculator or Equation to solve problems respectively
	"""
	train_dataset = []
	test_dataset = []
	with jsonlines.open('../dataset/gsm8k_new_train.jsonl', 'r') as reader:
		for sample in reader:
			train_dataset.append(sample)
	with jsonlines.open('../dataset/gsm8k_new_test.jsonl', 'r') as reader:
		for sample in reader:
			test_dataset.append(sample)


	train_questions = [sample['question'] for sample in train_dataset]
	test_questions = [sample['question'] for sample in test_dataset]

	transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
	train_questions_embeddings = transformer.encode(train_questions)
	test_questions_embeddings = transformer.encode(test_questions)

	train_cal_answers = []
	test_cal_answers = []
	train_eq_answers = []
	test_eq_answers = []

	with jsonlines.open('../dataset/gsm8k_calculator_chatgpt.jsonl', 'r') as reader:
		samples = [sample for sample in reader]
		questions = [sample['question'] for sample in samples]
		for question in train_questions:
			if question in questions:
				index = questions.index(question)
				sample = samples[index]
				if sample['is_correct']:
					train_cal_answers.append(1)
				else:
					train_cal_answers.append(0)
		for question in test_questions:
			if question in questions:
				index = questions.index(question)
				sample = samples[index]
				test_cal_answers.append(sample['is_correct'])

	with jsonlines.open('../dataset/gsm8k_equation_chatgpt.jsonl', 'r') as reader:
		samples = [sample for sample in reader]
		questions = [sample['question'] for sample in samples]
		for question in train_questions:
			if question in questions:
				index = questions.index(question)
				sample = samples[index]
				if sample['is_correct']:
					train_eq_answers.append(1)
				else:
					train_eq_answers.append(0)
		for question in test_questions:
			if question in questions:
				index = questions.index(question)
				sample = samples[index]
				test_eq_answers.append(sample['is_correct'])

	print(len(train_cal_answers))

	cal_estimator = Ridge(alpha=100)
	eq_estimator = Ridge(alpha=100)
	cal_estimator.fit(train_questions_embeddings, train_cal_answers)
	eq_estimator.fit(train_questions_embeddings,train_eq_answers)

	test_cal_predict = cal_estimator.predict(test_questions_embeddings)
	test_eq_predict = eq_estimator.predict(test_questions_embeddings)

	final_result = []
	for i in range(len(test_questions)):
		if test_cal_predict[i] > test_eq_predict[i]:
			result = test_cal_answers[i]
			final_result.append(result)
		else:
			result = test_eq_answers[i]
			final_result.append(result)


	print("The accuracy after tool selection {}%".format(
		len([1 for result in final_result if result == True]) / len(final_result) * 100))


def classify():
	"""
	Classify problems in the new dataset to check if overfitting happens
	"""
	transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
	samples_different_results = []
	train_samples = []
	test_samples = []
	with jsonlines.open('../dataset/gsm8k_calculator_chatgpt', 'r') as reader:
		cal_samples = [sample for sample in reader]
	with jsonlines.open('../dataset/gsm8k_equation_chatgpt', 'r') as reader:
		eq_samples = [sample for sample in reader]
	for i in range(len(cal_samples)):
		cal_sample = cal_samples[i]
		eq_sample = eq_samples[i]
		if cal_sample['is_correct'] != eq_sample['is_correct']:
			samples_different_results.append(cal_sample) # Notice that cal_sample is appended to train_samples
	with jsonlines.open('../dataset/gsm8k_new_train.jsonl', 'r') as reader1:
		with jsonlines.open('../dataset/gsm8k_new_test.jsonl', 'r') as reader2:
			new_train_questions = [sample['question'] for sample in reader1]
			new_test_questions = [sample['question'] for sample in reader2]
	for sample in samples_different_results:
		if sample['question'] in new_train_questions:
			train_samples.append(sample)
		elif sample['question'] in new_test_questions:
			test_samples.append(sample)

	for sample in train_samples:
		if sample['is_correct']:
			sample['is_correct'] = 1 # Classified as calculator
		else:
			sample['is_correct'] = 0 # Classified as equation
	for sample in test_samples:
		if sample['is_correct']:
			sample['is_correct'] = 1 # Classified as calculator
		else:
			sample['is_correct'] = 0 # Classified as equation

	train_questions = [sample['question'] for sample in train_samples]
	train_questions_embeddings = transformer.encode(train_questions)
	test_questions = [sample['question'] for sample in test_samples]
	test_questions_embeddings = transformer.encode(test_questions)
	train_answers = [sample['is_correct'] for sample in train_samples]
	test_answers = [sample['is_correct'] for sample in test_samples]

	estimator = LogisticRegression()
	estimator.fit(train_questions_embeddings, train_answers)

	test_predict = estimator.predict(test_questions_embeddings)
	print("Results of prediction: ", test_predict)
	print("Compare real answers and predictions: ", test_answers == test_predict)

	auc = roc_auc_score(test_answers, estimator.predict_proba(test_questions_embeddings)[:, 1])
	print("AUC: ", auc)






if __name__ == '__main__':
	# recover_dataset()
	# construct_train_test()
	# train_cal_eq_models()
	classify()

