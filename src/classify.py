from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sentence_transformers import SentenceTransformer
import string
import random


def ans2num(answer):
	"""
	Convert answers in the dataset to features

	input: answer, a list of strings, each representing an answer to a question in the dataset

	output: ans_copy, a list of integers, each representing the category of the question
	(0 for Calculator and 1 for EquationSolver)
	"""
	ans_copy = []
	for ans in answer:
		ans_copy.append(ans)
	alphabet_lower = list(string.ascii_lowercase)
	alphabet_upper = list(string.ascii_uppercase)
	alphabet_list = []
	for i in alphabet_lower:
		alphabet_list.append(i)
	for j in alphabet_upper:
		alphabet_list.append(j)
	# print(alphabet_list)

	for k in range(len(ans_copy)):
		isEquation = False
		for l in alphabet_list:
			eq_label1 = 'Let ' + l + ' '
			eq_label2 = 'let ' + l + ' '
			if eq_label1 in ans_copy[k] or eq_label2 in ans_copy[k]:
				isEquation = True
				break
		if isEquation:
			ans_copy[k] = 1
		else:
			ans_copy[k] = 0
	# print(ans_copy)
	return ans_copy



def process_questions(question):
    """
    Convert questions into embeddings.

    input: question, a list of strings representing questions in the dataset

    output: question_embeddings, a list of list representing the embeddings of questions
    """
    transformer = SentenceTransformer('all-MiniLM-L6-v2')

    question_embeddings = transformer.encode(question)

    return question_embeddings
    



def main():



    dataset = load_dataset("gsm8k", "main")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Take 1500 samples from 7000+ samples as train dataset; take all 1319 samples as test dataset
    train_samples = list(train_dataset)
    train_samples = random.sample(train_samples, 1500)
    # print(type(train_samples))
    # print(type(test_dataset))

    train_questions = []
    train_answers = []
    for sample in train_samples:
    	train_questions.append(sample['question'])
    	train_answers.append(sample['answer'])
    train_questions_embeddings = process_questions(train_questions)
    train_answers_embeddings = ans2num(train_answers)
    # print(train_answers_embeddings)

    test_questions = []
    test_answers = []
    for sample in test_dataset:
    	# print(type(sample))
    	test_questions.append(sample['question'])
    	test_answers.append(sample['answer'])
    test_questions_embeddings = process_questions(test_questions)
    test_answers_embeddings = ans2num(test_answers)
    # print(test_questions_embeddings, test_answers_embeddings)




    estimator = LogisticRegression()
    estimator.fit(train_questions_embeddings, train_answers_embeddings)


    test_predict = estimator.predict(test_questions_embeddings)
    print("Results of prediction: ", test_predict)
    print("Compare real answers and predictions: ", test_answers_embeddings == test_predict)

    accuracy = estimator.score(test_questions_embeddings, test_answers_embeddings)
    print("The accuracy of predictions: ", accuracy)

    # Introduce AUC as an evaluation metric
    auc = roc_auc_score(test_answers_embeddings, estimator.predict_proba(test_questions_embeddings)[:, 1])
    print("AUC: ", auc)

    coef = estimator.coef_
    intercept = estimator.intercept_
    print("Coefficients and intercept: ", coef, intercept)

if __name__ == "__main__":
    main()



