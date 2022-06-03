import json
import pandas as pd
from datasets import prepare_answers, prepare_questions
import nltk
from tqdm import tqdm
import numpy as np
from nlgeval import NLGEval


nlgeval = NLGEval()


with open('./output_file/IMVQG.json', 'r') as fd:
    baseline_answer = json.load(fd)

# with open('./output_file/baseline_answer.json', 'r') as fd:
#     baseline_answer_fusion = json.load(fd)

with open('./output_file/DGCN_GDC_DDC.json', 'r') as fd:
    DC_DDC = json.load(fd)

with open('./output_file/reverse_DGCN_GDC_DDC.json', 'r') as fd:
    reverse_DC_DDC = json.load(fd)

with open('../qg_split/v2_mscoco_test2015_annotations.json', 'r') as fd:
    answer = json.load(fd)

result = pd.DataFrame()
a = list(prepare_answers(answer))
a = [' '.join(i) for i in a]

q = list(prepare_questions(baseline_answer))
result['image_id'] = [line["image_id"] for line in baseline_answer['questions']]
result['answer'] = a
result['difficult_level'] = [line["difficult_level"] for line in baseline_answer['questions']]
result['question'] = q
result['IMVQG'] = [line["pred_question"] for line in baseline_answer['questions']]
# result['answer_fusion'] = [line["pred_question"] for line in baseline_answer_fusion['questions']]
result['Object-AFGDD'] = [line["pred_question"] for line in DC_DDC['questions']]
result['reverse_Object-AFGDD'] = [line["pred_question"] for line in reverse_DC_DDC['questions']]


reference = [[' '.join(q)] for q in result['question']]

hypothesis = [q for q in result['Object-AFGDD']]


metrics_dict = nlgeval.compute_metrics(reference, hypothesis)
print(metrics_dict)

print('yes or no question')
bool_reference = []
bool_hypothesis = []
for a, q, h_q in zip(result['answer'], result['question'], result['Object-AFGDD']):
    if a in ['yes', 'no']:
        bool_reference.append([' '.join(q)])
        bool_hypothesis.append(h_q)
metrics_dict = nlgeval.compute_metrics(bool_reference, bool_hypothesis)
print('num : {}'.format(len(bool_reference)))
print(metrics_dict)

print('num question')
num_reference = []
num_hypothesis = []
for a, q, h_q in zip(result['answer'], result['question'], result['Object-AFGDD']):
    if a.isdigit():
        num_reference.append([' '.join(q)])
        num_hypothesis.append(h_q)
metrics_dict = nlgeval.compute_metrics(num_reference, num_hypothesis)
print('num : {}'.format(len(num_hypothesis)))
print(metrics_dict)






