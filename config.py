# paths
mode = 'pred'
qa_path = './VQA-all/VQA2/'
preprocessed_trainval_path = './VQA-all/VQG-code/x_genome-trainval.h5'
vocabulary_path = './VQA-all/VQG-code/xxxx_vocab.json'

train_answers_path =  './ST-VQA_Loc-master/STVQA_v2/train_answer_2.json'

train_questions_path = './ST-VQA_Loc-master/STVQA_v2/train_question.json'
valid_questions_path = './ST-VQA_Loc-master/STVQA_v2/test_question.json'

valid_answers_path =  './ST-VQA_Loc-master/STVQA_v2/test_answer_2.json'


glove_index = '../data/dictionary.pkl'
embedding_path = '../data/glove6b_init_300d.npy'
glove_emc = './VQA2.0-Recent-Approachs-2018.pytorch-master/VQG/word_embedding.pkl'
min_word_freq = 3
max_q_length = 666
max_a_length = 4

max_o_length = 20
batch_size = 32
data_workers = 4
normalize_box = True

seed = 2020
