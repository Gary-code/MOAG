# paths
mode = 'pred'
qa_path = '/home/xie/下载/VQA-all/VQA2/'#'../data/'#''/home/xie/下载/VQA-all/VQA2/'  # directory containing the question and annotation jsons
preprocessed_trainval_path = '/home/xie/下载/VQA-all/VQG-code/x_genome-trainval.h5'#'/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/faster-rcnn.pytorch-pytorch-1.0/x_genome-trainval.h5'#'/home/xie/下载/VQA-all/VQG-code/x_genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
vocabulary_path = '/home/xie/下载/VQA-all/VQG-code/xxxx_vocab.json'#'./processed/xxxx_vocab.json'#'/home/xie/下载/VQA-all/VQG-code/xxx_vocab.json'  # path where the used vocabularies for question and answers are saved to
#train_questions_path = qa_path + 'v2_OpenEnded_mscoco_train2014_questions.json'
#train_answers_path = qa_path + 'v2_mscoco_train2014_annotations.json'
train_answers_path =  '/home/xie/下载/ST-VQA_Loc-master/STVQA_v2/train_answer_2.json'

train_questions_path = '/home/xie/下载/ST-VQA_Loc-master/STVQA_v2/train_question.json'
valid_questions_path = '/home/xie/下载/ST-VQA_Loc-master/STVQA_v2/test_question.json'
#valid_questions_path = qa_path + 'v2_OpenEnded_mscoco_val2014_questions.json'
#valid_answers_path = qa_path + 'v2_mscoco_train2014_annotations.json'
valid_answers_path =  '/home/xie/下载/ST-VQA_Loc-master/STVQA_v2/test_answer_2.json'

#test_questions_path = qa_path + 'v2_OpenEnded_mscoco_test2015_questions.json'
#test_answers_path = qa_path + 'v2_mscoco_test2015_annotations.json'


#train_questions_path = '/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/VQG/VQA1/OpenEnded_mscoco_train2014_questions.json'
#train_answers_path = '/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/VQG/VQA1/scoco_train2014_annotations.json'
#valid_questions_path = '/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/VQG/VQA1/OpenEnded_mscoco_val2014_questions.json'
#valid_answers_path = '/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/VQG/VQA1/mscoco_val2014_annotations.json'
#test_questions_path = qa_path + 'v2_OpenEnded_mscoco_test2015_questions.json'
#test_answers_path = qa_path + 'v2_mscoco_test2015_annotations.json'

#train_questions_path = qa_path + 'OpenEnded_mscoco_train2014_questions.json'
#train_answers_path = qa_path + 'mscoco_train2014_annotations.json'
#valid_questions_path = qa_path + 'OpenEnded_mscoco_val2014_questions.json'
#valid_answers_path = qa_path + 'mscoco_val2014_annotations.json'
#test_questions_path = qa_path + 'v2_OpenEnded_mscoco_test2015_questions.json'
#test_answers_path = qa_path + 'v2_mscoco_test2015_annotations.json'


glove_index = '../data/dictionary.pkl'
embedding_path = '../data/glove6b_init_300d.npy'
glove_emc = '/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/VQG/word_embedding.pkl'
min_word_freq = 3
max_q_length = 666 # question_length = min(max_q_length, max_length_in_dataset)
max_a_length = 4

max_o_length = 20
batch_size = 32
data_workers = 4
normalize_box = True

seed = 2020
