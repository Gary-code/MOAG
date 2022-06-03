import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.autograd import Variable
from models import DecoderWithAttention
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval
from datasets import get_loader
import random
import numpy as np



# Parameters
data_folder = 'final_dataset'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file = 'best_checkpoint.pth.tar'  # model checkpoint

word_map_file = 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
emb_dim = 1024  # dimension of word embeddings
attention_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
dropout = 0.5


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

with open(config.vocabulary_path, 'r') as j:
    word_map = json.load(j)

seed_everything(config.seed)


# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file, map_location = device)
decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
decoder = decoder.cuda()

decoder.load_state_dict(checkpoint['decoder'])
decoder.eval()

nlgeval = NLGEval()  # loads the evaluator

# Load word map (word2ix)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    # DataLoader
    loader = get_loader('test', config.preprocessed_trainval_path)
    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    with torch.no_grad():
        for i, (v, ques, ans, item, q_length, difficulty_level, b, v_mask, adj) in enumerate(
                tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
            k = beam_size

            # Move to GPU device, if available
            var_params = {
                'requires_grad': False,
            }
            v = Variable(v.cuda(), **var_params) # 1 36 1024
            ans = Variable(ans.cuda(), **var_params)
            difficulty_level = Variable(difficulty_level.cuda(), **var_params)
            b = Variable(b.cuda(), **var_params)
            v_mask = Variable(v_mask.cuda(), **var_params)
            adj = Variable(adj.cuda(), **var_params)



            # hid_v2 = decoder.graph_conv2(hid_v1, v_mask, new_coord, adj_matrix, top_ind, weight_adj=False)
            # imgs = decoder.relu(hid_v2)  # [batch, num_obj, dim]

            # image_features = decoder.encoder_cnn(imgs)
            # image_features = image_features.expand(k, 2048)
            # imgs = image_features.unsqueeze(di6TT5GFm=1)

            difficulty_level = decoder.difficult_embedding(difficulty_level)[:, 0, :]

            answers_embedding = decoder.embedding(ans)
            answers_lstm, (h_a, c_a) = decoder.lstm(answers_embedding)

            vb = torch.cat((v, b), dim=2)

            new_coord = decoder.pseudo_coord(b)  # [batch, num_obj, num_obj, 2]
            adj_matrix, top_ind = decoder.graph_learner(vb, answers_lstm[:, -1, :], difficulty_level, v_mask, top_K=decoder.top_k_sparse)  # [batch, num_obj, K]

            hid_v1 = decoder.graph_conv1(vb, v_mask, new_coord, adj_matrix, top_ind, adj, weight_adj=True)
            hid_v1 = decoder.relu(hid_v1) + v

            hid_v2 = decoder.graph_conv2(hid_v1, v_mask, new_coord, adj_matrix, top_ind, adj, weight_adj=True)
            hid_v2 = decoder.relu(hid_v2) + v

            hid_v3 = decoder.graph_conv2(hid_v2, v_mask, new_coord, adj_matrix, top_ind, adj, weight_adj=True)
            imgs = decoder.relu(hid_v3) + v

            image_features_mean = decoder.attention_1(imgs, answers_lstm[:, -1, :])
            image_features_mean = image_features_mean.expand(k, 2048)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).cuda() # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).cuda()  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            # h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
            h_a = decoder.difficult_fc(torch.cat([h_a[0], difficulty_level], dim=1))
            c_a = decoder.difficult_fc(torch.cat([c_a[0], difficulty_level], dim=1))
            h_a = h_a.expand(k, 1024)
            c_a = c_a.expand(k, 1024)
            h1, c1 = h_a, c_a

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                d = difficulty_level.expand(k, 1024)

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                h1, c1 = decoder.top_down_attention(
                    torch.cat([image_features_mean, embeddings, d], dim=1),
                    (h1,c1))  # (batch_size_t, decoder_dim)
                scores = decoder.fc1(h1)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h1 = h1[prev_word_inds[incomplete_inds]]
                c1 = c1[prev_word_inds[incomplete_inds]]
                image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            for j in range(ques.shape[0]):
                img_ques = ques[j].tolist()
                img_questions = [rev_word_map[w] for w in img_ques if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                references.append([' '.join(img_questions)])

            # Hypotheses
            hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            hypothesis = ' '.join(hypothesis)
            #print(hypothesis)
            hypotheses.append(hypothesis)
            assert len(references) == len(hypotheses)



    # Calculate scores
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    # with open('result_word_true.txt', 'w') as f:
    #     for r in tqdm(references):
    #         f.write(r[0] + '\n')
    # with open('result_word_pred.txt', 'w') as f:
    #     for r in tqdm(hypotheses):
    #         f.write(r + '\n')
    # test_questions_path = '../qg_split/v2_OpenEnded_mscoco_train2014_questions.json'
    # with open(test_questions_path, 'r') as fd:
    #     test_questions_json = json.load(fd)
    # questions = []
    # for question_dict, pred_q in zip(test_questions_json['questions'], hypotheses):
    #     question_dict['pred_question'] = pred_q
    #     questions.append(question_dict)
    # test_questions_json['questions'] = questions
    # with open('./output_file/pseudo_label_train2014_questions.json', 'w') as f:
    #     json.dump(test_questions_json, f)
    return metrics_dict


if __name__ == '__main__':
    beam_size = 5
    metrics_dict = evaluate(beam_size)
    print(metrics_dict)
