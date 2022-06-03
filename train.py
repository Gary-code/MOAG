import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from models import DecoderWithAttention
from datasets import get_loader
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import config
from data import _create_coco_id_to_index
from nlgeval import NLGEval
import h5py
import random
import numpy as np
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Data parameters
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 1024  # dimension of word embeddings
attention_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 15 # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

nlgeval = NLGEval()  # loads the evaluator

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch,data_name, word_map

    seed_everything(config.seed)
    # Read word map
    with open(config.vocabulary_path, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)
    decoder = nn.DataParallel(decoder, device_ids=[0]).cuda()

    if checkpoint is None:
        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder.module.load_state_dict(checkpoint['decoder'])
        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))
       
    # Move to GPU, if available
    # decoder = decoder.cuda()

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss(reduction='sum').to(device)
    criterion_dis = nn.MultiLabelMarginLoss().to(device)

    # Custom dataloaders
    train_loader = get_loader('train', config.preprocessed_trainval_path)
    valid_loader = get_loader('valid', config.preprocessed_trainval_path)
    # test_loader = get_loader('test', config.preprocessed_trainval_path)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 10:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
    
        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce = criterion_ce,
              criterion_dis=criterion_dis,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=valid_loader,
                                decoder=decoder,
                                criterion_ce=criterion_ce,
                                criterion_dis=criterion_dis,
                                word_map=word_map)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            print('save best checkpoint')
            save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, recent_bleu4,
                            is_best)


def train(train_loader, decoder, criterion_ce, criterion_dis, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    count = 0
    # Batches
    for i, (imgs, ques, ans, item, q_length, b, obj_mask, adj,labels,diff_2,ocr_text ) in enumerate(train_loader):
        data_time.update(time.time() - start)


        box_diff = diff_2
        # Move to GPU, ifbels)

        var_params = {
            'requires_grad': False,
        }
        imgs = Variable(imgs.cuda(), **var_params)
        ques = Variable(ques.cuda(), **var_params)
        ans = Variable(ans.cuda(), **var_params)
        q_length = Variable(q_length.cuda(), **var_params)
        #difficult_level = Variable(difficult_level.cuda(), **var_params)
        b = Variable(b.cuda(), **var_params)
        obj_mask = Variable(obj_mask.cuda(), **var_params)
        adj = Variable(adj.cuda(), **var_params)
        labels = Variable(labels.cuda(), **var_params)

        #print(imgs.size())
        #print('--------------------------------------------------')
        #print(labels.size())
        #print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        #print(ques.size())
        #print('**************************************************')
        #print(adj.size())

        # Forward prop.
        scores, scores_d, ques_sorted, decode_lengths, sort_ind = decoder(imgs, b, obj_mask, ans, ques, q_length, box_diff,ocr_text,encoded_labels=labels, adj=adj)
        
        #Max-pooling across predicted words across time steps for discriminative supervision
        scores_d = scores_d.max(2)[0] # batch_size, vocab_size

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = ques_sorted[:, 1:]
        targets_d = torch.zeros(scores_d.size(0),scores_d.size(1)).to(device)
        targets_d.fill_(-1)

        for length in decode_lengths:
            targets_d[:,:length-1] = targets[:,:length-1]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        # loss_d = criterion_dis(scores_d, targets_d.long())
        # print(scores.size())
        # print(targets.size())
        loss_g = criterion_ce(scores, targets)
        # loss = loss_g + (10 * loss_d)
        loss = loss_g
        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()
	
        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.5)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            count += 1


def validate(val_loader, decoder, criterion_ce, criterion_dis, word_map):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad(): 
        for i, (imgs, ques, ans, item, q_length, b, obj_mask, adj, labels,diff_2,ocr_text) in enumerate(val_loader):
            box_diff = diff_2

            #print('----------------------------------------------------------------------------------------------------')
            #print(labels)
            #print('****************************************************************************************************')
            var_params = {
                'requires_grad': False,
            }
            imgs = Variable(imgs.cuda(), **var_params)
            ques = Variable(ques.cuda(), **var_params)
            ans = Variable(ans.cuda(), **var_params)
            q_length = Variable(q_length.cuda(), **var_params)
            #difficult_level = Variable(difficult_level.cuda(), **var_params)
            b = Variable(b.cuda(), **var_params)
            obj_mask = Variable(obj_mask.cuda(), **var_params)
            adj = Variable(adj.cuda(), **var_params)
            labels = Variable(labels.cuda(), **var_params)

            scores, scores_d, ques_sorted, decode_lengths, sort_ind = decoder(imgs, b, obj_mask, ans, ques, q_length, box_diff, ocr_text,encoded_labels=labels, adj=adj)
            
            #Max-pooling across predicted words across time steps for discriminative supervision
            scores_d = scores_d.max(2)[0]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = ques_sorted[:, 1:]
            targets_d = torch.zeros(scores_d.size(0),scores_d.size(1)).to(device)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:,:length-1] = targets[:,:length-1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            # loss_d = criterion_dis(scores_d,targets_d.long())
            loss_g = criterion_ce(scores, targets)
            # loss = loss_g + (10 * loss_d)
            loss = loss_g

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            for j in range(ques_sorted.shape[0]):
                img_ques = ques_sorted[j].tolist()
                img_questions = [w for w in img_ques if w not in {word_map['start'], word_map['pad']}]
                references.append([img_questions])

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2) # batch_size len
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            hypotheses.extend(temp_preds)
            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    bleu4 = round(bleu4, 4)
    with open('result_id_true.txt', 'w') as f:
        for r in tqdm(references):
            r = list(map(str, r[0]))
            f.write(' '.join(r) + '\n')
    with open('result_id_pred.txt', 'w') as f:
        for r in tqdm(hypotheses):
            r = list(map(str, r))
            f.write(' '.join(r) + '\n')

    references_2 = []
    hypotheses_2 = []
    idx2word = {index: word for word, index in word_map.items()}
    with open('result_word_true.txt', 'w') as f:
        for r in tqdm(references):
            w = [idx2word[i] for i in r[0]]
            if 'end' in w:
                w = w[:w.index('end')]
            f.write(' '.join(w) + '\n')
            references_2.append([' '.join(w)])
    with open('result_word_pred.txt', 'w') as f:
        for r in tqdm(hypotheses):
            w = [idx2word[i] for i in r]
            if 'end' in w:
                w = w[:w.index('end')]
            f.write(' '.join(w) + '\n')
            hypotheses_2.append(' '.join(w))

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))
    print(len(references_2))
    print(len(hypotheses_2))
    #metrics_dict = nlgeval.compute_metrics(references_2, hypotheses_2)
    #print(metrics_dict)

    return bleu4#metrics_dict['Bleu_4']


if __name__ == '__main__':
    main()
