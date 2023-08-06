import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torchvision import models
import random
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import pickle

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(1024, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim*2, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(1024, 100)

    def forward(self, image_features, answers_features,labels_features,ocr):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        #att1 = self.features_att(labels_features)  # (batch_size, 36, attention_dim)
        #att2 = self.decoder_att(answers_features)  # (batch_size, 4, attention_dim)


        # bbb =   att2.transpose(1, 2)

        # atten_1 = torch.matmul(att1, att2.transpose(1, 2))

        # attent_2 = torch.matmul(atten_1,att2)

        # attent_3 = torch.cat([att1,attent_2],2)

        # att = self.full_att(self.dropout(self.relu(attent_3)))

        # alpha = self.sigmoid(att)
        # att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        # alpha = self.softmax(att)  # (batch_size, 36)

        # attention_weight = labels_features * alpha.unsqueeze(2)
        # attention_weighted = self.l1(attention_weight)

        # attention_weighted_encoding = torch.cat([image_features,attention_weighted],2).sum(dim=1)
        # attention_weighted_encoding = image_features.sum(dim=1)
        # attention_weighted_encoding = self.l1(attention_weighted).sum(dim=1)
                                       #labels_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)


        # attention_weighted_encoding = (image_features * alpha).sum(dim=1)

        attention_weighted_encoding = image_features.sum(dim=1)

        # ocr = self.l1(ocr)

        # attention_weighted_encoding = torch.cat([attention_weighted_encoding,ocr],axis=1)

        return attention_weighted_encoding


class EncoderCNN(nn.Module):
    """
    Generates a representation for an image input.
    """
    def __init__(self, output_size=1024):
        """
        Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.vgg16(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.classifier[-1] = nn.Linear(4096, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights.
        """
        self.cnn.classifier[-1].weight.data.normal_(0.0, 0.02)
        self.cnn.classifier[-1].bias.data.fill_(0)

    def forward(self, images):
        """
        Extract the image feature vectors.
        """
        features = self.cnn(images)
        output = self.bn(features)
        return output


class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)

        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)

        x = self.lin(x)

        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class PseudoCoord(nn.Module):
    def __init__(self):
        super(PseudoCoord, self).__init__()

    def forward(self, b):
        '''
        Input:
        b: bounding box        [batch, num_obj, 4]  (x1,y1,x2,y2)
        Output:
        pseudo_coord           [batch, num_obj, num_obj, 2] (rho, theta)
        '''
        batch_size, num_obj, _ = b.shape
        # [batch, num_obj, 2]
        centers = (b[:, :, 2:] + b[:, :, :2]) * 0.5  # center of each bounding box

        relative_coord = centers.view(batch_size, num_obj, 1, 2) - \
                         centers.view(batch_size, 1, num_obj, 2)  # broadcast: [batch, num_obj, num_obj, 2]

        rho = torch.sqrt(relative_coord[:, :, :, 0] ** 2 + relative_coord[:, :, :, 1] ** 2)
        theta = torch.atan2(relative_coord[:, :, :, 0], relative_coord[:, :, :, 1])
        new_coord = torch.cat((rho.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)
        return new_coord


class GraphLearner(nn.Module):
    def __init__(self, v_features, q_features, mid_features, dropout=0.0, sparse_graph=True):
        super(GraphLearner, self).__init__()
        self.sparse_graph = sparse_graph
        # self.lin1 = FCNet(v_features+q_features, mid_features, activate='relu')
        self.lin1 = FCNet(v_features, mid_features,activate='relu')
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')

    def forward(self, v, q, v_mask, top_K):
        '''
        Input:
        v: visual feature      [batch, num_obj, 2048]
        q: bounding box        [batch, 1024]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none

        Return:
        adjacent_logits        [batch, num_obj, K(sum=1)]
        adjacent_matrix        [batch, num_obj, K(sum=1)]
        '''
        batch_size, num_obj, _ = v.shape
        q_repeated = q.unsqueeze(1).repeat(1, num_obj, 1)
        # d_repeated = difficulty_level.unsqueeze(1).repeat(1, num_obj, 1)

        # v_cat_q = torch.cat((v, q_repeated, d_repeated), dim=2)
        # v_cat_q = torch.cat((v, q_repeated), dim=2)
        v_cat_q = v

        h = self.lin1(v_cat_q)
        h = self.lin2(h)
        h = h.view(batch_size, num_obj, -1)  # batch_size, num_obj, feat_size

        adjacent_logits = torch.matmul(h, h.transpose(1, 2)) # batch_size, num_obj, num_obj

        # object mask
        # mask = torch.matmul(v_mask.unsqueeze(2),  v_mask.unsqueeze(1))
        # adjacent_logits = adjacent_logits * mask
        # sparse adjacent matrix
        top_ind = None
        if self.sparse_graph:
            top_value, top_ind = torch.topk(adjacent_logits, k=top_K, dim=-1, sorted=False)  # batch_size, num_obj, K
        # softmax attention
        adjacent_matrix = F.softmax(adjacent_logits, dim=-1) # batch_size, num_obj, K

        return adjacent_matrix, top_ind


class GraphConv(nn.Module):
    def __init__(self, v_features, mid_features, num_kernels, bias=False):
        super(GraphConv, self).__init__()
        self.num_kernels = num_kernels
        # for graph conv
        self.conv_weights = nn.ModuleList([nn.Linear(
            v_features, mid_features // (num_kernels), bias=bias) for i in range(num_kernels)])
        # for gaussian kernels
        self.mean_rho = Parameter(torch.FloatTensor(num_kernels, 1))
        self.mean_theta = Parameter(torch.FloatTensor(num_kernels, 1))
        self.precision_rho = Parameter(torch.FloatTensor(num_kernels, 1))
        self.precision_theta = Parameter(torch.FloatTensor(num_kernels, 1))

        self.init_param()

    def init_param(self):
        self.mean_rho.data.uniform_(0, 1.0)
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.precision_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0, 1.0)

    def forward(self, v, v_mask, coord, adj_matrix, top_ind, adj, weight_adj=True):
        """
        Input:
        v: visual feature      [batch, num_obj, 2048]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        coord: relative coord  [batch, num_obj, num_obj, 2]  obj to obj relative coord
        adj_matrix: sparse     [batch, num_obj, K(sum=1)]
        top_ind:               [batch, num_obj, K]
        Output:
        v: visual feature      [batch, num_obj, dim]
        """
        batch_size, num_obj, feat_dim = v.shape
        K = adj_matrix.shape[-1]

        conv_v = v.unsqueeze(1).expand(batch_size, num_obj, num_obj,
                                       feat_dim)  # batch_size, num_obj(same), num_obj(diff), feat_dim
        coord_weight = self.get_gaussian_weights(coord)  # batch, num_obj, num_obj(diff), n_kernels

        # slice_idx1 = top_ind.unsqueeze(-1).expand(batch_size, num_obj, K, feat_dim)  # batch_size, num_obj, K, feat_dim
        # slice_idx2 = top_ind.unsqueeze(-1).expand(batch_size, num_obj, K,
        #                                           self.num_kernels)  # batch_size, num_obj, K, num_kernels
        # sparse_v = torch.gather(conv_v, dim=2, index=slice_idx1)  # batch num_obj K feat_dim
        # sparse_weight = torch.gather(coord_weight, dim=2, index=slice_idx2)  # batch, num_obj, K, n_kernels
        if weight_adj:
            adj_mat = adj_matrix.unsqueeze(-1)  # batch, num_obj, K(sum=1), 1
            adj = adj.unsqueeze(-1)
            attentive_v = conv_v * adj_mat * adj # update feature : batch_size, num_obj, K(diff), feat_dim
        else:
            attentive_v = conv_v  # update feature : batch_size, num_obj(same), K(diff), feat_dim
        weighted_neighbourhood = torch.matmul(coord_weight.transpose(2, 3),
                                              attentive_v)  # batch, num_obj, n_kernels, feat_dim
        weighted_neighbourhood = [self.conv_weights[i](weighted_neighbourhood[:, :, i, :]) for i in
                                  range(self.num_kernels)]  # each: batch, num_obj, feat_dim
        output = torch.cat(weighted_neighbourhood, dim=2)  # batch, num_obj(same), feat_dim

        return output

    def get_gaussian_weights(self, coord):
        """
        Input:
        coord: relative coord  [batch, num_obj, num_obj, 2]  obj to obj relative coord

        Output:
        weights                [batch, num_obj, num_obj, n_kernels)
        """
        batch_size, num_obj, _, _ = coord.shape
        # compute rho weights
        diff = (coord[:, :, :, 0].contiguous().view(-1, 1) - self.mean_rho.view(1,
                                                                                -1)) ** 2  # batch*num_obj*num_obj,  n_kernels
        weights_rho = torch.exp(-0.5 * diff /
                                (1e-14 + self.precision_rho.view(1, -1) ** 2))  # batch*num_obj*num_obj,  n_kernels

        # compute theta weights
        first_angle = torch.abs(coord[:, :, :, 1].contiguous().view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle) ** 2)
                                  / (1e-14 + self.precision_theta.view(1, -1) ** 2))

        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0

        # normalise weights
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-14)  # batch*num_obj*num_obj,  n_kernels (sum=-1)

        return weights.view(batch_size, num_obj, num_obj, self.num_kernels)


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, vision_features=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.vision_features = vision_features
        self.attention_dim = attention_dim
        self.embed_dim = 1024
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.top_k_sparse = 16
        num_kernels = 8
        sparse_graph = False
        mid_features = 1024

        self.pseudo_coord = PseudoCoord()
        self.v_dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.graph_learner = GraphLearner(
            v_features=vision_features + 4,
            q_features=embed_dim,
            mid_features=mid_features,
            dropout=0.5,
            sparse_graph=sparse_graph,
        )

        self.graph_conv1 = GraphConv(
            v_features=vision_features + 4,
            mid_features=mid_features * 2,
            num_kernels=num_kernels,
            bias=False
        )

        self.graph_conv2 = GraphConv(
            v_features=mid_features * 2,
            mid_features=mid_features * 2,
            num_kernels=num_kernels,
            bias=False
        )

        self.graph_conv3 = GraphConv(
            v_features=mid_features * 2,
            mid_features=mid_features * 2,
            num_kernels=num_kernels,
            bias=False
        )

        # self.encoder_cnn = EncoderCNN(features_dim)
        self.embedding_x = nn.Embedding(vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)#.from_pretrained(embedding)  # embedding layer
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        self.attention_1 = Attention(vision_features, decoder_dim, attention_dim)  # attention network
        self.difficult_embedding = nn.Embedding(2, self.decoder_dim)  # embedding layer
        self.difficult_fc = weight_norm(nn.Linear(decoder_dim * 2, decoder_dim))

        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(embed_dim + vision_features, decoder_dim, bias=True) # top down attention LSTMCell
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.init_weights()  # initialize some layers with the uniform distribution

        self.features_att = weight_norm(nn.Linear(1024, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim*2, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.difficult_fc.bias.data.fill_(0)
        self.difficult_fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self,batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size,self.decoder_dim).cuda()  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).cuda()
        return h, c

    def forward(self, v,b, v_mask, encoded_answers, encoded_questions, question_lengths,box_diff,ocr_text, encoded_labels=None, adj=None):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_questions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param question_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = v.size(0)
        vocab_size = self.vocab_size
        self.lstm.flatten_parameters()

        # hid_v2 = self.graph_conv2(hid_v1, v_mask, new_coord, adj_matrix, top_ind, weight_adj=False)
        # image_features = self.relu(hid_v2)  # [batch, num_obj, dim]

        # Flatten image

        # Sort input data by decreasing lengths; why? apparent below
        question_lengths, sort_ind = question_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_questions = encoded_questions[sort_ind]
        encoded_answers = encoded_answers[sort_ind]
        encoded_labels = encoded_labels[sort_ind]
        encoded_ocrs = ocr_text[sort_ind]

        # difficulty_level = difficulty_level[sort_ind]
        v = v[sort_ind]
        b = b[sort_ind]
        v_mask = v_mask[sort_ind]
        adj = adj[sort_ind]
        box_diff = box_diff[sort_ind]
        # difficulty_level = self.difficult_embedding(difficulty_level)[:, 0, :]

        # v = self.v_dropout(v)

        # Embedding
        embeddings = self.embedding(encoded_questions)  # (batch_size, max_caption_length, embed_dim)
        answers_embedding = self.embedding(encoded_answers)
        labels_embedding = self.embedding(encoded_labels)

        ocrs_embedding = self.embedding(encoded_ocrs)
        ocrs_embedding = torch.mean(ocrs_embedding, dim=1)


        # att2, (h_a, c_a)  = self.lstm(answers_embedding)
        # att1 = self.features_att(labels_embedding)  # (batch_size, 36, attention_dim)
        # att2 = self.decoder_att(answers_embedding)  # (batch_size, 4, attention_dim)
        # bbb =   att2.transpose(1, 2)
        # atten_1 = torch.matmul(att1, att2.transpose(1, 2))

        # alpha = torch.max(atten_1,dim=2)

        # attent_2 = torch.matmul(atten_1,att2)
        # attent_3 = torch.cat([att1,attent_2],2)
        # att = self.full_att(self.dropout(self.relu(attent_3)))
        # alpha = self.sigmoid(att)
        v = v  # *box_diff.unsqueeze(2)#.values.unsqueeze(2)

        vb = torch.cat((v, b), dim=2)  # [batch, 2048+4]
        new_coord = self.pseudo_coord(b)  # [batch, num_obj, num_obj, 2]


        # print(encoded_answers.size())
        # print(labels_embedding.size())

        answers_lstm, (h_a, c_a) = self.lstm(answers_embedding)

        # adj_matrix, top_ind = self.graph_learner(vb, answers_lstm[:, -1, :], difficulty_level, v_mask, top_K=self.top_k_sparse)  # [batch, num_obj, K]
        # adj_matrix, top_ind = self.graph_learner(vb, answers_lstm[:, -1, :], v_mask, top_K=self.top_k_sparse)  # [batch, num_obj, K]
        adj_matrix, top_ind = self.graph_learner(vb, answers_lstm[:, -1, :], v_mask, top_K=self.top_k_sparse)  # [batch, num_obj, K]

        hid_v1 = self.graph_conv1(vb, v_mask, new_coord, adj_matrix, top_ind, adj, weight_adj=True)
        hid_v1 = self.v_dropout(self.relu(hid_v1)) + v

        hid_v2 = self.graph_conv2(hid_v1, v_mask, new_coord, adj_matrix, top_ind, adj, weight_adj=True)
        hid_v2 = self.v_dropout(self.relu(hid_v2)) + v

        hid_v3 = self.graph_conv3(hid_v2, v_mask, new_coord, adj_matrix, top_ind, adj, weight_adj=True)
        image_features = self.v_dropout(self.relu(hid_v3)) + v

        # image_features_attention = self.attention_1(image_features, answers_lstm[:, -1, :],labels_embedding)
        image_features_attention = self.attention_1(image_features,answers_lstm,labels_embedding, ocrs_embedding)

        # Initialize LSTM state
        h1 = h_a[0]  # self.difficult_fc(torch.cat([h_a[0], difficulty_level], dim=1))
        c1 = c_a[0]  # self.difficult_fc(torch.cat([c_a[0], difficulty_level], dim=1))

        # h1, c1 = h_a[0], c_a[0]

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (question_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, encoded_questions.size(1)-1, vocab_size).cuda()

        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # h1,c1 = self.top_down_attention(
            # torch.cat([image_features_attention[:batch_size_t], embeddings[:batch_size_t, t, :], difficulty_level[:batch_size_t]], dim=1), (h1[:batch_size_t], c1[:batch_size_t]))
            h1, c1 = self.top_down_attention(
                torch.cat([image_features_attention[:batch_size_t], embeddings[:batch_size_t, t, :]], dim=1), (h1[:batch_size_t], c1[:batch_size_t]))
            preds = self.fc1(self.dropout(h1))
            predictions[:batch_size_t, t, :] = preds

        return predictions, predictions, encoded_questions, question_lengths - 1, sort_ind
