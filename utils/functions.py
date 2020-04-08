# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-14 12:23:52
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def to_adj_matrix(adj):
    matrix = []
    for _ in range(len(adj)):
        matrix.append([0]*len(adj))
    for i, adj_i in enumerate(adj):
        for j in adj_i:
            matrix[i][j] = 1
            matrix[j][i] = 1
    return matrix


def heads_to_adj(heads, max_hop):
    # find all the 1-order adj nodes
    adj_1 = [[] for _ in range(len(heads))]
    for i, head in enumerate(heads):
        if head >= 0:
            if head not in adj_1[i]:
                adj_1[i].append(head)
            if i not in adj_1[head]:
                adj_1[head].append(i)
    if max_hop == 1:
        return [to_adj_matrix(adj_1)]

    # find all the 2-order adj nodes based on adj_1
    adj_2 = [[] for _ in range(len(heads))]
    for i, i_adj in enumerate(adj_1):
        for j in i_adj:
            for k in adj_1[j]:
                if k not in adj_2[i] and k not in adj_1[i] and k != i:
                    adj_2[i].append(k)
    if max_hop == 2:
        return [to_adj_matrix(adj_1), to_adj_matrix(adj_2)]

    # find all the 3-order adj nodes based on adj_2
    adj_3 = [[] for _ in range(len(heads))]
    for i, i_adj in enumerate(adj_2):
        for j in i_adj:
            for k in adj_1[j]:
                if k not in adj_3[i] and k not in adj_2[i] and k not in adj_1[i] and k != i:
                    adj_3[i].append(k)

    if max_hop == 3:
        return [to_adj_matrix(adj_1), to_adj_matrix(adj_2), to_adj_matrix(adj_3)]

    return RuntimeError("not support")

# def heads_to_adj(heads):
#     adj = []
#     for _ in range(len(heads)):
#         adj.append([0]*len(heads))
#     for i, head in enumerate(heads):
#         if head >= 0:
#             adj[head][i] = 1
#             adj[i][head] = 1 # undirected
#
#     return adj




def read_instance(input_file, max_hop, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length, sentence_classification=False, split_token='\t', lowercase_tokens=False, char_padding_size=-1, char_padding_symbol = '</pad>'):
    feature_num = len(feature_alphabets)
    in_lines = open(input_file,'r', encoding="utf8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    words_processed = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []
    heads = []

    ## if sentence classification data format, splited by \t
    if sentence_classification:
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split(split_token)
                sent = pairs[0]
                if sys.version_info[0] < 3:
                    sent = sent.decode('utf-8')
                original_words = sent.split()
                for word in original_words:
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    word_Ids.append(word_alphabet.get_index(word))
                    ## get char
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                        assert(len(char_list) == char_padding_size)
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)

                label = pairs[-1]
                label_Id = label_alphabet.get_index(label)
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                ## combine together and return, notice the feature/label as different format with sequence labeling task
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                    instence_texts.append([words, feat_list, chars, label])
                    instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
                words = []
                features = []
                chars = []
                char_Ids = []
                word_Ids = []
                feature_Ids = []
                label_Ids = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
            instence_texts.append([words, feat_list, chars, label])
            instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
            words = []
            features = []
            chars = []
            char_Ids = []
            word_Ids = []
            feature_Ids = []
            label_Ids = []

    else:
    ### for sequence labeling data format i.e. CoNLL 2003
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                words.append(word)
                if number_normalized:
                    word = normalize_word(word)
                if lowercase_tokens:
                    word = word.lower()
                words_processed.append(word)
                label = pairs[-1]
                labels.append(label)
                word_Ids.append(word_alphabet.get_index(word))
                label_Ids.append(label_alphabet.get_index(label))
                # get_heads
                head = int(pairs[1])
                heads.append(head)
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                features.append(feat_list)
                feature_Ids.append(feat_Id)
                ## get char
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                    assert(len(char_list) == char_padding_size)
                else:
                    ### not padding
                    pass
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
            else:
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                    adj = heads_to_adj(heads, max_hop)
                    instence_texts.append([words, features, chars, labels, words_processed, adj])
                    instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids])
                words = []
                words_processed = []
                features = []
                chars = []
                labels = []
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                label_Ids = []
                heads = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
            adj = heads_to_adj(heads, max_hop)
            instence_texts.append([words, features, chars, labels, words_processed, adj])
            instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids])
            # words = []
            # words_processed = []
            # features = []
            # chars = []
            # labels = []
            # word_Ids = []
            # feature_Ids = []
            # char_Ids = []
            # label_Ids = []
    return instence_texts, instence_Ids



def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.zeros([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
