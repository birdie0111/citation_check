import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text

def models():
    bert = 'bert_en_uncased_L-12_H-768_A-12' 

    map_name_to_handle = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    }
    bert = hub.KerasLayer(map_name_to_handle[bert])
    return bert

def pre_process():
    bert_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    pre_model = hub.load(bert_preprocess)
    return pre_model

def cos_similarity(vec1, vec2):
    cos_simi = tf.keras.losses.CosineSimilarity(axis=1)
    simi = -cos_simi(vec1, vec2).numpy()
    return simi

def get_cited_abs(author, f_in):
    for info in f_in:
        if author in info:
            info = info.split('\t')[1].strip()
            return info
    return 'author_not_found\n'
        


def make_input(input_f):
    multi = 1
    with open(input_f, 'r', encoding='utf-8') as f_in:
        dic = {}
        for info in f_in:
            info = info.strip()
            info = info.split('\t')
            if (len(info) == 3):
                title = info[0]
                context = info[1]
                category = info[2]
                if title in dic:
                    dic[title + '_' + str(multi)] = [context, category]
                    multi += 1
                else:
                    dic[title] = [context, category]
                    multi = 1
            else:
                print('something leaked\n')
                print(info)
    return dic

def calculate_simi(contexts, cited_txt, tok_size, f_out):
    # initialize models
    bert_pre = pre_process()
    bert = models()

    # get embedding of the cited abs
    tokens_cited = bert_pre.tokenize([cited_txt])
    pre_cited = bert_pre.bert_pack_inputs([tokens_cited], tf.constant(tok_size))
    vec_cited = bert(pre_cited)['sequence_output']

    # get embeddings of the contexts
    f_out.write(f'Title\tContext\tCited_text_abs\tCos_similarity\tCategory')
    for key in contexts:
        context = contexts[key][0]
        cate = contexts[key][1]
        tokens = bert_pre.tokenize([context])
        pre = bert_pre.bert_pack_inputs([tokens], tf.constant(tok_size))
        vec_context = bert(pre)['sequence_output']

        simi = cos_similarity(vec_cited, vec_context)
        f_out.write(f'\n{key}\t{context}\t{cited_txt}\t{simi}\t{cate}')



