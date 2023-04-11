import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text

import matplotlib.pyplot as plt
import seaborn as sb

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

def make_input(input_f):

    with open(input_f, 'r', encoding='utf-8') as f_in:
        df = pd.read_csv(f_in, delimiter='\t', encoding='utf-8')
    return df

def get_embedding(text, tok_size, bert_pre, bert):
    tokens = bert_pre.tokenize([text])
    pre = bert_pre.bert_pack_inputs([tokens], tf.constant(tok_size))
    embedding = bert(pre)['sequence_output']
    return embedding


def calculate_simi(df, compare1, compare2, tok_size, f_out):
    # initialize models
    f_out.write(f'{compare1}\t{compare2}\tCos_similarity\tCategory')
    bert_pre = pre_process()
    bert = models()

    for id in df.index:
        text1 = df[compare1][id]
        text2 = df[compare2][id]
        cate = df['category'][id]

        vec1 = get_embedding(text1, tok_size, bert_pre, bert)
        vec2 = get_embedding(text2, tok_size, bert_pre, bert)

        simi = cos_similarity(vec1, vec2)
        f_out.write(f'\n{text1}\t{text2}\t{simi}\t{cate}')

def calculate_simi_all(df, tok_size, f_out):
    # initialize models
    f_out.write(f'citing_title\tcited_title\tabs_abs\tcontext_context\tciting_context_cited_abs\tCategory')
    bert_pre = pre_process()
    bert = models()

    for id in df.index:
        citing_abs = df['citing_abs'][id]
        cited_abs = df['cited_abs'][id]
        citing_context = df['citing_context'][id]
        cited_context = df['cited_context'][id]
        cate = df['category'][id]
        citing_title = df['citing_title'][id]
        cited_title = df['cited_title'][id]

        vec_citing_abs = get_embedding(citing_abs, tok_size, bert_pre, bert)
        vec_cited_abs = get_embedding(cited_abs, tok_size, bert_pre, bert)
        vec_citing_context = get_embedding(citing_context, tok_size, bert_pre, bert)
        vec_cited_context = get_embedding(cited_context, tok_size, bert_pre, bert)

        simi_abs_abs = cos_similarity(vec_citing_abs, vec_cited_abs)
        simi_cc = cos_similarity(vec_citing_context, vec_cited_context)
        simi_c_abs = cos_similarity(vec_citing_context, vec_cited_abs)

        f_out.write(f'\n{citing_title}\t{cited_title}\t{simi_abs_abs}\t{simi_cc}\t{simi_c_abs}\t{cate}')


def histplot(file, mode):

    df = pd.read_csv(file, delimiter='\t', encoding='utf-8')
    sb.histplot(data=df, x='Cos_similarity', hue= 'Category',  kde=True)
    plt.title(label = mode)
    plt.show()

def pairplot(file,mode):
    df = pd.read_csv(file, delimiter='\t', encoding='utf-8')
    pp = sb.pairplot(df, hue='Category')
    #pp.set(xlim=(0,0.6), ylim = (0,0.6))
    plt.show()

