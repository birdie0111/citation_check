import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text
from transformers import *
import torch
import torch.nn.functional as F

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

def cos_similarity():
    cos_simi = tf.keras.losses.CosineSimilarity(axis=1)
    return cos_simi

def make_input(input_f):
    with open(input_f, 'r', encoding='utf-8') as f_in:
        df = pd.read_csv(f_in, delimiter='\t', encoding='utf-8')
    return df

def padding(vec, size):
    #print(vec.size(dim=1))
    vec = F.pad(vec, (0,size-vec.size(dim=1)), "constant", 0)
    return vec

def get_embedding_bert(text, tok_size, bert_pre, bert):
    tokens = bert_pre.tokenize([text])
    pre = bert_pre.bert_pack_inputs([tokens], tf.constant(tok_size))
    embedding = bert(pre)['sequence_output']
    return embedding

def get_embedding_scibert(text, tok_size, tokenizer, scibert):
    tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    tokens = padding(tokens, tok_size)
    embedding = scibert(tokens)['last_hidden_state']
    return embedding


def calculate_simi(df, compare1, compare2, tok_size, f_out, model):
    # initialize models
    f_out.write(f'{compare1}\t{compare2}\tCos_similarity\tCategory')
    cos_simi = cos_similarity()
    if (model == '1'): # bert
        bert_pre = pre_process()
        bert = models()
        for id in df.index:
            text1 = df[compare1][id]
            text2 = df[compare2][id]
            cate = df['category'][id]

            vec1 = get_embedding_bert(text1, tok_size, bert_pre, bert)
            vec2 = get_embedding_bert(text2, tok_size, bert_pre, bert)

            simi = -cos_simi(vec1, vec2).numpy()
            f_out.write(f'\n{text1}\t{text2}\t{simi}\t{cate}')

    elif(model == '0'): # sci-bert
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        for id in df.index:
            text1 = df[compare1][id]
            text2 = df[compare2][id]
            cate = df['category'][id]

            vec1 = get_embedding_scibert(text1, tok_size, tokenizer, scibert)
            vec2 = get_embedding_scibert(text2, tok_size, tokenizer, scibert)

            vec1 = vec1.detach().numpy()
            vec2 = vec2.detach().numpy()

            simi = -cos_simi(vec1, vec2).numpy()
            f_out.write(f'\n{text1}\t{text2}\t{simi}\t{cate}')
    else:
        print('No model chosen, error\n')
        exit(0)

    

def calculate_simi_all(df, tok_size, f_out, model):
    # initialize models
    cos_simi = cos_similarity()
    f_out.write(f'citing_title\tcited_title\tabs_abs\tcontext_context\tciting_context_cited_abs\tCategory')
    if (model == '1'): # bert
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

            vec_citing_abs = get_embedding_bert(citing_abs, tok_size, bert_pre, bert)
            vec_cited_abs = get_embedding_bert(cited_abs, tok_size, bert_pre, bert)
            vec_citing_context = get_embedding_bert(citing_context, tok_size, bert_pre, bert)
            vec_cited_context = get_embedding_bert(cited_context, tok_size, bert_pre, bert)

            simi_abs_abs = -cos_simi(vec_citing_abs, vec_cited_abs).numpy()
            simi_cc = -cos_simi(vec_citing_context, vec_cited_context).numpy()
            simi_c_abs = -cos_simi(vec_citing_context, vec_cited_abs).numpy()

            f_out.write(f'\n{citing_title}\t{cited_title}\t{simi_abs_abs}\t{simi_cc}\t{simi_c_abs}\t{cate}')

    elif (model == '0'): # scibert
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        for id in df.index:
            citing_abs = df['citing_abs'][id]
            cited_abs = df['cited_abs'][id]
            citing_context = df['citing_context'][id]
            cited_context = df['cited_context'][id]
            cate = df['category'][id]
            citing_title = df['citing_title'][id]
            cited_title = df['cited_title'][id]

            vec_citing_abs = get_embedding_scibert(citing_abs, tok_size, tokenizer, scibert)
            vec_cited_abs = get_embedding_scibert(cited_abs, tok_size, tokenizer, scibert)
            vec_citing_context = get_embedding_scibert(citing_context, tok_size, tokenizer, scibert)
            vec_cited_context = get_embedding_scibert(cited_context, tok_size, tokenizer, scibert)

            simi_abs_abs = -cos_simi(vec_citing_abs.detach().numpy(), vec_cited_abs.detach().numpy()).numpy()
            simi_cc = -cos_simi(vec_citing_context.detach().numpy(), vec_cited_context.detach().numpy()).numpy()
            simi_c_abs = -cos_simi(vec_citing_context.detach().numpy(), vec_cited_abs.detach().numpy()).numpy()

            f_out.write(f'\n{citing_title}\t{cited_title}\t{simi_abs_abs}\t{simi_cc}\t{simi_c_abs}\t{cate}')

    else:
        print('No model chosen, error\n')
        exit(0)


def histplot(file, mode):

    df = pd.read_csv(file, delimiter='\t', encoding='utf-8')
    sb.histplot(data=df, x='Cos_similarity', hue= 'Category',  kde=True)
    plt.title(label = mode)
    plt.show()

def pairplot(file,model):
    df = pd.read_csv(file, delimiter='\t', encoding='utf-8')
    pp = sb.pairplot(df, hue='Category')
    if (model == '1'):
        pp.set(xlim=(0,0.6), ylim = (0,0.6))
    elif (model == '0'):
        pp.set(xlim=(0,1), ylim = (0,1))
    plt.show()

