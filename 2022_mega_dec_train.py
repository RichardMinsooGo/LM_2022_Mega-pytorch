'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
# !pip install sentencepiece

data_dir = "/content"

! pip list | grep sentencepiece

import sentencepiece as spm

'''
D1. Import Libraries for Data Engineering
'''
import csv
import sys
import os
import math
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

from tqdm import tqdm, tqdm_notebook, trange

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from IPython.display import display

# Setup seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# for using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
D3. [PASS] Tokenizer Install & import
'''
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D4. Define Hyperparameters for Data Engineering
'''
ENCODER_LEN  = 15
DECODER_LEN  = 23
BATCH_SIZE   = 16

'''
D5. Load and modifiy to pandas dataframe
'''
import pandas as pd

pd.set_option('display.max_colwidth', None)

"""
raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),

    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),

    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
    ("I have something very important to tell you.", "Il me faut vous dire quelque chose de très important."),
    ("I have three times as many books as he does.", "J'ai trois fois plus de livres que lui."),
    ("I have to change the batteries in the radio.", "Il faut que je change les piles de cette radio."),
    ("I have to finish up some things before I go.", "Je dois finir deux trois trucs avant d'y aller."),

    ("I have to think about what needs to be done.", "Je dois réfléchir sur ce qu'il faut faire."),
    ("I haven't been back here since the incident.", "Je ne suis pas revenu ici depuis l'accident."),
    ("I haven't eaten anything since this morning.", "Je n'ai rien mangé depuis ce matin."),
    ("I hear his business is on the verge of ruin.", "Apparemment son entreprise est au bord de la faillite."),
    ("I hope I didn't make you feel uncomfortable.", "J'espère que je ne t'ai pas mis mal à l'aise."),
    ("I hope to continue to see more of the world.", "J'espère continuer à voir davantage le monde."),
    ("I hope to see reindeer on my trip to Sweden.", "J'espère voir des rennes lors de mon voyage en Suède."),
    ("I hope you'll find this office satisfactory.", "J'espère que ce bureau vous conviendra."),

    ("I hurried in order to catch the first train.", "Je me dépêchai pour avoir le premier train."),
    ("I just can't stand this hot weather anymore.", "Je ne peux juste plus supporter cette chaleur."),
    ("I just don't want there to be any bloodshed.", "Je ne veux tout simplement pas qu'il y ait une effusion de sang."),
    ("I just thought that you wouldn't want to go.", "J'ai simplement pensé que vous ne voudriez pas y aller."),
    ("I plan to go. I don't care if you do or not.", "Je prévois d'y aller. Ça m'est égal que vous y alliez aussi ou pas."),
    ("I prefer soap as a liquid rather than a bar.", "Je préfère le savon liquide à une savonnette."),
    ("I promise you I'll explain everything later.", "Je vous promets que j'expliquerai tout plus tard."),
    ("I ran as fast as I could to catch the train.", "Je courus aussi vite que je pus pour attraper le train."))


raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"))
"""

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"))

import unicodedata
import re

from tensorflow.keras.preprocessing.text import Tokenizer

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
def preprocess_en(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

def preprocess_fr(sentence):
    # 위에서 구현한 함수를 내부적으로 호출
    sentence = unicode_to_ascii(sentence.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

# 인코딩 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous deja dine?"

print(preprocess_en(en_sent))
print(preprocess_fr(fr_sent).encode('utf-8'))

raw_encoder_input, raw_data_fr = list(zip(*raw_data))
raw_encoder_input, raw_data_fr = list(raw_encoder_input), list(raw_data_fr)

raw_src = [preprocess_en(data) for data in raw_encoder_input]
raw_trg = [preprocess_fr(data) for data in raw_data_fr]

print(raw_src[:4])
print(raw_trg[:4])

'''
D9. Define dataframe
'''
SRC_df = pd.DataFrame(raw_src)
TRG_df = pd.DataFrame(raw_trg)

SRC_df.rename(columns={0: "SRC"}, errors="raise", inplace=True)
TRG_df.rename(columns={0: "TRG"}, errors="raise", inplace=True)
total_df = pd.concat([SRC_df, TRG_df], axis=1)

print('Translation Pair :',len(total_df)) # 리뷰 개수 출력
total_df.sample(3)

raw_src_df  = total_df['SRC']
raw_trg_df  = total_df['TRG']

src_sentence  = raw_src_df
trg_sentence  = raw_trg_df

'''
D10. Define tokenizer
'''

with open('corpus_src.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['SRC']))

with open('corpus_trg.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['TRG']))

# This is the folder to save the data. Modify it to suit your environment.
data_dir = "/content"

corpus = "corpus_src.txt"
prefix = "nmt_src_vocab"
vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

corpus = "corpus_trg.txt"
prefix = "nmt_trg_vocab"

vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

for f in os.listdir("."):
    print(f)

vocab_src_file = f"{data_dir}/nmt_src_vocab.model"
vocab_src = spm.SentencePieceProcessor()
vocab_src.load(vocab_src_file)

vocab_trg_file = f"{data_dir}/nmt_trg_vocab.model"
vocab_trg = spm.SentencePieceProcessor()
vocab_trg.load(vocab_trg_file)

n_enc_vocab = len(vocab_src)
n_dec_vocab = len(vocab_trg)

print('Word set size of Encoder :',n_enc_vocab)
print('Word set size of Decoder :',n_dec_vocab)

'''
Token List
'''
# Recommend : For small number of vocabulary, please test each IDs.
# src_vocab_list
src_vocab_list = [[vocab_src.id_to_piece(id), id] for id in range(vocab_src.get_piece_size())]

# trg_vocab_list
trg_vocab_list = [[vocab_trg.id_to_piece(id), id] for id in range(vocab_trg.get_piece_size())]

'''
D11. Tokenizer test
'''
# Source Tokenizer
lines = [  SRC_df.iloc[1,0],  SRC_df.iloc[2,0],  SRC_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_src.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_src.DecodeIds(txt_2_ids))

    txt_2_tkn = vocab_src.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_src.DecodePieces(txt_2_tkn))

    ids2 = vocab_src.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_src.id_to_piece(ids2))
    print("\n")

print("\n")

# Target Tokenizer
lines = [  TRG_df.iloc[1,0],  TRG_df.iloc[2,0],  TRG_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_trg.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_trg.DecodeIds(txt_2_ids))
    
    txt_2_tkn = vocab_trg.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_trg.DecodePieces(txt_2_tkn))

    ids2 = vocab_trg.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_trg.id_to_piece(ids2))
    print("\n")

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_src  = vocab_src.encode_as_ids(src_sentence.to_list())
tokenized_trg  = vocab_trg.encode_as_ids(trg_sentence.to_list())

# Add [BOS], [EOS] token ids to each target list elements.
new_list = [ x.insert(0, 2) for x in tokenized_trg]
new_list = [ x.insert(len(x), 3) for x in tokenized_trg]

tokenized_inputs  = tokenized_src
tokenized_outputs = tokenized_trg

'''
D13. [EDA] Explore the tokenized datasets
'''

len_result = [len(s) for s in tokenized_inputs]

print('Maximum length of source : {}'.format(np.max(len_result)))
print('Average length of source : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

len_result = [len(s) for s in tokenized_outputs]

print('Maximum length of target : {}'.format(np.max(len_result)))
print('Average length of target : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
tkn_sources = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
tkn_targets = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')

'''
D15. Send data to device
'''

tensors_src   = torch.tensor(tkn_sources).to(device)
tensors_trg   = torch.tensor(tkn_targets).to(device)

'''
D16. [EDA] Explore the Tokenized datasets
'''
print('Size of source language data(shape) :', tkn_sources.shape)
print('Size of target language data(shape) :', tkn_targets.shape)

# Randomly output the 0th sample
print(tkn_sources[0])
print(tkn_targets[0])

'''
D17. [PASS] Split Data
'''

'''
D18. Build dataset
'''

from torch.utils.data import TensorDataset   # 텐서데이터셋
from torch.utils.data import DataLoader      # 데이터로더

dataset    = TensorDataset(tensors_src, tensors_trg)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


'''
D19. [PASS] Define some useful parameters for further use
'''

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''
from tqdm import tqdm, tqdm_notebook, trange

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange
from einops.layers.torch import Rearrange

from scipy.fftpack import next_fast_len

# functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    return val if exists(val) else d

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = rfft(x, n = fast_len, dim = dim)
    f_weight = rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# positional bias for single-headed attention

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# classes

class LaplacianAttnFn(nn.Module):
    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class SingleHeadedAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_qk,
        dim_value,
        causal = False,
        laplacian_attn_fn = False
    ):
        super().__init__()
        self.causal = causal
        self.laplacian_attn_fn = laplacian_attn_fn

        self.attn_fn = partial(F.softmax, dim = -1) if not laplacian_attn_fn else LaplacianAttnFn()

        self.rel_pos_bias = T5RelativePositionBias(causal = causal, scale = dim_qk ** 0.5)

        self.to_qk = nn.Sequential(
            nn.Linear(dim, dim_qk),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(dim_qk, heads = 2)

        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_value),
            nn.SiLU()
        )

    def forward(self, x, v_input = None):
        seq_len, dim, device, dtype = *x.shape[-2:], x.device, x.dtype

        v_input = default(v_input, x)

        qk, v = self.to_qk(x), self.to_v(v_input)
        q, k = self.offsetscale(qk)

        scale = (seq_len ** -1) if self.laplacian_attn_fn else (dim ** -0.5)

        sim = einsum('b i d, b j d -> b i j', q, k) * scale

        sim = sim + self.rel_pos_bias(sim)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)

        if self.causal and not self.laplacian_attn_fn:
            # is softmax attention and using large negative value pre-softmax
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = self.attn_fn(sim)

        if self.causal and self.laplacian_attn_fn:
            # if using laplacian attention function, zero out upper triangular with 0s
            attn = attn.masked_fill(causal_mask, 0.)

        return einsum('b i j, b j d -> b i d', attn, v)

class MultiHeadedEMA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        bidirectional = False,
        norm_mhesa_heads = False
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.expansion = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))
        self.reduction = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))

        # learned alpha and dampening factors

        self.alphas = nn.Parameter(torch.randn(heads))
        self.dampen_factors = nn.Parameter(torch.randn(heads))

        if bidirectional:
            self.reverse_alphas = nn.Parameter(torch.randn(heads))
            self.reverse_dampen_factors = nn.Parameter(torch.randn(heads))

        self.heads = heads

        self.norm_heads = nn.Identity()

        if norm_mhesa_heads:
            # https://arxiv.org/abs/2210.06423 - retnet used sub-ln with some success as groupnorm

            self.norm_heads = nn.Sequential(
                Rearrange('b n h d -> b (h d) n'),
                nn.GroupNorm(heads, dim * heads),
                Rearrange('b (h d) n -> b n h d', h = heads)
            )

    def forward(self, x):
        device, seq_len = x.device, x.shape[1]

        # project in and split heads

        x = einsum('... d, h d -> ... h d', x, self.expansion)

        if self.bidirectional:
            x, x_reversed = x.chunk(2, dim = -2)
            x_reversed = torch.flip(x_reversed, dims = (1,))

        # weights derived from alphas (learned exponential smoothing decay rate)

        def apply_learned_ema_with_damping(x, alphas, dampen_factors):
            alphas = alphas.sigmoid()
            dampen_factors = dampen_factors.sigmoid()

            reversed_powers = torch.arange(seq_len - 1, -1, -1, device = device)
            K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))

            # conv1d fft O(nlog(n))

            return conv1d_fft(x, K, dim = -3, weight_dim = -2)

        x = apply_learned_ema_with_damping(x, self.alphas, self.dampen_factors)

        if self.bidirectional:
            x_reversed = apply_learned_ema_with_damping(x_reversed, self.reverse_alphas, self.reverse_dampen_factors)
            x_reversed = torch.flip(x_reversed, dims = (1,))
            x = torch.cat((x, x_reversed), dim = -2)

        # maybe norm heads

        x = self.norm_heads(x)

        # combine heads and out

        return einsum('... h d, h d -> ... d', x, self.reduction)

# Mega Layer
# Single headed Attention + Multi-headed EMA, then GRU-esque gating

class MegaLayer(nn.Module):
    def __init__(
        self,
        *,
        dim = 128,
        ema_heads = 16,
        attn_dim_qk = 64,
        attn_dim_value = 256,
        laplacian_attn_fn = False,
        causal = True,
        norm_mhesa_heads = False
    ):
        super().__init__()

        self.single_headed_attn = SingleHeadedAttention(
            dim = dim,
            dim_qk = attn_dim_qk,
            dim_value = attn_dim_value,
            causal = causal,
            laplacian_attn_fn = laplacian_attn_fn
        )

        self.multi_headed_ema = MultiHeadedEMA(
            dim = dim,
            heads = ema_heads,
            bidirectional = not causal,
            norm_mhesa_heads = norm_mhesa_heads
        )

        self.to_reset_gate = nn.Sequential(
            nn.Linear(dim, attn_dim_value),
            nn.SiLU()
        )

        self.to_update_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # equation 14, for calculating H

        self.Wh = nn.Parameter(torch.randn(dim, dim))
        self.Uh = nn.Parameter(torch.randn(attn_dim_value, dim))
        self.bh = nn.Parameter(torch.randn(dim))

    def forward(self, x, residual = None):
        residual = default(residual, x)

        ema_output = self.multi_headed_ema(x)
        attn_output = self.single_headed_attn(ema_output, x)

        reset_gate = self.to_reset_gate(ema_output)
        update_gate = self.to_update_gate(ema_output)

        gated_attn_output = attn_output * reset_gate

        # equation 14

        H = F.silu(ema_output @ self.Wh + gated_attn_output @ self.Uh + self.bh)

        # update gate

        return update_gate * H + (1 - update_gate) * residual

# Mega

def FeedForward(dim, ff_mult):
    dim_hidden = int(dim * ff_mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Linear(dim_hidden, dim)
    )

class Mega(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        ff_mult = 2,
        pre_norm = False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pre_norm = pre_norm

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MegaLayer(dim = dim, **kwargs),
                nn.LayerNorm(dim),
                FeedForward(dim = dim, ff_mult = ff_mult),
                nn.LayerNorm(dim)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim) if pre_norm else nn.Identity(),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        pre_norm = self.pre_norm
        post_norm = not self.pre_norm

        x = self.token_emb(x)

        for mega_layer, mega_norm, ff, ff_norm in self.layers:
            mega_maybe_prenorm = mega_norm if pre_norm else identity
            ff_maybe_prenorm = ff_norm if pre_norm else identity

            mega_maybe_postnorm = mega_norm if post_norm else identity
            ff_maybe_postnorm = ff_norm if post_norm else identity

            x = mega_layer(mega_maybe_prenorm(x), x)

            x = mega_maybe_postnorm(x)

            x = ff(ff_maybe_prenorm(x)) + x

            x = ff_maybe_postnorm(x)

        return self.to_logits(x)

import torch
# from mega_pytorch import Mega

model = Mega(
    num_tokens = n_dec_vocab,            # number of tokens
    dim = 128,                   # model dimensions
    depth = 6,                   # depth
    ema_heads = 16,              # number of EMA heads
    attn_dim_qk = 64,            # dimension of queries / keys in attention
    attn_dim_value = 256,        # dimensino of values in attention
    laplacian_attn_fn = True,    # whether to use softmax (false) or laplacian attention activation fn (true)
)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 네트워크 초기화
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# TransformerBlock모듈의 초기화 설정
model.apply(initialize_weights)

import os.path

if os.path.isfile('./checkpoints/2022_mega.pt'):
    model.load_state_dict(torch.load('./checkpoints/2022_mega.pt'))

print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
# learning_rate = 2e-4
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

from IPython.display import clear_output
import datetime

Model_start_time = time.time()

# 학습 정의
def train(epoch, model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    accuracies = []

    with tqdm_notebook(total=len(dataloader), desc=f"Train {epoch+1}") as pbar:
        for batch_idx, samples in enumerate(dataloader):
            src_inputs, trg_outputs = samples

            # print("src_inputs  Shape :", src_inputs.shape)
            # print(src_inputs)
            mask_src = (src_inputs!=0).int()
            # print(mask_src)

            # print("trg_outputs Shape :", trg_outputs.shape)
            # print("trg_outputs :\n", trg_outputs)
            mask_trg = (trg_outputs!=0).int()
            # print(mask_trg)

            Input_concat = torch.concat((src_inputs, trg_outputs),dim=1)
            # print("Input_concat Shape :", Input_concat.shape)
            # print("Input_concat :\n", Input_concat)

            with torch.set_grad_enabled(True):

                # Transformer에 입력
                logits_lm = model(Input_concat)
                # print("logits_lm  Shape :", logits_lm.shape)
                
                pad       = torch.LongTensor(trg_outputs.size(0), 1).fill_(0).to(device)
                preds_id  = torch.transpose(logits_lm,1,2)
                labels_lm = torch.cat((trg_outputs[:, 1:], pad), -1)
                # print("labels_lm Shape: \n",labels_lm.shape)
                # print("labels_lm : \n",labels_lm)

                labels_concat = torch.concat((src_inputs, labels_lm),dim=1)
                # print("labels_concat Shape :", labels_concat.shape)
                # print("labels_concat :\n", labels_concat)
                
                optimizer.zero_grad()
                loss = criterion(preds_id, labels_concat)  # loss 계산

                # Accuracy
                # print("preds_id  : \n",preds_id.shape)
                mask_0 = (labels_concat!=0).int()
                arg_preds_id = torch.argmax(preds_id, axis=1)
                # print("arg_preds : \n",arg_preds_id)
                # print("arg_preds : \n",arg_preds_id.shape)
                # print("mask_0    : \n",mask_0)

                accuracy_1 = torch.eq(labels_concat, arg_preds_id).int()
                # print("accuracy_1 : \n",accuracy_1)

                accuracy_2 = torch.mul(arg_preds_id, accuracy_1).int()
                # print("accuracy_2 : \n",accuracy_2)

                accuracy = torch.count_nonzero(accuracy_2) / torch.count_nonzero(mask_0)
                # print("Accuracy : ",accuracy.clone().detach().cpu().numpy())
                accuracies.append(accuracy.clone().detach().cpu().numpy())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss +=loss.item()

            pbar.update(1)
            # pbar.set_postfix_str(f"Loss {epoch_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            # pbar.set_postfix_str(f"Loss {loss.result():.4f}")
    print("accuracies :", np.mean(accuracies))
    return epoch_loss / len(dataloader)

CLIP = 0.5

epoch_ = []
epoch_train_loss = []
# 네트워크가 어느정도 고정되면 고속화
torch.backends.cudnn.benchmark = True
# epoch 루프
best_epoch_loss = float("inf")

N_EPOCHS = 100

for epoch in range(N_EPOCHS):

    train_loss = train(epoch, model, dataloader, optimizer, criterion, CLIP)

    if train_loss < best_epoch_loss:
        if not os.path.isdir("checkpoints"):
            os.makedirs("checkpoints")
        best_epoch_loss = train_loss
        torch.save(model.state_dict(), './checkpoints/2022_mega.pt')

    epoch_.append(epoch)
    epoch_train_loss.append(train_loss)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    # print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, N_EPOCHS, epoch_loss))
    # clear_output(wait = True)

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(epoch_,epoch_train_loss, label='Average loss')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

plt.show()

# Build evaluation code.

# Predict the trained model
trained_model = Mega(
    num_tokens = n_dec_vocab,            # number of tokens
    dim = 128,                   # model dimensions
    depth = 6,                   # depth
    ema_heads = 16,              # number of EMA heads
    attn_dim_qk = 64,            # dimension of queries / keys in attention
    attn_dim_value = 256,        # dimensino of values in attention
    laplacian_attn_fn = True,    # whether to use softmax (false) or laplacian attention activation fn (true)
).to(device)
trained_model.load_state_dict(torch.load('./checkpoints/2022_mega.pt'))

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(text):
    text = preprocess_sentence(text)
    # print(text)
    text = [vocab_src.encode_as_ids(text)]
    # print(text)
    encoder_input = pad_sequences(text, maxlen=ENCODER_LEN, padding='post', truncating='post')
    # print(encoder_input)

    decoder_input = [2]   #[BOS] token is 2
    # print(decoder_input)
    
    input  = torch.tensor(encoder_input).to(device)
    output = torch.tensor([decoder_input]).to(device)

    # print("input :", input)
    # print("output:", output)

    for i in range(DECODER_LEN):
        concate_input = torch.concat((input, output),dim=1)
        # print("concate_input :", concate_input)
        predictions = trained_model(concate_input)
        # print(predictions)

        predictions = predictions[:, -1:, :]
        # print(predictions)

        # PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions, axis=-1)
        # print(predicted_id)
        if predicted_id== 3:
            break

        output = torch.cat((output, predicted_id),-1)
    return output

def predict(text):
    prediction = evaluate(text)[0].detach().cpu().numpy()
    prediction = prediction[1:]
    # print("Pred IDs :", prediction)

    predicted_sentence = vocab_trg.DecodeIds(prediction.tolist())
    # print(predicted_sentence)
    return predicted_sentence

for idx in (0, 1, 2, 3):
    print("Input        :", src_sentence[idx])
    print("Prediction   :", predict(src_sentence[idx]))
    print("Ground Truth :", trg_sentence[idx],"\n")


'''
M13. [PASS] Explore the training result with test dataset
'''
    
