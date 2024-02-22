import re
import os
import gc
import pickle
import random
from selectors import EpollSelector
# from typing_extensions import assert_type
import sklearn
import warnings
import datetime
import numpy as np
import pandas as pd

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
from torch.distributions.normal import Normal

import transformers
import seaborn as sns
import matplotlib.pyplot as plt

# from tokenizers import *
from datetime import date
# from transformers import *
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
# from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer
# from transformers import AutoModel, AutoConfig
from transformers import set_seed
# , AdamW
from transformers import AlbertModel, AlbertConfig, DistilBertModel, DistilBertConfig, BertModel, BertConfig
from transformers import AlbertTokenizerFast
from transformers import DebertaV2Model, DebertaV2Config

# DebertaV3 uses DebertaV2Model
from itertools import product
# from tqdm import tqdm_notebook as tqdm
# This is specific for notebook
# from tqdm.notebook import trange, tqdm
from tqdm import trange, tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold

# WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import csv

from sklearn.metrics import f1_score

# ================ New change for AutoModel
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Maybe Useful 240209
from sentence_transformers import SentenceTransformer
import scipy.stats

from sklearn.cluster import KMeans
from sklearn import metrics

SEED = 3
K = 5

DATA_PATH = '../input/datasets/'

MODEL_PATHS = {
    'bert-base-uncased': '../input/bertconfigs/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/',
    'bert-large-uncased-whole-word-masking-finetuned-squad': '../input/bertconfigs/wwm_uncased_L-24_H-1024_A-16/wwm_uncased_L-24_H-1024_A-16/',
    'albert-large-v2': '../input/albertconfigs/albert-large-v2/albert-large-v2/',
    'albert-base-v2': '../input/albertconfigs/albert-base-v2/albert-base-v2/',
    'distilbert': '../input/albertconfigs/distilbert/distilbert/',
    'deberta-v3-large-old': '../input/debertaconfigs/deberta-v3-large-old/',
    'deberta-v3-large-auto': None
    # 'deberta-v3-large-auto': '../input/AutoModelconfigs/deberta-v3-large-auto/'
    }


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_weights(model, filename, verbose=1, cp_folder=""):
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    
    try:
        # Seems that strict wasn't defined if run in standalone
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        # But the keys in the pikle doesn't match the latest transformer version
        print('Loading weights to cpu using torch.load instead of strict')
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )
    return model


def trim_tensors(tokens, input_ids, model_name='bert', min_len=10):
    pad_token = 1 if "roberta" in model_name else 0
    # torch.sum((tokens != pad_token), 1) actually sums over the int
    # see testingcenter for details
    # So this actually find the longest sequence len, except the padded token.
    max_len = max(torch.max(torch.sum((tokens != pad_token), 1)), min_len)
    # Then trim the tokens to the max_len
    return tokens[:, :max_len], input_ids[:, :max_len]


import os
import sys


# ///////////////////// Deal with dataframes with pandas to prepare the training list
def process_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)
    return text


def clean_spaces4text(txt):
    # feature text doesn't need to count the offset
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    txt = re.sub('\s+$', '', txt)
    return txt


def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    # replace more than one \s+ will cause location error
    # but not replacin \s+ will cause offset including precursor spaces
#     txt = re.sub(r'\s+', ' ', txt)
    # just for test round 3, 2

    # txt = txt.replace('\u2581',' ')
    # txt = txt.replace('\u0020',' ')
    # txt = txt.replace('\u00A0',' ')
    # txt = txt.replace('\u1680',' ')
    # txt = txt.replace('\u180E',' ')
    # txt = txt.replace('\u2000',' ')
    # txt = txt.replace('\u2001',' ')
    # txt = txt.replace('\u2002',' ')
    # txt = txt.replace('\u2003',' ')
    # txt = txt.replace('\u2004',' ')
    # txt = txt.replace('\u2005',' ')
    # txt = txt.replace('\u2006',' ')
    # txt = txt.replace('\u2007',' ')
    # txt = txt.replace('\u2008',' ')
    # txt = txt.replace('\u2009',' ')
    # txt = txt.replace('\u200A',' ')
    # txt = txt.replace('\u200B',' ')
    # txt = txt.replace('\u202F',' ')
    # txt = txt.replace('\u205F',' ')
    # txt = txt.replace('\u3000',' ')
    # txt = txt.replace('\uFEFF',' ')
    # txt = txt.replace('\u2423',' ')
    # txt = txt.replace('\u2422',' ')
    # txt = txt.replace('\u2420',' ')

    # looks that the irregular character appears there
    # txt = re.sub('[\u0080-\u0090]', '_', txt)

    # txt = re.sub('\u0091', '\'', txt)
    # txt = re.sub('\u0092', '\'', txt)
    # txt = re.sub('\u0093', '\"', txt)
    # txt = re.sub('\u0094', '\"', txt)
    # txt = re.sub('\u0095', '.', txt)
    # txt = re.sub('\u0096', '-', txt)
    # txt = re.sub('\u0097', '-', txt)
    # txt = re.sub('\u0098', '~', txt)
    # txt = re.sub('\u0099', '#', txt)
    # txt = re.sub('\u009a', 'S', txt)
    # txt = re.sub('\u009b', '>', txt)
    # txt = re.sub('\u009c', '.', txt)
    # txt = re.sub('\u009d', '_', txt)
    # txt = re.sub('\u009e', '_', txt)
    # txt = re.sub('\u009f', '_', txt)

    # # looks that the irregular character appears there, YES, some character appeared here
    # txt = re.sub('[\u00A0-\u00BF]', '_', txt)
    # txt = re.sub('[\u00A0-\u00A7]', '_', txt)
    # txt = re.sub('[\u00A8-\u00AF]', '_', txt)
    # txt = re.sub('[\u00B0-\u00B7]', '_', txt)
    txt = re.sub('\u00B0', ' ', txt) # degree sign
    
    # txt = re.sub('\u00C0|\u00C1|\u00C2|\u00C3|\u00C4|\u00C5|\u00C6', 'A', txt)
    # txt = re.sub('\u00C7', 'C', txt)
    # txt = re.sub('\u00C8|\u00C9|\u00CA|\u00CB', 'E', txt)
    # txt = re.sub('\u00CC|\u00CD|\u00CE|\u00CF', 'I', txt)
    # txt = re.sub('\u00D0', 'T', txt)
    # txt = re.sub('\u00D1', 'N', txt)
    # txt = re.sub('\u00D2|\u00D3|\u00D4|\u00D5|\u00D6', 'O', txt)
    # txt = re.sub('\u00D7', '*', txt)
    # txt = re.sub('\u00D8', '0', txt)
    # txt = re.sub('\u00D9|\u00DA|\u00DB|\u00DC', 'U', txt)
    # txt = re.sub('\u00DD', 'Y', txt)
    # txt = re.sub('\u00DE', 'S', txt)
    # txt = re.sub('\u00DF', 's', txt)
    # txt = re.sub('\u00E0|\u00E1|\u00E2|\u00E3|\u00E4|\u00E5|\u00E6', 'a', txt)
    # txt = re.sub('\u00E7', 'c', txt)
    # txt = re.sub('\u00E8|\u00E9|\u00EA|\u00EB', 'e', txt)
    # txt = re.sub('\u00EC|\u00ED|\u00EE|\u00EF', 'i', txt)
    # txt = re.sub('\u00F0', 't', txt)
    # txt = re.sub('\u00F1', 'n', txt)
    # txt = re.sub('\u00F2|\u00F3|\u00F4|\u00F5|\u00F6', 'o', txt)
    # txt = re.sub('\u00F7', '/', txt)
    # txt = re.sub('\u00F8', '0', txt)
    # txt = re.sub('\u00F9|\u00FA|\u00FB|\u00FC', 'u', txt)
    # txt = re.sub('\u00FD', 'y', txt)
    # txt = re.sub('\u00FE', 's', txt)
    # txt = re.sub('\u00FF', 'y', txt)
    
    # just for test round 3, 2
    # Improvement VAD 1: there are spaces in the new annotated dataset
    # txt = re.sub('\s+$', '', txt)
    # if txt.find('  ') != -1:
    #     print('Error: double space')
    return txt


def process_train_location(txt):
    if txt == '':
        return ''
    else:
        return txt

def process_train_location2text(text, txtloc):
    if txtloc.find(':') == -1:
        txtloc = '0:0'
    txtloc_s, txtloc_e = txtloc.split(':')
    itxtloc_s = int(txtloc_s)
    itxtloc_e = int(txtloc_e)
    return text[itxtloc_s:itxtloc_e]

def stance2cate(text):
    text2cate = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    return text2cate[text]

def load_and_prepare_train(root=""):
    df = pd.read_csv(root + "VAD_train_pairwise.csv")
    
    # left join with two tables
    # df = df.merge(features, how="left", on=["case_num", "feature_num"])
    # df = df.merge(patient_notes, how="left", on=['case_num', 'pn_num'])

    # need improvement
    # df['pn_history'] = df['pn_history'].apply(lambda x: x.strip())
    # df['feature_text'] = df['feature_text'].apply(process_feature_text)
    # df['feature_text'] = df['feature_text'].apply(clean_spaces4featuretext)
    # df['clean_text'] = df['pn_history'].apply(clean_spaces)

    df['lefttext'] = df['lefttext'].apply(process_text)
    df['righttext'] = df['righttext'].apply(process_text)
    df['notorsame'] = df['notorsame'].apply(lambda x: int(x))
    df['stance'] = df['stance'].apply(clean_spaces4text)
    df['stance_cate'] = df['stance'].apply(stance2cate)
    # df['feature_text'] = df['feature_text'].apply(clean_spaces4text)
    # df['clean_text'] = df['text']
    # df['clean_text'] = df['text'].apply(clean_spaces)
    df['clean_lefttext'] = df['lefttext'].apply(clean_spaces)
    df['clean_righttext'] = df['righttext'].apply(clean_spaces)

    # comment our for testing in main
    df['aspect_span'] = df['aspect_span'].apply(process_train_location)
    df['aspect_span_text'] = df.apply(lambda row : process_train_location2text(row['clean_lefttext'],
                                      row['aspect_span']), axis=1)
    
    df['left_aspect_category'] = df['left_aspect_category'].apply(lambda x: int(x))
    df['right_aspect_category'] = df['right_aspect_category'].apply(lambda x: int(x))
    # print(df['aspect_span_text'].head)
    # df['target'] = ""
    return df

# ///////////////////// Load and prepare test
def load_and_prepare_test(root=""):
    df = pd.read_csv(root + 'VAD_test_pairwise.csv')

    # df['text'] = df['text'].apply(process_text)
    df['lefttext'] = df['lefttext'].apply(process_text)
    df['righttext'] = df['righttext'].apply(process_text)
    df['notorsame'] = df['notorsame'].apply(lambda x: int(x))

    df['stance'] = df['stance'].apply(clean_spaces4text)
    df['stance_cate'] = df['stance'].apply(stance2cate)
    # df['feature_text'] = df['feature_text'].apply(clean_spaces4text)
    # df['clean_text'] = df['text']
    df['clean_lefttext'] = df['lefttext'].apply(clean_spaces)
    df['clean_righttext'] = df['righttext'].apply(clean_spaces)

    # comment our for testing in main
    df['aspect_span'] = df['aspect_span'].apply(process_train_location)
    df['aspect_span_text'] = df.apply(lambda row : process_train_location2text(row['clean_text'],
                                      row['aspect_span']), axis=1)

    df['left_aspect_category'] = df['left_aspect_category'].apply(lambda x: int(x))
    df['right_aspect_category'] = df['right_aspect_category'].apply(lambda x: int(x))

    # print(df['aspect_span_text'].head)
    # df['target'] = ""
    return df

def load_and_prepare_test_left_uniqued(root=""):
    df = pd.read_csv(root + 'VAD_test_pairwise.csv')

    # df = df['lefttext'].unique(return_index=True)
    uidces = df.reset_index().groupby(['ID'])['index'].min().to_list()
    print('len(uidces) = -----------------------', len(uidces))
    df = df.iloc[uidces]

    # df['text'] = df['text'].apply(process_text)
    df['lefttext'] = df['lefttext'].apply(process_text)
    df['righttext'] = df['righttext'].apply(process_text)
    df['notorsame'] = df['notorsame'].apply(lambda x: int(x))

    df['stance'] = df['stance'].apply(clean_spaces4text)
    df['stance_cate'] = df['stance'].apply(stance2cate)
    # df['feature_text'] = df['feature_text'].apply(clean_spaces4text)
    # df['clean_text'] = df['text']
    df['clean_lefttext'] = df['lefttext'].apply(clean_spaces)
    df['clean_righttext'] = df['righttext'].apply(clean_spaces)

    # comment our for testing in main
    df['aspect_span'] = df['aspect_span'].apply(process_train_location)
    df['aspect_span_text'] = df.apply(lambda row : process_train_location2text(row['clean_lefttext'],
                                      row['aspect_span']), axis=1)

    df['left_aspect_category'] = df['left_aspect_category'].apply(lambda x: int(x))
    df['right_aspect_category'] = df['right_aspect_category'].apply(lambda x: int(x))

    # print(df['aspect_span_text'].head)
    # df['target'] = ""
    return df

# ///////////////////// PerText Encoder Decoder
import sentencepiece as spm

# This actually add the module to the path before execution.
# It is similar to import .input.sentencepiece_pb2 if sentencepiece_pb2 is placed within the same directory of this .py (entry.py)
# sys.path.insert(0, "../input/sentencepiece-pb2")
# Important: no '/' in the tail please!
sys.path.insert(0, "../input/sentencepiece-pb2")
# when in VScode use this. when in notebook use the previous one
# sys.path.insert(0, "./input/sentencepiece-pb2")
# Have already changed the vscode setting using the directory of the executed .py file, see book mark 220202
import sentencepiece_pb2
# See google colab sentence piece
# https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb#scrollTo=SUcAbKnRVAv6
# See
# !pip install protobuf
# !wget https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_pb2.py
# in the same page
# Fine, by searching through google, finally find its location https://github.com/google/sentencepiece/tree/master/python/src/sentencepiece

# About the version problem Missing key(s) in state_dict: "transformer.embeddings.position_ids".,
# see https://github.com/huggingface/transformers/issues/6882


class EncodedText:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets

        
class SentencePieceTokenizer:
    def __init__(self, model_path, lowercase=True):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path))
        self.lowercase = lowercase
    
    def encode(self, sentence):
        if self.lowercase:
            sentence = sentence.lower()
            
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            # The tokenizer has provided the begin and end tokens
            offsets.append((piece.begin, piece.end))
        return EncodedText(tokens, offsets)

    def decode_ids(self, ids_list):
        '''
        tokenize only used for testing, attributes to google sentence piece colab
        '''  
        decoded_from_ids = self.sp.decode_ids(ids_list)
        return decoded_from_ids


TRANSFORMERS = {
    'albert-base-v2': (AlbertModel, 'albert/albert-base-v2', AlbertConfig),
    'albert-large-v2': (AlbertModel, 'albert/albert-large-v2', AlbertConfig),
    'albert-xxlarge-v2': (AlbertModel, 'albert/albert-xxlarge-v2', AlbertConfig),
    "bert-base-uncased": (BertModel, "bert-base-uncased", BertConfig),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (BertModel, "bert-large-uncased-whole-word-masking-finetuned-squad", BertConfig),
    "distilbert": (DistilBertModel, "distilbert-base-uncased-distilled-squad", DistilBertConfig),
    'deberta-v3-large-old': (DebertaV2Model, "microsoft/deberta-v3-large", DebertaV2Config),
    # =============
    'deberta-v3-large-auto': (None, "microsoft/deberta-v3-large", None)
}


def create_tokenizer_and_tokens(df, config):
    '''
    equivalent to precompute_tokens in mRSB
    '''
    uu_model_class, str_model_offical_name, uu_config_class = TRANSFORMERS[config.selected_model]

    if config.is_automodel is True:
        print('AutoTokenizer always use online pre-trained tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(str_model_offical_name)
        print(str_model_offical_name)
        print('Interesting, Just a test')
        tokenizer.save_pretrained(f'../input/autotokenizers/{config.selected_model}')
        print(f'The tokenizer for {str_model_offical_name} is', tokenizer)
        special_tokens = {
            "sep": tokenizer.sep_token_id,
            "cls": tokenizer.cls_token_id,
            "pad": tokenizer.pad_token_id,
        }
    elif config.is_automodel is False:
        print('Tokenizer always load weights from online pre-trained tokenizer, except for deberta-v3-large-old, which load weights from local pre-trained tokenizer')
        print('TokenizerClass is hard-coded in create_tokenizer_and_tokens function, unlike ModelClass in DOCTransformer')

        if 'albert-xxlarge-v2' in config.selected_model:
            tokenizer = AlbertTokenizerFast.from_pretrained(str_model_offical_name)
            tokenizer.save_pretrained('../input/albertconfigs/albert-xxlarge-v2/tokenizer')
            special_tokens = {
                "sep": tokenizer.sep_token_id,
                "cls": tokenizer.cls_token_id,
                "pad": tokenizer.pad_token_id,
            }
        elif 'albert-large-v2' in config.selected_model:
            tokenizer = AlbertTokenizerFast.from_pretrained('albert-large-v2')
            tokenizer.save_pretrained('../input/albertconfigs/albert-large-v2/tokenizer')
            special_tokens = {
                "sep": tokenizer.sep_token_id,
                "cls": tokenizer.cls_token_id,
                "pad": tokenizer.pad_token_id,
            }
        elif "albert" in config.selected_model:
            sys.exit('tokenizer undefined, code under construction, it\'s an depreciated branch')
            # They used a stand_alone tokenizer in albert
            # tokenizer = SentencePieceTokenizer(f'{MODEL_PATHS[config.selected_model]}/{config.selected_model}-spiece.model')
            tokenizer = SentencePieceTokenizer(f'{MODEL_PATHS["albert-large-v2"]}/albert-large-v2-spiece.model')
            special_tokens = {
                'cls': 2,
                'sep': 3,
                'pad': 0,
            }
        elif 'deberta-v3-large-old' in config.selected_model:
            tokenizer = DebertaV2TokenizerFast.from_pretrained('../input/debertaconfigs/deberta-v2-xxlarge/get-token/tokenizer')
            special_tokens = {
                "sep": tokenizer.sep_token_id,
                "cls": tokenizer.cls_token_id,
                "pad": tokenizer.pad_token_id,
            }
            print(special_tokens["pad"])
            print(special_tokens["cls"])
            print(special_tokens["sep"])
        else:
            sys.exit('tokenizer undefined, code under construction, it\'s an depreciated branch')
            # Other models use BertWordPieceTokenizer
            # This is from Tokenizers package, also produced by the Huggingface team.
            # Which also uses .encode method, which is compatible for the bert model

            # BertWordPieceTokenizer in the tokenizers can do the same thing
            tokenizer = BertWordPieceTokenizer(
                MODEL_PATHS[config.selected_model] + 'vocab.txt',
                lowercase=config.lowercase,
    #             add_special_tokens=False  # This doesn't work smh
            )

            special_tokens = {
                'cls': tokenizer.token_to_id('[CLS]'),
                'sep': tokenizer.token_to_id('[SEP]'),
                'pad': tokenizer.token_to_id('[PAD]'),
            }

    feature_texts = df["stance"].unique()
    ids = {}
    offsets = {}
    sEmbeds = {}

    sbertModel = SentenceTransformer('all-mpnet-base-v2')

    if 'albert-xxlarge-v2' in config.selected_model:
        # need improvement 6: maybe we need to store the pre-computed tokenization like reberta so the transformer version difference won't affect the performance 
        longest_feature_ids_len_for_display = 0
        for feature_text in feature_texts:
            encoding = tokenizer(
                feature_text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False)
            ids[feature_text] = encoding["input_ids"]
            offsets[feature_text] = encoding["offset_mapping"]
            if longest_feature_ids_len_for_display < len(encoding["input_ids"]):
                longest_feature_ids_len_for_display = len(encoding["input_ids"])
        # I have given up the inspection on the length and the nan, may be due to the clean_text pre-processor
        # But the inconsistency between longest input_text_ids and assert have to be figured out.
        # Then we will change the network architecture and add the stacking
        print(f"longest_feature_ids_len_for_display = {longest_feature_ids_len_for_display}")
        
        longest_input_ids_text_len_for_display = 0
        texts = df["clean_text"].unique()
        for text in texts:
            encoding = tokenizer(
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False)
            # assert text not in ids
            ids[text] = encoding["input_ids"]
            offsets[text] = encoding["offset_mapping"]
            if longest_input_ids_text_len_for_display < len(encoding["input_ids"]):
                longest_input_ids_text_len_for_display = len(encoding["input_ids"])
        print(f"longest_input_ids_text_len_for_display = {longest_input_ids_text_len_for_display}")
    elif 'albert-large-v2' in config.selected_model:
        # need improvement 6: maybe we need to store the pre-computed tokenization like reberta so the transformer version difference won't affect the performance 
        longest_feature_ids_len_for_display = 0
        for feature_text in feature_texts:
            encoding = tokenizer(
                feature_text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False)
            ids[feature_text] = encoding["input_ids"]
            offsets[feature_text] = encoding["offset_mapping"]
            if longest_feature_ids_len_for_display < len(encoding["input_ids"]):
                longest_feature_ids_len_for_display = len(encoding["input_ids"])
        # I have given up the inspection on the length and the nan, may be due to the clean_text pre-processor
        # But the inconsistency between longest input_text_ids and assert have to be figured out.
        # Then we will change the network architecture and add the stacking
        print(f"longest_feature_ids_len_for_display = {longest_feature_ids_len_for_display}")
        
        longest_input_ids_text_len_for_display = 0
        texts = df["clean_text"].unique()
        for text in texts:
            encoding = tokenizer(
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False)
            # assert text not in ids
            ids[text] = encoding["input_ids"]
            offsets[text] = encoding["offset_mapping"]
            if longest_input_ids_text_len_for_display < len(encoding["input_ids"]):
                longest_input_ids_text_len_for_display = len(encoding["input_ids"])
        print(f"longest_input_ids_text_len_for_display = {longest_input_ids_text_len_for_display}")

    elif "albert" in config.selected_model:
        longest_input_ids_text_len_for_display = 0
        texts = df["clean_text"].unique()
        for text in texts:
            tokenized = tokenizer.encode(text)
            ids[text] = tokenized.ids
            offsets[text] = tokenized.offsets
            if longest_input_ids_text_len_for_display < len(tokenized.ids):
                longest_input_ids_text_len_for_display = len(tokenized.ids)
        print(f"longest_input_ids_text_len_for_display = {longest_input_ids_text_len_for_display}")
    elif config.selected_model == 'deberta-v3-large-auto':
        # for auto, I use == for easy comprehension
        texts = pd.concat([df["clean_lefttext"], df["clean_righttext"]], ignore_index=True)
        texts = texts.unique()

        longest_input_ids_text_len_for_display = 0
        for text in texts:
            encoding = tokenizer(
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False)
            assert text not in ids
            ids[text] = encoding["input_ids"]
            offsets[text] = encoding["offset_mapping"]

            if longest_input_ids_text_len_for_display < len(encoding["input_ids"]):
                longest_input_ids_text_len_for_display = len(encoding["input_ids"])
            sEmbeds[text] = sbertModel.encode([text])[0]
        print(f"longest_input_ids_text_len_for_display = {longest_input_ids_text_len_for_display}")
    elif 'deberta-v3-large-old' in config.selected_model:
        texts = pd.concat([df["clean_lefttext"], df["clean_righttext"]], ignore_index=True)
        texts = texts.unique()
        # list(.unique()) + list(.unique())

        longest_input_ids_text_len_for_display = 0
        for text in texts:
            encoding = tokenizer(
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False)
            assert text not in ids
            ids[text] = encoding["input_ids"]
            offsets[text] = encoding["offset_mapping"]

            if longest_input_ids_text_len_for_display < len(encoding["input_ids"]):
                longest_input_ids_text_len_for_display = len(encoding["input_ids"])
            # 2022/08/08
            sEmbeds[text] = torch.zeros(config.nb_sbert)
        print(f"longest_input_ids_text_len_for_display = {longest_input_ids_text_len_for_display}")
    else:
        sys.exit('config.selected_model unexpected!!!')
        texts = df["clean_text"].unique()
        for text in texts:
            encoding = tokenizer(
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            ids[text] = encoding["input_ids"]
            offsets[text] = encoding["offset_mapping"]
    # 2022/08/08
    precomputed_tokens_and_offsets = {"ids": ids, "offsets": offsets, "sEmbeds": sEmbeds}
    # So tokens are special token ids including 'positive', 'negative' and 'neutral''s encoded token ids (for the three they are ids)
    # Looks that autotokenizer can use .tokenizer.precomputed = precompute_tokens(df, tokenizer)
    # But I am still using this version
    return tokenizer, special_tokens, precomputed_tokens_and_offsets

# This is the original one in the albert.py
# equivalent to encodings_from_precomputed
# Processn test data for prediction. For the benefit of evaluation, we still use process_training_data
def process_test_data(clean_text, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=100, model_name=None):
    # clean_text, sentiment, tokenizer, tokens, max_len=100, model_name="bert", use_old_sentiment=False):
    # text = " " + " ".join(str(text).split())
    text = clean_text

    if 'albert-large-v2' in model_name:
        # tokenized = tokenizer.encode(text)
        # input_ids_text = tokenized.ids
        # text_offsets = tokenized.offsets
        encoding = tokenizer(
            text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False)
        input_ids_text = encoding["input_ids"]
        text_offsets = encoding["offset_mapping"]
    elif 'deberta-v3' in model_name:
        # need improvement 6: maybe we need to store the pre-computed tokenization like reberta so the transformer version difference won't affect the performance 
        encoding = tokenizer(
            text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False)
        input_ids_text = encoding["input_ids"]
        text_offsets = encoding["offset_mapping"]
        assert np.max(text_offsets) == len(text)
    else:
        assert False

    # # just for test round 2, 3.3, triggered an error there when text is ending with spaces， after fixing up clean_spaces by deleting tail spaces, the error still exists
    # # assert np.max(text_offsets) == len(text)
    # assert not text.endswith(' ')
    # assert not text.endswith('\t')
    # assert not text.endswith('\n')
    # assert not text.endswith('\f')
    # assert not text.endswith('\r')
    # just for test round 2, 3.3

    if input_ids_text[0] == special_tokens["cls"]:
        sys.exit('!!!!!Error: special_tokens are not supposed to appear')
        # getting rid of special tokens
        # if cls is in the sentence, means sep is in the entence as well,
        #  This sep is different from the "adding an extra SEP after tokenization"
        input_ids_text = input_ids_text[1:-1]
        text_offsets = text_offsets[1:-1]

    # # Pre-computed text needs trim of head and tail token
    # if stance_as_feature is not None:
    #     # print('Is not none')
    #     feature_text_ids = precomputed_tokens_and_offsets["ids"][stance]
    #     feature_text_offsets = precomputed_tokens_and_offsets["offsets"][stance]
    #     if feature_text_ids[0] == special_tokens["cls"]:
    #         feature_text_ids = feature_text_ids[1:-1]
    #         feature_text_offsets = feature_text_offsets[1:-1]
    # else:
    feature_text_ids = []
    # feature_text_offsets = []
    
    new_max_len = max_len - 3 - len(feature_text_ids)
    # So the max_len is the global max_len, if new_max_len is smaller than the input_ids_text len, this means that the input_ids_text will be trimmed so information will lost partially. This must keeps align with training. Otherwise will cause assertion stop. Actually this can differ within a small margin, but here I just require new_max_len to be equal or larger.

    # just for test round 4, 1
    assert new_max_len >= len(input_ids_text)
    # just for test round 4, 1
    if new_max_len < len(input_ids_text):
        # print('Warning: new_max_len %d > input_ids_text len %d' % (new_max_len, input_ids_text))
        print('Warning: new_max_len %d < input_ids_text len %d' % (new_max_len, len(input_ids_text)))
    
    input_ids = (
        [special_tokens["cls"]] + feature_text_ids + [special_tokens["sep"]]
            + input_ids_text[:new_max_len]
            + [special_tokens["sep"]]
    )

    token_type_ids = [0] + [0] * len(feature_text_ids) + [0] + [1] * (len(input_ids_text[:new_max_len]) + 1)
    text_offsets = [(0, 0)] * (2 + len(feature_text_ids)) + text_offsets[:new_max_len] + [(0, 0)]

    assert len(input_ids) == len(token_type_ids) 
    assert len(input_ids) == len(text_offsets), (len(input_ids), len(text_offsets))

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([special_tokens["pad"]] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)
    
    # Just for test round 2, 3.4
    if np.max(text_offsets) > len(text):
        print('Warning: np.max(text_offsets) %d > len(text) len %d' % (np.max(text_offsets), len(text)))
    # Just for test round 2, 3.4

    return {
        "ids": input_ids,
        "token_type_ids": token_type_ids,
        "text": text,
        "offsets": text_offsets,
    }


class TweetTestDataset(Dataset):
    def __init__(self, df, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=200, model_name="bert"):
        self.special_tokens = special_tokens
        self.precomputed_tokens_and_offsets = precomputed_tokens_and_offsets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

        self.text_external_ids = df['ID'].values
        self.clean_texts = df['clean_text'].values
        # self.feature_texts = df['feature_text'].values

    def __len__(self):
        return len(self.clean_texts)

    def __getitem__(self, idx):
        # Obtain the pre-processed data, for each instance. Data is a dictionary
        # This method wrap arrays or lists into tensor
        # Unlike me. My Dataset class wrap arrays or lists into tensor once for all.
        # So texts from df['text']
        #  sentiments from df['sentiment']
        # tokens are special token ids including 'positive', 'negative' and 'neutral''s encoded token ids (for the three they are ids)
        data = process_test_data(self.clean_texts[idx], self.tokenizer, self.special_tokens, self.precomputed_tokens_and_offsets,
                                 max_len=self.max_len, model_name=self.model_name)
        return {
            'external_id': self.text_external_ids[idx],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'text': data["text"],
            'offsets': np.array(data["offsets"], dtype=np.int_)
        }


# We can check this with test to see if the offsets align with the testing set.
def process_training_data(clean_lefttext, clean_righttext, aspect_span, aspect_span_text, stance, left_aspect_cate, right_aspect_cate, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=100, model_name=None, stance_as_feature=None):
    # text, that is text_1 in inference_roberta_hiki.py, the same as process_test_data in this file.
    # Model_name is not used in both testing_data and training_data
    # replace any white space with space, also note that '  ' will also be kept so no change to offset
    # text = " ".join(clean_text.split())
    # be aware that clean_text.split() <=> re.split('\s+', clean_text)
    # text = " ".join(clean_text.split(' '))
    intra_weight_dct = {
        0: 16.774555297757157,
        1: 24.437725928360173,
        # 2: 43379.000000000015,
        2: 433.0,
        3: 11.528451203307252,
        4: 11.097921330348578,
        # 5: 2344.8108108108113,
        5: 234.0,
        # 6: 2478.8,
        6: 247.0,
        # 7: 547.9452631578948
        7: 147.0}
    instance_weight = 1.0
    if left_aspect_cate == right_aspect_cate:
        instance_weight = intra_weight_dct[left_aspect_cate] 

    lefttext = clean_lefttext
    righttext = clean_righttext

    # # shifted_id selected_text from text
    # idx = text.find(selected_text)

    # chars is 0 except for those chars in location
    chars = np.zeros((len(lefttext)))
    idx = lefttext.find(aspect_span_text)
    chars[idx:idx + len(aspect_span_text)] = 1
    
    if aspect_span.find(':') == -1:
        aspect_span = '0:0'
    s_aspect, e_aspect = aspect_span.split(':')
    i_s_aspect = int(s_aspect)
    i_e_aspect = int(e_aspect)
    assert i_s_aspect == idx
    assert i_e_aspect == idx + len(aspect_span_text)

    # if tokenizer.name:
    if 'albert' in model_name:
        # tokenized = tokenizer.encode(text)
        # # not same as doing self.convert_tokens_to_ids(self.tokenize(text)) since it's customized
        # # print('tokenized_text')
        # # print(tokenizer.tokenize(text))
        # input_ids_text = tokenized.ids
        # # The dataset should be removed of the full space in advance.
        # # if len(input_ids_text) == 0:
        # #     print('danger!', text)
        # #     print(len(text))
        # text_offsets = tokenized.offsets
        left_input_ids_text = precomputed_tokens_and_offsets['ids'][lefttext]
        left_text_offsets = precomputed_tokens_and_offsets['offsets'][lefttext]
        right_input_ids_text = precomputed_tokens_and_offsets['ids'][righttext]
        right_text_offsets = precomputed_tokens_and_offsets['offsets'][righttext]
    elif 'deberta-v3' in model_name:
        # # need improvement 6: maybe we need to store the pre-computed tokenization like reberta so the transformer version difference won't affect the performance 
        # encoding = tokenizer(
        #     text,
        #     return_token_type_ids=True,
        #     return_offsets_mapping=True,
        #     return_attention_mask=False,
        #     add_special_tokens=False)
        # input_ids_text = encoding["input_ids"]
        # text_offsets = encoding["offset_mapping"]
        left_input_ids_text = precomputed_tokens_and_offsets['ids'][lefttext]
        left_text_offsets = precomputed_tokens_and_offsets['offsets'][lefttext]
        right_input_ids_text = precomputed_tokens_and_offsets['ids'][righttext]
        right_text_offsets = precomputed_tokens_and_offsets['offsets'][righttext]
        sEmbeds = precomputed_tokens_and_offsets['sEmbeds'][lefttext]
        if np.max(left_text_offsets) != len(lefttext):
            if lefttext.endswith(' '):
                len_text = len(lefttext)
                i_last_non_space = 0
                for e in range(len_text - 1, -1, -1):
                    if lefttext[e] != ' ':
                        i_last_non_space = e
                        break
                assert np.max(left_text_offsets) == i_last_non_space + 1
                # assert np.max(text_offsets) == len(text)
            else:
                print(f"-->{lefttext}<--", file=sys.stderr)
                print(left_text_offsets, file=sys.stderr)
                print(np.max(left_text_offsets), len(lefttext), file=sys.stderr)
                sys.exit("!!!!!!np.max(left_text_offsets) != len(lefttext) and the sentence doesnt ends with space")
        if np.max(right_text_offsets) != len(righttext):
            if righttext.endswith(' '):
                len_text = len(righttext)
                i_last_non_space = 0
                for e in range(len_text - 1, -1, -1):
                    if righttext[e] != ' ':
                        i_last_non_space = e
                        break
                assert np.max(right_text_offsets) == i_last_non_space + 1
                # assert np.max(text_offsets) == len(text)
            else:
                print(f"-->{righttext}<--", file=sys.stderr)
                print(right_text_offsets, file=sys.stderr)
                print(np.max(right_text_offsets), len(righttext), file=sys.stderr)
                sys.exit("!!!!!!np.max(right_text_offsets) != len(righttext) and the sentence doesnt ends with space")

        # assert np.max(left_text_offsets) == len(lefttext)
        # assert np.max(right_text_offsets) == len(righttext)

        # This is just a test for the maximum text len.
        # assert len(left_input_ids_text) <= 309
        # if len(right_input_ids_text) > 309:
        #     print(righttext)
        # assert len(right_input_ids_text) <= 309
    else:
        assert False

    # # ID_OFFSETS this one equivalent to text_offsets = tokenized.offsets, and do not further process of deleting extra SEP
    # tweet_offsets = []
    # idx = 0
    # # not know if ['cls'] would be 3, seems that 'cls' is still counted.
    # # If both counted, then it will be OK
    # for t in enc.ids:
    #     w = tokenizer.decode([t])
    #     tweet_offsets.append((idx, idx + len(w)))
    #     idx += len(w)

    # Still need to get rid of special tokens. They will be manually added later.
    #  So both token ids and offsets are removed, and then added later
    if left_input_ids_text[0] == special_tokens["cls"]:
        sys.exit('!!!!!Error: special_tokens are not supposed to appear')
        # getting rid of special tokens
        # if cls is in the sentence, means sep is in the entence as well,
        #  This sep is different from the "adding an extra SEP after tokenization"
        left_input_ids_text = left_input_ids_text[1:-1]
        left_text_offsets = left_text_offsets[1:-1]

    # Pre-computed text needs trim of head and tail token
    if stance_as_feature is not None:
        # print('Is not none')
        feature_text_ids = precomputed_tokens_and_offsets["ids"][stance]
        feature_text_offsets = precomputed_tokens_and_offsets["offsets"][stance]
        if feature_text_ids[0] == special_tokens["cls"]:
            sys.exit('!!!!!Error: special_tokens are not supposed to appear')
            feature_text_ids = feature_text_ids[1:-1]
            feature_text_offsets = feature_text_offsets[1:-1]
    else:
        feature_text_ids = []
        feature_text_offsets = []

    # So the new is CLS + SEP + text_ids + SEP
    if 'deberta' in model_name:
        assert len(feature_text_ids) <= 13
    else:
        assert len(feature_text_ids) <= 16
    new_max_len = max_len - 3 - len(feature_text_ids)
    if new_max_len < len(left_input_ids_text):
        print('new_max_len {}, max_len {}, len(feature_text_ids) {}, len(left_input_ids_text) {}'.format(
            new_max_len, max_len, len(feature_text_ids), len(left_input_ids_text)))
    if new_max_len < len(right_input_ids_text):
        print('new_max_len {}, max_len {}, len(feature_text_ids) {}, len(right_input_ids_text) {}'.format(
            new_max_len, max_len, len(feature_text_ids), len(right_input_ids_text)))
    # # To accommodate softmax
    # assert new_max_len >= len(input_ids_text)
    # # To accommodate softmax
    if new_max_len < len(left_input_ids_text):
        # print('Warning: new_max_len %d > input_ids_text len %d' % (new_max_len, input_ids_text))
        print('Warning: new_max_len %d < left_input_ids_text len %d' % (new_max_len, len(left_input_ids_text)))
        print('         The feature_len is %d' % (len(feature_text_ids)))
    if new_max_len < len(right_input_ids_text):
        # print('Warning: new_max_len %d > input_ids_text len %d' % (new_max_len, input_ids_text))
        print('Warning: new_max_len %d < right_input_ids_text len %d' % (new_max_len, len(right_input_ids_text)))
        print('         The feature_len is %d' % (len(feature_text_ids)))
    # see tokens[sentiment] = ids[0] if ids[0] != tokens['cls'] else ids[1]
    #     feature_text_ids originally was supposed to be a token id
    left_input_ids = (
        [special_tokens["cls"]] + feature_text_ids + [special_tokens["sep"]]
            + left_input_ids_text[:new_max_len]
            + [special_tokens["sep"]]
    )
    right_input_ids = (
        [special_tokens["cls"]] + feature_text_ids + [special_tokens["sep"]]
            + right_input_ids_text[:new_max_len]
            + [special_tokens["sep"]]
    )
    # token_type_ids indicates whether the token is special or non-special
    # Be aware that the last sep is 1 according to https://huggingface.co/transformers/v3.2.0/glossary.html
    # token_type_ids = [0, 0, 0] + [1] * (len(input_ids_text[:new_max_len]) + 1)
    left_token_type_ids = [0] + [0] * len(feature_text_ids) + [0] + [1] * (len(left_input_ids_text[:new_max_len]) + 1)
    right_token_type_ids = [0] + [0] * len(feature_text_ids) + [0] + [1] * (len(right_input_ids_text[:new_max_len]) + 1)
    # So the special tokens (sep sentiment sep) correspond to no offsets, as well as the last sep
    # So tokenizer.encode() did the first tokenization,
    # Then the text become pure text (had been removed from [CLS] and [SEP])
    # text_offsets = [(0, 0)] * 3 + text_offsets[:new_max_len] + [(0, 0)]
    # print(new_max_len)
    # Be aware that new_max_len is the token count, instead of text len
    # These can be accelerated by setting num_workers = 2
    left_text_offsets = [(0, 0)] * (2 + len(feature_text_ids)) + left_text_offsets[:new_max_len] + [(0, 0)]
    right_text_offsets = [(0, 0)] * (2 + len(feature_text_ids)) + right_text_offsets[:new_max_len] + [(0, 0)]
    assert len(left_input_ids) == len(left_token_type_ids) 
    assert len(left_input_ids) == len(left_text_offsets), (len(left_input_ids), len(left_text_offsets))
    assert len(right_input_ids) == len(right_token_type_ids) 
    assert len(right_input_ids) == len(right_text_offsets), (len(right_input_ids), len(right_text_offsets))

    # Padding is meant for those too short. But too long sentences will suffer from loss of data
    left_padding_length = max_len - len(left_input_ids)
    left_len_input_ids_before_padding_length = len(left_input_ids)
    # # To accommodate softmax
    # assert input_ids[len_input_ids_before_padding_length - 1] ==  special_tokens["sep"]
    # # To accommodate softmax
    if left_padding_length > 0:
        left_input_ids = left_input_ids + ([special_tokens["pad"]] * left_padding_length)
        left_token_type_ids = left_token_type_ids + ([0] * left_padding_length)
        left_text_offsets = left_text_offsets + ([(0, 0)] * left_padding_length)

    right_padding_length = max_len - len(right_input_ids)
    if right_padding_length > 0:
        right_input_ids = right_input_ids + ([special_tokens["pad"]] * right_padding_length)
        right_token_type_ids = right_token_type_ids + ([0] * right_padding_length)
        right_text_offsets = right_text_offsets + ([(0, 0)] * right_padding_length)

    toks = []

    for i, (a, b) in enumerate(left_text_offsets):
        sm = np.sum(chars[a:b])
        if sm > 0:
            toks.append(i)
    if len(toks) == 0:
        toks = [0]

    targets_start = toks[0]
    targets_end = toks[-1]

    return {
        "leftids": left_input_ids,
        "rightids": right_input_ids,
        "left_token_type_ids": left_token_type_ids,
        "right_token_type_ids": right_token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        "lefttext": lefttext,
        "righttext": righttext,
        'aspect_span': aspect_span,
        'aspect_span_text': aspect_span_text,
        # "sentiment": sentiment,
        # Originally, sentiment text was not used, so here, I directly no longer include it.
        "left_offsets": left_text_offsets,
        'sEmbeds': sEmbeds,
        "instance_weight": instance_weight
    }


# TweetTrainDataset always focuses on the token level, and only uses CrossEntropy loss
# Metric is used only in the validation set.
class TweetTrainingDataset(Dataset):
    def __init__(self, df, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=310, model_name="bert"):
        self.special_tokens = special_tokens
        self.precomputed_tokens_and_offsets = precomputed_tokens_and_offsets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

        self.text_external_ids = df['ID'].values
        self.clean_lefttexts = df['clean_lefttext'].values
        self.clean_righttexts = df['clean_righttext'].values
        self.notorsame = df['notorsame'].values
        self.aspect_spans = df['aspect_span'].values
        self.aspect_span_texts = df['aspect_span_text'].values
        
        self.stances = df['stance'].values
        self.stances_cate = df['stance_cate'].values

        self.left_aspect_cates = df['left_aspect_category'].values
        self.right_aspect_cates = df['right_aspect_category'].values

    def __len__(self):
        return len(self.clean_lefttexts)

    def __getitem__(self, idx):
        data = process_training_data(self.clean_lefttexts[idx], self.clean_righttexts[idx], self.aspect_spans[idx], self.aspect_span_texts[idx], self.stances[idx], self.left_aspect_cates[idx], self.right_aspect_cates[idx], self.tokenizer, self.special_tokens, self.precomputed_tokens_and_offsets,
                                     max_len=self.max_len, model_name=self.model_name)
        return {
            'external_id': self.text_external_ids[idx],
            'leftids': torch.tensor(data["leftids"], dtype=torch.long),
            'rightids': torch.tensor(data["rightids"], dtype=torch.long),
            'left_token_type_ids': torch.tensor(data["left_token_type_ids"], dtype=torch.long),
            'right_token_type_ids': torch.tensor(data["right_token_type_ids"], dtype=torch.long),
            'targets_start': data["targets_start"],
            'targets_end': data["targets_end"],
            'lefttext': data["lefttext"],
            'righttext': data["righttext"],
            # This is list, then it will be automatically transferred to tensor. Unless nestged, in that case I have to do it manually. Non nested strings won't be converted to Torch tensor
            'notorsame': self.notorsame[idx],
            "aspect_span": data['aspect_span'],
            "aspect_span_texts": data["aspect_span_text"],
            'stance_cate': self.stances_cate[idx],
            'left_offsets': np.array(data["left_offsets"], dtype=np.int_),
            'instance_weight': data['instance_weight'],
            'sEmbeds': data['sEmbeds']
        }


class DOCTransformer(nn.Module):
    # 2022/08/08
    def __init__(self, model, is_automodel, nb_layers=1, nb_ft=None, nb_class=None, pretrained=False, local_pretrained=False, nb_cate=None, nb_sbert=None, multi_sample_dropout=False, training=False, use_squad_weights=True):
        super().__init__()
        self.name = model
        self.is_automodel = is_automodel
        # number of layers
        self.nb_layers = nb_layers
        self.multi_sample_dropout = multi_sample_dropout
        # self.training = training

        # padding token_id
        self.pad_idx = 1 if "roberta" in self.name else 0

        # So here they actualy load model class and config from a dict. This is unlike C.
        model_class, str_model_official_name, config_class = TRANSFORMERS[self.name]

        # Change 0421: tokenizer and config_class and model loading changed
        # try:
        #     config = config_class.from_json_file(MODEL_PATHS[model] + 'bert_config.json')
        # except:
        #     config = config_class.from_json_file(MODEL_PATHS[model] + 'config.json')
        if self.is_automodel is True:
            uu_model_class, str_model_offical_name, uu_config_class = TRANSFORMERS[self.name]
            if MODEL_PATHS[self.name] is None:
                config = AutoConfig.from_pretrained(str_model_offical_name, output_hidden_states=True)
                config.save_pretrained(f'../input/AutoModelconfigs/{self.name}/')
                sys.exit(f'MODEL_PATHS[self.name] is None, use online config value and saved to \'../input/AutoModelconfigs/{self.name}/\', please specify it in MODEL_PATHS in .py and restart the program')
                MODEL_PATHS[self.name] = f'../input/AutoModelconfigs/{self.name}/'
            else:
                config = AutoConfig.from_pretrained(MODEL_PATHS[self.name], output_hidden_states=True)
            # Load model weights
            if local_pretrained:
                print('I am loading from local pre-trained')
                self.transformer = AutoModel.from_pretrained('~/blablapath', config=config)
            elif pretrained:
                print('I am loading from online pre-trained')
                self.transformer = AutoModel.from_pretrained(str_model_offical_name, config=config,cache_dir="/scratch/prj/inf_llmcache/hf_cache")
                print('Loading completed')
            else:
                print('I am training from scratch')
                self.transformer = AutoModel(config)
        elif self.is_automodel is False:
            model_class, str_model_offical_name, config_class = TRANSFORMERS[self.name]
            if 'deberta-v3-large-old' in self.name:
                if MODEL_PATHS[model] is None:
                    config = config_class.from_pretrained(str_model_offical_name, output_hidden_states=True)
                    config.save_pretrained(f'../input/debertaconfigs/{self.name}/')
                    sys.exit(f'MODEL_PATHS[self.name] is None, use online config value and saved to \'../input/debertaconfigs/{self.name}/\', please specify it in MODEL_PATHS in .py and restart the program')
                    MODEL_PATHS[model] = '../input/debertaconfigs/deberta-v3-large/'
                else:
                    # assert False
                    config = config_class.from_pretrained(MODEL_PATHS[self.name], output_hidden_states=True)
                if local_pretrained:
                    print('I am loading from local pre-trained')
                    self.transformer = model_class.from_pretrained('~/blablapath', config=config)
                elif pretrained:
                    print('I am loading from pre-trained')
                    self.transformer = model_class.from_pretrained(str_model_offical_name, config=config,cache_dir="/scratch/prj/inf_llmcache/hf_cache")
                else:
                    print('I am training from scratch')
                    self.transformer = model_class(config)
            elif 'albert-xxlarge-v2' in self.name:
                if MODEL_PATHS[self.name] is None:
                    config = config_class.from_pretrained(str_model_offical_name, output_hidden_states=True)
                    config.save_pretrained(f'../input/albertconfigs/{self.name}/')
                    sys.exit(f'MODEL_PATHS[self.name] is None, use online config value and saved to \'../input/albertconfigs/{self.name}/\', please specify it in MODEL_PATHS in .py and restart the program')
                    MODEL_PATHS[self.name] = f'../input/albertconfigs/{self.name}/'
                else:
                    # assert False
                    config = config_class.from_pretrained(MODEL_PATHS[self.name], output_hidden_states=True)
                if local_pretrained:
                    print('I am loading from local pre-trained')
                    self.transformer = model_class.from_pretrained('~/blablapath', config=config)
                elif pretrained:
                    print('I am loading from online pre-trained')
                    self.transformer = model_class.from_pretrained(str_model_offical_name, config=config, cache_dir="/scratch/prj/inf_llmcache/hf_cache")
                else:
                    print('I am loading from local pre-trained')
                    self.transformer = model_class(config)
            elif 'albert-large-v2' in self.name:
                if MODEL_PATHS[self.name] is None:
                    config = config_class.from_pretrained(str_model_offical_name, output_hidden_states=True)
                    config.save_pretrained(f'../input/albertconfigs/{self.name}/')
                    sys.exit(f'MODEL_PATHS[self.name] is None, use online config value and saved to \'../input/albertconfigs/{self.name}/\', please specify it in MODEL_PATHS in .py and restart the program')
                    MODEL_PATHS[self.name] = '../input/albertconfigs/{self.name}'
                else:
                    # assert False
                    config = config_class.from_pretrained(MODEL_PATHS[self.name], output_hidden_states=True)
                if pretrained:
                    print('I am loading from online pre-trained')
                    self.transformer = model_class.from_pretrained(str_model_offical_name, config=config, cache_dir="/scratch/prj/inf_llmcache/hf_cache")
                else:
                    self.transformer = model_class(config)
            elif 'albert' in self.name:
                sys.exit('config_class.from_json_file, code still in construction, so exit')
                config = config_class.from_json_file(MODEL_PATHS[model] + 'config.json')
                if pretrained:
                    print('I am loading from pre-trained')
                    self.transformer = model_class.from_pretrained('microsoft/deberta-v3-large', config=config)
                else:
                    self.transformer = model_class(config)
        else:
            sys.exit('The is_automodel is None!!!')

        # config.output_hidden_states = True

        # This is actually the deep pretrained language model
        

        if "distil" in self.name:
            self.nb_features = self.transformer.transformer.layer[-1].ffn.lin2.out_features
        elif "albert" in self.name:
            # The last layers
            self.nb_features = self.transformer.encoder.albert_layer_groups[-1].albert_layers[-1].ffn_output.out_features
        elif "deberta-v3" in self.name:
            self.nb_features = self.transformer.encoder.layer[-1].output.dense.out_features
        else:
            self.nb_features = self.transformer.pooler.dense.out_features

        if nb_ft is None:
            # Number of features
            nb_ft = self.nb_features
        # 128
        # print('nb_ft', nb_ft)
        # 8
        # print('nb_layers', nb_layers)

        # So a linear layer to aggregate all the hidden layers of all token positions, to nb_ft, and is shared among every token positions
        self.logits = nn.Linear(self.nb_features, nb_class)
        self.cates = nn.Linear(self.nb_features, nb_cate)

        # 2022/08/08
        self.sbert_linear = nn.Linear(self.nb_features, nb_sbert)

        self.high_dropout = nn.Dropout(p=0.5)

        self.siameseW = nn.Linear(3 * self.nb_features, 2)

    def forward(self, left_tokens, left_token_type_ids, right_tokens, right_token_type_ids, sentiment=0):
        # 2022/08/11 TODO: actually we still need to balance the training instances regarding other cls losses
        # 2022/08/11 Some versions compute attention_mask in the process_training_instances or pre-process
        if "distil" in self.name:
            left_hidden_states = self.transformer(
                left_tokens,
                attention_mask=(left_tokens != self.pad_idx).long(),
            )[-1]
            right_hidden_states = self.transformer(
                right_tokens,
                attention_mask=(right_tokens != self.pad_idx).long(),
            )[-1]
        else:
            # This actually runs an albert-large-v2 model with specified tokens
            left_hidden_states = self.transformer(
                left_tokens,
                attention_mask=(left_tokens != self.pad_idx).long(),
                token_type_ids=left_token_type_ids,
            )[0]
            right_hidden_states = self.transformer(
                right_tokens,
                attention_mask=(right_tokens != self.pad_idx).long(),
                token_type_ids=right_token_type_ids,
            )[0]
            # hidden_states = self.transformer(
            #     tokens,
            #     attention_mask=(tokens != self.pad_idx).long(),
            #     token_type_ids=token_type_ids,
            # )[-1]
            
        # By testing, the hidden state size is the length of the tuple.
        # print('hidden_states tuple size', len(hidden_states))
        # print('hidden_states[0] size', hidden_states[0].size())

        # old_hidden_states = hidden_states

        # hidden_states = hidden_states[::-1]
        left_hidden_states = left_hidden_states
        right_hidden_states = right_hidden_states

        # So the original hidden_states order is [0, 1, 2, 3] layer
        # and 3rd layer is the last layer.
        # now, 0th layer is the last layer.
        # # test if the last layer of the status is the final output layer
        # last_layer_states = self.transformer(
        #     tokens,
        #     attention_mask=(tokens != self.pad_idx).long(),
        #     token_type_ids=token_type_ids,
        # )[0]
        # print('len(last_layer_states)', len(last_layer_states))

        # # eq: element-wise, equal: two tensors
        # print('If the last layer of hidden states is the first layer states', torch.equal(hidden_states[0], last_layer_states))

        # print('if only change the order', torch.eq(old_hidden_states[0], hidden_states[1]))

        # print('The actual layer size is %d and we only select the top %d layers' % (len(hidden_states), self.nb_layers))
        # The actual layer size is 25 and we only select the top 8 layers

        # The last nb_layers has been used and concatenated with:
        # [[1,2,3]
        # [4,5,6]
        # [5,6,7]
        # [6,7,8]
        # =>
        # [1,2,3,4,5,6,5,6,7,6,7,8]

        left_features = left_hidden_states
        right_features = right_hidden_states

        if self.multi_sample_dropout and self.training:
            left_logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(left_features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
            left_clss = torch.mean(
                torch.stack(
                    [self.cates(self.high_dropout(left_features[:,0,:])) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
            # 2022/08/08
            usbert = torch.mean(self.sbert_linear(left_features), dim=1)
            left_meandrep = torch.mean(left_features, dim=1)

            # right_logits = torch.mean(
            #     torch.stack(
            #         [self.logits(self.high_dropout(right_features)) for _ in range(5)],
            #         dim=0,
            #     ),
            #     dim=0,
            # )
            right_meandrep = torch.mean(right_features, dim=1)
        # # else:
        #     logits = self.logits(features)
        else:
            # This is used in the evaluation mode.
            left_logits = self.logits(left_features)
            left_clss = self.cates(left_features[:,0,:])
            # 2022/08/08
            usbert = torch.mean(self.sbert_linear(left_features), dim=1)
            left_meandrep = torch.mean(left_features, dim=1)
            # Well, 32 is the batch size
            # print(logits.size())
            # right_logits = self.logits(right_features)
            right_meandrep = torch.mean(right_features, dim=1)
        # 32 * 256 * 2
        # Then 32 * 256

        # position_logits = logits[:, :]
        start_logits, end_logits = left_logits[:, :, 0], left_logits[:, :, 1]

        siamesecls = self.siameseW(torch.cat((left_meandrep, right_meandrep, torch.abs(left_meandrep - right_meandrep)), dim=1))

        return start_logits, end_logits, left_clss, usbert, siamesecls

        # return position_logits


class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.best_start = None
        self.best_end = None
        # self.best_pos_logits_sgmd = None

    # def __call__(self, epoch_score, model, model_path, pos_logits_sgmd=None):
    def __call__(self, epoch_score, model, model_path, start_oof, end_oof):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.best_start = start_oof
            self.best_end = end_oof
            # self.best_pos_logits_sgmd = pos_logits_sgmd
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_start = start_oof
            self.best_end = end_oof
            # self.best_pos_logits_sgmd = pos_logits_sgmd
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            # torch.save(model.state_dict(), model_path)
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), model_path)
        self.val_score = epoch_score


def loss_fn(start_logits, end_logits, start_positions, end_positions,clss, stance_cate, usEmbeds, target_sbertEmb, t_sigma, notorsame_clss, notorsame, nli_instance_weight, weight=None):
    if weight is not None:
        # reduction by default mean, meaning that the tuples within a batch will be meaned
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        start_loss = loss_fct(start_logits, start_positions)
        # print(start_loss.shape)
        # start_loss = start_loss * weight
        # This is also instance-level loss
        start_loss = (start_loss * weight / weight.sum()).sum()
        # start_loss = (start_loss * weight / weight.sum()).sum()
        start_loss = start_loss.mean()
        end_loss = loss_fct(end_logits, end_positions)
        # end_loss = end_loss * weight
        end_loss = (end_loss * weight / weight.sum()).sum()
        # mean by default iterates all the elements in a matrix
        end_loss = end_loss.mean()
        cls_loss = loss_fct(clss, stance_cate).mean()
        
        # 2022/08/08
        normaldist = Normal(target_sbertEmb, t_sigma)
        usEmbed_loss = normaldist.log_prob(usEmbeds).sum(dim=1)
        usEmbed_loss = - usEmbed_loss.mean()

        if nli_instance_weight is not None:
            nli_loss = loss_fct(notorsame_clss, notorsame)
            # The weight has been pre-defined
            nli_loss = (nli_loss * nli_instance_weight).sum()
            nli_loss = nli_loss.mean()
        else:
            loss_fct_nli = nn.CrossEntropyLoss()
            nli_loss = loss_fct_nli(notorsame_clss, notorsame)

        # # reduction by default mean, meaning that the tuples within a batch will be meaned
        # # loss_fct = nn.CrossEntropyLoss(reduction='none')
        # # See tentetive tests 
        # # Also see '# Be aware that here the last layer is a Linear' to see that the sigmoid is not called
        # #    The last layer is a Linear so we need to include the sigmoid function here.
        # loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        # # start_loss = loss_fct(start_logits, start_positions)

        # pos_loss = loss_fct(position_logits, pos_logits_grt)
        # # print(start_loss.shape)
        # # start_loss = start_loss * weight
        # # start_loss = (start_loss * weight / weight.sum()).sum()

        # # pos_loss = (pos_loss * weight / weight.sum(dim=1).unsqueeze(1).expand_as(pos_loss))
        # # print(pos_loss.size())
        # # pos_loss_2 = pos_loss * weight
        # # assert pos_loss.equal(pos_loss_2)
        # # print(weight.size())
        # # pos_loss = (pos_loss * weight / weight.sum(dim=1).unsqueeze(1).expand_as(pos_loss))
        # pos_loss = pos_loss * weight
        # pos_loss = torch.masked_select(pos_loss, pos_logits_grt != -1).mean()
        # # pos_loss = torch.masked_select(pos_loss, pos_logits_grt != -1).sum()

        # # start_loss = start_loss.mean()
        # # pos_loss = pos_loss.mean()

        # # end_loss = loss_fct(end_logits, end_positions)
        # # # end_loss = end_loss * weight
        # # end_loss = (end_loss * weight / weight.sum()).sum()
        # # # mean by default iterates all the elements in a matrix
        # end_loss = end_loss.mean()
    else:
        loss_fct = nn.CrossEntropyLoss()
        # loss_fct = LabelSmoothingLoss(config.MAX_LEN, smoothing=0.1)
        # loss_fct = SmoothCrossEntropyLoss(smoothing=0.9)

        # So start_logits is 32 * 256, what will start_loss produce?
        # end_position is indeed the index here, it's correct
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        cls_loss = loss_fct(clss, stance_cate)

        # 2022/08/08
        normaldist = Normal(target_sbertEmb, t_sigma)
        usEmbed_loss = normaldist.log_prob(usEmbeds).sum(dim=1)
        usEmbed_loss = - usEmbed_loss.mean()

        if nli_instance_weight is not None:
            nli_loss_fct = nn.CrossEntropyLoss(reduction='none')
            nli_loss = nli_loss_fct(notorsame_clss, notorsame)
            # The weight has been pre-defined
            nli_loss = (nli_loss * nli_instance_weight).sum()
            nli_loss = nli_loss.mean()
        else:
            nli_loss = loss_fct(notorsame_clss, notorsame)

        # # loss_fct = nn.CrossEntropyLoss()
        # # # loss_fct = LabelSmoothingLoss(config.MAX_LEN, smoothing=0.1)
        # # # loss_fct = SmoothCrossEntropyLoss(smoothing=0.9)

        # # # So start_logits is 32 * 256, what will start_loss produce?
        # # # end_position is indeed the index here, it's correct
        # # start_loss = loss_fct(start_logits, start_positions)
        # # end_loss = loss_fct(end_logits, end_positions)
        # # # start_loss = ohem_loss(start_logits, start_positions)
        # # # end_loss = ohem_loss(end_logits, end_positions)
        # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        # pos_loss = loss_fct(position_logits, pos_logits_grt)
        # pos_loss = torch.masked_select(pos_loss, pos_logits_grt != -1).mean()

    # total_loss = (start_loss + end_loss)
    # total_loss = pos_loss
    total_loss = start_loss + end_loss + cls_loss + usEmbed_loss + nli_loss
    # total_loss = distloss
    # total_loss = (start_loss + end_loss)
    return total_loss


def token_pred_to_char_pred(token_pred, offsets):

    # Well, due to the change of loss function (scilicet, we use BCDlogitCrossEntropy loss in replacement of ), token_pred has been squeezed, so here we change it accordingly
    if len(token_pred.shape) == 1:
        char_pred = np.zeros(np.max(offsets))
    else:
        char_pred = np.zeros((np.max(offsets), token_pred.shape[1]))
    for i in range(len(token_pred)):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        char_pred[s:e] = token_pred[i]

    return char_pred


def post_process_spaces(target, text):
    '''
    before this there's a step to lambda x: (x > 0.5).flatten(), actually it's equal to argmax(1 - x, x)
    deal with spaces
    target is the predicted char-wise TorF array
    '''
    # won't affect the original target
    target = np.copy(target)

    # clean_text shouldn't affect the text length
    #  the len(text) and len(target) doesn't equal. The error will happen when len(text) > len(target), the cause of this error, is because white space (i.e., \n \r \t \s) in tail is not counted into the offset, and is not tokenized
    assert len(target) == len(text)

    if len(text) > len(target):
        padding = np.zeros(len(text) - len(target))
        target = np.concatenate([target, padding])
    else:
        target = target[:len(text)]

    # To facilitate i + 1, to avoid that text starts with space
    if text[0] == " ":
        target[0] = 0
    if text[-1] == " ":
        target[-1] = 0

    for i in range(1, len(text) - 1):
        if text[i] == " ":
            if target[i] and not target[i - 1]:  # space before
                target[i] = 0

            # if the space is a space precursoring a word
            if target[i] and not target[i + 1]:  # space after
                target[i] = 0

            # any interval spaces will be kept
            if target[i - 1] and target[i + 1]:
                target[i] = 1

    return target


def char_target_to_span(char_target):
    spans = []
    start, end = 0, 0
    for i in range(len(char_target)):
        if char_target[i] == 1 and char_target[i - 1] == 0:
            if end:
                spans.append([start, end])
            start = i
            end = i + 1
        elif char_target[i] == 1:
            end = i + 1
        else:
            if end:
                spans.append([start, end])
            start, end = 0, 0
    if end == len(char_target) and start != 0:
        # The last one is included
        spans.append([start, end])
        start = 0
        end = 0
    assert start == 0, end == 0
    return spans


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MicroF1Meter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # self.val_TP = 0
        # self.avg_TP = 0
        # self.avg_
        self.sum_TP = 0
        self.sum_FN = 0
        self.sum_FP = 0
        self.prec = 0
        self.reca = 0
        self.microF1 = 0
        self.count = 0

    def update(self, val_TP, val_FN, val_FP, n=1):
        # self.val = val_TP
        self.sum_TP += val_TP
        self.sum_FN += val_FN
        self.sum_FP += val_FP
        self.count += n
        # self.avg = self.sum / self.count
        if self.sum_TP == 0:
            self.prec = 0
            self.reca = 0
            self.microF1 = 0
        else:
            self.prec = self.sum_TP / (self.sum_TP + self.sum_FP)
            self.reca = self.sum_TP / (self.sum_TP + self.sum_FN)
            self.microF1 = 2 * self.prec * self.reca / (self.prec + self.reca)


# if the two intervals overlaps
def is_overlaping(a, b):
  if b[0] >= a[0] and b[0] < a[1]:
    return True
  else:
    return False


def merge_two_sorted_spans(spans_pred, spans_grt):
    test_list1 = spans_pred
    test_list2 = spans_grt
    size_1 = len(test_list1)
    size_2 = len(test_list2) 
    res = []
    i, j = 0, 0

    while i < size_1 and j < size_2:
        if test_list1[i][0] < test_list2[j][0]: 
            res.append(test_list1[i]) 
            i += 1
        else:
            res.append(test_list2[j]) 
            j += 1
    res = res + test_list1[i:] + test_list2[j:]
    # # Only for test group 2
    # flag_is_sorted = True
    # for idx in range(len(res)):
    #     if idx + 1 < len(res):
    #         if res[idx][0] > res[idx + 1][0]:
    #             flag_is_sorted = False
    # assert flag_is_sorted
    # # Only for test group 2

    arr = res
    # sort the intervals by its first value
    arr.sort(key = lambda x: x[0])

    merged_list= []
    if len(arr) > 0:
        merged_list.append(arr[0])
        for i in range(1, len(arr)):
            pop_element = merged_list.pop()
            if is_overlaping(pop_element, arr[i]):
                new_element = [pop_element[0], max(pop_element[1], arr[i][1])]
                merged_list.append(new_element)
            else:
                merged_list.append(pop_element)
                merged_list.append(arr[i])

    return merged_list


def intersect_two_sorted_spans(spans_pred, spans_grt):
    i = j = 0
    
    n = len(spans_pred)
    m = len(spans_grt)

    res = []
    while i < n and j < m:
        
        # Left bound for intersecting segment
        l = max(spans_pred[i][0], spans_grt[j][0])
        
        # Right bound for intersecting segment
        r = min(spans_pred[i][1], spans_grt[j][1])
        
        if l < r:
            res.append([l, r])

        if spans_pred[i][1] < spans_grt[j][1]:
            i += 1
        else:
            j += 1
    return res


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard_score_modified(
    original_tweet,
    offsets,
    spans_pred,
    spans_grt):
    idx_start = spans_pred[0][0]
    idx_end = spans_pred[0][1]
    if idx_end < idx_start:
        filtered_output = original_tweet
    else:
        filtered_output = original_tweet[offsets[idx_start][0]:offsets[idx_end][1]]
    target_string = original_tweet[spans_grt[0][0]:spans_grt[0][1]]
    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac
#     # return jac, filtered_output.strip()


def calculate_jaccard_score(
    original_tweet,
    offsets,
    spans_pred,
    spans_grt):
    # union
    idx_start = spans_pred[0][0]
    idx_end = spans_pred[0][1]
    if idx_end < idx_start:
        spans_pred_temp_s = 0
        spans_pred_temp_e = len(original_tweet)
    # Changed in 2022/08/08 to accommodate the idx_start=start_end_pair[0][0],
    #     IndexError: list index out of range
    # Error
    elif idx_start == -1 and idx_end == 0:
        spans_pred_temp_s = 0
        spans_pred_temp_e = 0
    else:
        spans_pred_temp_s = offsets[idx_start][0]
        spans_pred_temp_e = offsets[idx_end][1]
    l = max(spans_pred_temp_s, spans_grt[0][0])
    # Right bound for intersecting segment
    r = min(spans_pred_temp_e, spans_grt[0][1])
    
    if l < r:
        intersec_num = r - l
    else:
        intersec_num = 0
        
    if l < r:
        l_border = min(spans_pred_temp_s, spans_grt[0][0])
        r_border = max(spans_pred_temp_e, spans_grt[0][1])
        merged_list_sum = r_border - l_border
    else:
        merged_list_sum = spans_pred_temp_e - spans_pred_temp_s + spans_grt[0][1] - spans_grt[0][0]
    
    if intersec_num == 0 and merged_list_sum == 0:
        # The only difference theoretically
        # return 1.0
        return 0.5
    else:
        return float(intersec_num) / float(merged_list_sum)


def calculate_jaccard_score_old(
        original_tweet,
        target_string,
        idx_start,
        idx_end,
        offsets,
        model_name,
        verbose=False):
    if idx_end < idx_start:
        filtered_output = original_tweet
    elif idx_start == -1 and idx_end == 0:
        filtered_output = ''
    else:
        filtered_output = original_tweet[offsets[idx_start][0]:offsets[idx_end][1]]

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output.strip()


def calculate_TP_FN_FP_between_2lstsofintervals(
    text,
    offsets,
    spans_pred,
    spans_grt):
    char_pred = np.zeros(len(text))
    char_grt = np.zeros(len(text))
    tp = 0
    fn = 0
    fp = 0
    for aspan_pred in spans_pred:
        a = offsets[aspan_pred[0]][0]
        # if aspan_pred[1] > 120:
        #     print(b)
        b = offsets[aspan_pred[1]][1]
        char_pred[offsets[aspan_pred[0]][0]:offsets[aspan_pred[1]][1]] = 1
    for aspan_grt in spans_grt:
        char_grt[aspan_grt[0]:aspan_grt[1]] = 1
        # if is_overlaping(aspan_pred, aspan_grt):
        #     print('Overlapping detected!')
    for i in range(len(text)):
        if char_pred[i] == 1 and char_grt[i] == 1:
            tp += 1
        elif char_grt[i] == 1:
            fn += 1
        elif char_pred[i] == 1:
            fp += 1
    return tp, fn, fp


def calculate_preds_between_2lstsofintervals(
    text,
    offsets,
    spans_pred,
    spans_grt):
    char_pred = np.zeros(len(text))
    char_grt = np.zeros(len(text))
    for aspan_pred in spans_pred:
        char_pred[offsets[aspan_pred[0]][0]:offsets[aspan_pred[1]][1]] = 1
    for aspan_grt in spans_grt:
        char_grt[aspan_grt[0]:aspan_grt[1]] = 1
    char_pred_trimmed = char_pred # np.zeros(len(text))
    char_grt_trimmed = char_grt # np.zeros(len(text))

    # only for testing group 0424 2.1
    return char_pred_trimmed, char_grt_trimmed
# only for testing group 0424 2

# 2022/08/08
def calculate_lg_normal_score(
        px_usEmbeds,
        px_target_sbertEmb):
    # ng_lg_normal_score = 0 - np.log(scipy.stats.multivariate_normal(px_target_sbertEmb, 1).pdf(px_usEmbeds))
    # normal_score = scipy.stats.multivariate_normal(px_target_sbertEmb, 1).pdf(px_usEmbeds)
    normal_score = scipy.stats.norm(px_target_sbertEmb, 1).pdf(px_usEmbeds)
    lg_normal_score = np.log(normal_score).sum()
    return lg_normal_score

# def eval_fn(data_loader, model, device, tokenizer, model_name):
def eval_fn(data_loader, model, device, model_name, activation):
    model.eval()
    losses = AverageMeter()
    # # testing group 0424 1
    jaccards = AverageMeter()
    # # testing group 0424 1
    mf1 = MicroF1Meter()
    mf1_se = MicroF1Meter()
    mf1_st = MicroF1Meter()

    mf1_notorsame = MicroF1Meter()

    # just for test group 0424 2
    bin_preds_byb = []
    bin_truths_byb = []
    # just for test group 0424 2

    # output_pos_logits_sgmd = []
    start_array, end_array = [], []

    # just for test group 0425 1
    ext_ids_byb = []
    locations_byb = []
    # just for test group 0425 1
    # 2022/08/08
    t_sigma = torch.tensor(1, dtype=torch.float32, device=device)
    ng_lg_normals = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            leftids = d["leftids"]
            left_token_type_ids = d["left_token_type_ids"]
            # looks that it's equivalent to d["offsets"].detach().numpy()
            left_offsets = d["left_offsets"].numpy()

            rightids = d["rightids"]
            right_token_type_ids = d["right_token_type_ids"]
            # right_offsets = d["right_offsets"].numpy()
            # 2022/08/08
            target_sbertEmb = d["sEmbeds"]

            leftids = leftids.to(device, dtype=torch.long)
            left_token_type_ids = left_token_type_ids.to(device, dtype=torch.long)
            rightids = rightids.to(device, dtype=torch.long)
            right_token_type_ids = right_token_type_ids.to(device, dtype=torch.long)
            
            # 2022/08/08
            target_sbertEmb = target_sbertEmb.to(device, dtype=torch.float32)
            # mask = mask.to(device, dtype=torch.long)

            # didn't trim_tensors here, not sure if it's because we need same-length output
            left_clean_text = d["lefttext"]
            # right_clean_text = d["righttext"]
            # aspect_span = d["aspect_span"]
            
            outputs_start, outputs_end, clss, usEmbeds, siameseclss = model(
                # ids=ids,
                left_tokens=leftids,
                left_token_type_ids=left_token_type_ids,
                right_tokens=rightids,
                right_token_type_ids=right_token_type_ids)
            
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            stance_cate = d["stance_cate"]

            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            stance_cate = stance_cate.to(device, dtype=torch.long)

            # 2022/08/08
            # notorsame_clss = model.siameseclassify(leftmeanrep, rightmeanrep)
            notorsame = d['notorsame'].to(device, dtype=torch.long)
            nli_instance_weight = d['instance_weight'].to(device, dtype=torch.float32)
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, clss, stance_cate, usEmbeds, target_sbertEmb, t_sigma, siameseclss, notorsame, nli_instance_weight, None)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            clss = torch.softmax(clss, dim=1).cpu().detach().numpy()
            siameseclss = torch.softmax(siameseclss, dim=1).cpu().detach().numpy()
            usEmbeds = usEmbeds.cpu().detach().numpy()
            target_sbertEmb = target_sbertEmb.cpu().detach().numpy()

            start_array.append(outputs_start)
            end_array.append(outputs_end)

            # outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            # outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            # start_array.append(outputs_start)
            # end_array.append(outputs_end)

            # clean_text = d["text"]
            location = d["aspect_span"]
            # # testing group 0424 1
            jaccard_scores = []
            # # testing group 0424 1

            # 2022/08/08
            lg_normal_scores = []

            # just for test group 0424 2
            bin_preds = []
            bin_truths = []
            # just for test group 0424 2

            # just for test group 0425 1
            ext_ids = d["external_id"]
            ext_ids_byb.extend(ext_ids)
            locations_byb.extend(location)
            # just for test group 0425 1
            tps = []
            fns = []
            fps = []
            tps_se = 0
            fns_se = 0
            fps_se = 0
            tps_st = 0
            fns_st = 0
            fps_st = 0
            tps_siamese = 0
            fns_siamese = 0
            fps_siamese = 0
            for px, clean_text_a in enumerate(left_clean_text):
                start_end_pair = []
                for start_index in np.argsort(outputs_start[px, :])[::-1][:20]:
                    for end_index in np.argsort(outputs_end[px, :])[::-1][:20]:
                        if end_index < start_index:
                            continue
                        start_end_pair.append((start_index, end_index))
                if len(start_end_pair) == 0:
                    # uncommented in 2022/08/08 to accommodate the idx_start=start_end_pair[0][0],
                    #     IndexError: list index out of range
                    # Error
                    start_end_pair.append((-1, 0))
                    # start_end_pair = []
                    # spans_pred = []
                else:
                    idx_start = start_end_pair[0][0]
                    idx_end = start_end_pair[0][1]
                    spans_pred = [(idx_start, idx_end)]
                location_px = location[px]
                if location_px.find(':') == -1:
                    location_px = '0:0'
                s_s, s_e = location_px.split(':')
                i_s = int(s_s)
                i_e = int(s_e)
                spans_grt = [(i_s, i_e)]

                tp, fn, fp = calculate_TP_FN_FP_between_2lstsofintervals(
                    clean_text_a,
                    offsets=left_offsets[px],
                    spans_pred=spans_pred,
                    spans_grt=spans_grt
                )
                tps.append(tp)
                fns.append(fn)
                fps.append(fp)

                # !!! It may happen that the offsets contain space, but it's neganegible
                if left_offsets[px][spans_pred[0][0]][0] == spans_grt[0][0]:
                    tps_se += 1
                else:
                    fps_se += 1
                    fns_se += 1
                if left_offsets[px][spans_pred[0][1]][1] == spans_grt[0][1]:
                    tps_se += 1
                else:
                    fps_se += 1
                    fns_se += 1
                stance_pred = np.argmax(clss[px, :])
                if stance_pred == stance_cate[px].item():
                    tps_st += 1
                else:
                    fns_st += 1
                    fps_st += 1
                
                siamese_pred = np.argmax(siameseclss[px, :])
                if siamese_pred == notorsame[px].item():
                    tps_siamese += 1
                else:
                    fns_siamese += 1
                    fps_siamese += 1

                # # just for test group 0424 1
                jaccard_score = calculate_jaccard_score(
                    clean_text_a,
                    offsets=left_offsets[px],
                    spans_pred=spans_pred,
                    spans_grt=spans_grt
                )
                jaccard_scores.append(jaccard_score)
                # # just for test group 0424 1

                # just for test group 0424 2
                pred, truth = calculate_preds_between_2lstsofintervals(
                    clean_text_a,
                    offsets=left_offsets[px],
                    spans_pred=spans_pred,
                    spans_grt=spans_grt
                )

                if not len(pred) and not len(truth):
                    print('Warning: not len(pred) and not len(truth) not satisified')
                    preds_for_f1score_input = []
                    truth_for_f1score_input = []
                else:
                    assert len(pred) == len(truth)
                    bin_preds.append(pred)
                    bin_truths.append(truth)
                    preds_for_f1score_input = np.concatenate(bin_preds)
                    truth_for_f1score_input = np.concatenate(bin_truths)
                # just for test group 0424 2
                
                # 2022/08/08
                lg_normal_score = calculate_lg_normal_score(
                    px_usEmbeds=usEmbeds[px, :],
                    px_target_sbertEmb=target_sbertEmb[px, :]
                )
                lg_normal_scores.append(lg_normal_score)

            # # just for test group 0424 1
            jaccards.update(np.mean(jaccard_scores), leftids.size(0))
            # # just for test group 0424 1

            # 2022/08/08
            # train_fn doesnt necessarily need to compute this, but eval need this
            ng_lg_normals.update(np.mean(lg_normal_scores), leftids.size(0))

            # just for test group 0424 2
            preds_for_f1score_input = np.concatenate(bin_preds)
            truth_for_f1score_input = np.concatenate(bin_truths)
            bin_preds_byb.extend(preds_for_f1score_input)
            bin_truths_byb.extend(truth_for_f1score_input)
            
            if bi % 50 == 0:
                micro_F1_sklearn_byb = f1_score(bin_preds_byb, bin_truths_byb)
                if micro_F1_sklearn_byb == 0.0:
                    is_there_any_overlap = False
                    is_there_any_non_zero = False
                    is_pred_non_zero = False
                    is_truth_non_zero = False
                    for pred_b, truth_b in zip(bin_preds_byb, bin_truths_byb):
                        if pred_b or truth_b:
                            is_there_any_non_zero = True
                            if pred_b:
                                is_pred_non_zero = True
                            if truth_b:
                                is_truth_non_zero = True
                            if pred_b and truth_b:
                                is_there_any_overlap = True
                    print('is_there_any_overlap {}\n is_there_any_non_zero {}\n is_pred_non_zero {}\n is_truth_non_zero {}\n'.format(
                        is_there_any_overlap, is_there_any_non_zero, is_pred_non_zero, is_truth_non_zero))
                    # print(ext_ids)
                    is_location_all_none = True
                    for alocation in locations_byb:
                        if alocation != '':
                            is_location_all_none = False
                    print('is_location_all_none {}'.format(is_location_all_none))
            # just for test group 0424 2
            mf1.update(np.sum(tps), np.sum(fns), np.sum(fps), leftids.size(0))
            mf1_se.update(tps_se, fns_se, fps_se, leftids.size(0))
            mf1_st.update(tps_st, fns_st, fps_st, leftids.size(0))
            mf1_notorsame.update(tps_siamese, fns_siamese, fps_siamese, leftids.size(0))
            losses.update(loss.item(), leftids.size(0))
            # tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
            # tk0.set_postfix(loss=losses.avg, micro_F1=mf1.microF1)
            # just for test group 0424 2
            # tk0.set_postfix(loss=losses.avg, micro_F1=mf1.microF1)
            # tk0.set_postfix(loss=losses.avg, mF1=mf1.microF1, jcd=jaccards.avg, mF1sklbyb=micro_F1_sklearn_byb)
            # tk0.set_postfix(loss=losses.avg, mF1=mf1.microF1, mF1_se=mf1_se.microF1, mF1sklbyb=micro_F1_sklearn_byb, jcd=jaccards.avg, mf1_st=mf1_st.microF1)
            tk0.set_postfix(loss=losses.avg, mF1=mf1.microF1, mF1_se=mf1_se.microF1, mF1_siamese=mf1_notorsame.microF1, mf1_st=mf1_st.microF1, ng_lg_normal=ng_lg_normals.avg)
            # just for test group 0424 2
    start_array = np.concatenate(start_array)
    end_array = np.concatenate(end_array)
    # output_pos_logits_sgmd = np.concatenate(output_pos_logits_sgmd)
    print(f"micro_F1 = {mf1.microF1}")
    # just for test group 0424 2
    print(f"Jaccard = {jaccards.avg}")
    print(f"ng_lg_normal = {ng_lg_normals.avg}")
    micro_F1_sklearn_byb = f1_score(bin_preds_byb, bin_truths_byb)
    if micro_F1_sklearn_byb == 0.0:
        is_there_any_overlap = False
        is_there_any_non_zero = False
        is_pred_non_zero = False
        is_truth_non_zero = False
        for pred_b, truth_b in zip(bin_preds_byb, bin_truths_byb):
            if pred_b or truth_b:
                is_there_any_non_zero = True
                if pred_b:
                    is_pred_non_zero = True
                if truth_b:
                    is_truth_non_zero = True
                if pred_b and truth_b:
                    is_there_any_overlap = True
        print('is_there_any_overlap {}\n is_there_any_non_zero {}\n is_pred_non_zero {}\n is_truth_non_zero {}\n'.format(
            is_there_any_overlap, is_there_any_non_zero, is_pred_non_zero, is_truth_non_zero))
        # print(ext_ids_byb)
        is_location_all_none = True
        for alocation in locations_byb:
            if alocation != '':
                is_location_all_none = False
        print('is_location_all_none {}'.format(is_location_all_none))
    print(f"mF1sklbyb = {micro_F1_sklearn_byb}")
    # just for test group 0424 2
    # return mf1.microF1, start_array, end_array
    return jaccards.avg, start_array, end_array, mf1_notorsame.microF1, ng_lg_normals.avg


def train_fn(data_loader, model, optimizer, device, scheduler=None, opt=None, epc=None, tokenizer=None, model_name=None, activation='sigmoid'):
    # model.name
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = AverageMeter()
    # # just for test group 0424 1
    jaccards = AverageMeter()
    jaccards_old = AverageMeter()
    # # just for test group 0424 1
    mf1 = MicroF1Meter()
    # mf1_se = MicroF1Meter()
    mf1_se = MicroF1Meter()
    mf1_st = MicroF1Meter()
    mf1_notorsame = MicroF1Meter()

    # 2022/08/08
    t_sigma = torch.tensor(1, dtype=torch.float32, device=device)

    # # just for test group 0424 2
    # bin_preds_byb = []
    # bin_truths_byb = []
    # # just for test group 0424 2

    # data loader iterator
    # tk0 = tqdm(data_loader, total=len(data_loader))
    tk0 = tqdm(data_loader, total=len(data_loader))
    # tk0 = tqdm.notebook.tqdm(data_loader, total=len(data_loader))

    # fgm = FGM(model)
    # ========== Only for test group 1
    # After discovering all the exceptions, now the saved exception no longer needs output
    # fpOut = open('theErrorMessages', 'wt')
    # ========== Only for test group 1
    print('for bi, d in enumerate(tk0):')
    for bi, d in enumerate(tk0):
        leftids = d["leftids"]
        left_token_type_ids = d["left_token_type_ids"]
        # don't need mask since the model will automatically mask it, and the tokenizer is inside the model
        # mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        # pos_logits_grt = d["pos_logits_grt"]
        # location = d["location"]

        left_clean_text = d["lefttext"]

        # sentiment = d["sentiment"]
        stance_cate = d["stance_cate"]

        orig_selected = d["aspect_span_texts"]
        # orig_tweet = d["text"]
        aspect_span = d["aspect_span"]
        # targets_start = d["targets_start"]
        # targets_end = d["targets_end"]
        left_offsets = d["left_offsets"]
        left_offsets = left_offsets.detach().numpy()
        # In eval_fn looks that it's equivalent to d["offsets"].detach().numpy() because it doesn't participate in the backprop
        # weight = d['weight']

        rightids = d["rightids"]
        right_token_type_ids = d["right_token_type_ids"]
        # right_offsets = d["right_offsets"].numpy()

        # 2022/08/08
        target_sbertEmb = d["sEmbeds"]

        leftids = leftids.to(device, dtype=torch.long)
        left_token_type_ids = left_token_type_ids.to(device, dtype=torch.long)
        rightids = rightids.to(device, dtype=torch.long)
        right_token_type_ids = right_token_type_ids.to(device, dtype=torch.long)
    
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        stance_cate = stance_cate.to(device, dtype=torch.long)
        
        # 2022/08/08
        target_sbertEmb = target_sbertEmb.to(device, dtype=torch.float32)
        # mask is not used here since the model will automatically provide the mask using the inbuilt tokenizer.
        # mask = mask.to(device, dtype=torch.long)
        # pos_logits_grt = pos_logits_grt.to(device, dtype=torch.long)
        
        # pos_logits_grt = pos_logits_grt.to(device, dtype=torch.float32)

        # location = location.to(device, dtype=torch.long)

        # targets_start = targets_start.to(device, dtype=torch.long)
        # targets_end = targets_end.to(device, dtype=torch.long)
        # weight = weight.to(device, dtype=torch.float)

        # print('original tweet')
        # print(orig_tweet[0])
        # print('ids')
        # print(ids[0])
        # print('offsets')
        # print(offsets[0])
        # print('selected_text')
        # print(orig_selected[0])
        # print('deciphered')
        # # Tokenizer is only employed for testing here.
        # print('1:', tokenizer.decode_ids([ids[0].tolist()[0]]))
        # print('2:', tokenizer.decode_ids([ids[0].tolist()[1]]))
        # print('3:', tokenizer.decode_ids([ids[0].tolist()[2]]))
        # print('4:', tokenizer.decode_ids([ids[0].tolist()[3]]))
        # print('targets_end token:', tokenizer.decode_ids([ids[0].tolist()[targets_end[0]]]))
        # # so should be until [targets_start, targets_end + 1], since i is added by enumerate, if there is one elem, then we shall call them [0: 0 + 1]
        # print('targets_start', targets_start[0])
        # print('targets_end', targets_end[0])

        model.zero_grad(set_to_none=True)
        # outputs_start, outputs_end = model(

        outputs_start, outputs_end, clss, usEmbeds, siameseclss = model(
            # ids=ids,
            left_tokens=leftids,
            left_token_type_ids=left_token_type_ids,
            right_tokens=rightids,
            right_token_type_ids=right_token_type_ids
        )

        # print('position_logits = model(')

        # trim tokens to the max len in a batch
        # Here, model.name acrually uses the selected_model, since in run_a_trial the model is initialized by name=selected_model
        # Seems that DataParallel object has no attribute 'name', it won't inherit the 'name' member
        # Not applicable when softmaxing over a padded sequence
        # ids, token_type_ids = trim_tensors(
        #     d["ids"], d["token_type_ids"], model_name
        # )

        # # Then each batch is trimmed to the longest sequence len (the length include special tokens, exclude pad tokens)
        # # The start logits is the first dim of the 2-dim logit
        # # The end logits is the second dim of the 2-dim logit
        # start_logits, end_logits = model(
        #     ids.cuda(), token_type_ids.cuda()
        # )

        # compute loss using position_logits and pos_logits_grt
        # print(position_logits.size())
        # print(pos_logits_grt.size())

        # if activation == 'sigmoid':
        #     position_logits = position_logits.squeeze(-1)
        
        # position_logits = position_logits.view(-1, 1)
        # pos_logits_grt = pos_logits_grt.view(-1, 1)
        # # makes no difference

        # 2022/08/08
        notorsame = d['notorsame'].to(device, dtype=torch.long)
        nli_instance_weight = d['instance_weight'].to(device, dtype=torch.float32)
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, clss, stance_cate, usEmbeds, target_sbertEmb, t_sigma, siameseclss, notorsame, nli_instance_weight, None)

        # loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, clss, stance_cate, usEmbeds, target_sbertEmb, t_sigma, None)
        # loss = loss_fn(position_logits, pos_logits_grt, None)
        # w4l = torch.where(pos_logits_grt > 0.5, 0 * pos_logits_grt + 1.2, pos_logits_grt * 0 + 0.8)
        # print(w4l.size())
        # print(pos_logits_grt.size())
        # loss = loss_fn(position_logits, pos_logits_grt, weight=w4l)

        # loss2 = loss_fn(position_logits, pos_logits_grt)
        # assert loss.equal(loss2)

        # This means meaning among batches?
        # This is redundant. It makes no difference whether adding this or not since loss is a scalar.
        # loss = loss.mean()

        # if config.gradient_accumulation_steps > 1:
        #     loss = loss / config.gradient_accumulation_steps

        # loss.backward()
        scaler.scale(loss).backward()
        
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        # # if (bi + 1) % config.gradient_accumulation_steps == 0:
        #     # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        # outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        # outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        # activation, be aware that this should be changed together with config and loss_fn, since currently
        #     we are using nn.BCEWithLogitsLoss
        # pos_logits_sgmd = None
        # if activation == 'sigmoid':
        #     pos_logits_sgmd = position_logits.sigmoid().cpu().detach().numpy()
        # else:
        #     print('Error: activation unspecified!')
        #     assert activation == 'sigmoid'
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        clss = torch.softmax(clss, dim=1).cpu().detach().numpy()
        siameseclss = torch.softmax(siameseclss, dim=1).cpu().detach().numpy()

        # # Only for test group 1
        # pos_logits_grt4test = pos_logits_grt.cpu().detach().numpy()
        # external_ids = d['external_id'] 
        # # Only for test group 1

        # # testing group 0424 1
        jaccard_scores = []
        jaccard_scores_old = []
        # # testing group 0424 1
        tps = []
        fns = []
        fps = []
        tps_se = 0
        fns_se = 0
        fps_se = 0
        tps_st = 0
        fns_st = 0
        fps_st = 0
        tps_siamese = 0
        fns_siamese = 0
        fps_siamese = 0
        # # testing group 0424 2
        # # This is batch-level actually, but meters are beyond the batch
        # bin_preds = []
        # bin_truths = []
        # # mf1_inb = MicroF1Meter()
        # # testing group 0424 2
        # jaccard_scores calculated in the tweet level, each jaccard_score features two sequences.
        for px, clean_text_a in enumerate(left_clean_text):
            # selected_tweet = orig_selected[px]
            # clean_text_a =  clean_text[px]
            # tweet_sentiment = sentiment[px]
            start_end_pair = []

            for start_index in np.argsort(outputs_start[px, :])[::-1][:20]:
                # Here, also the top20 end probabilities
                for end_index in np.argsort(outputs_end[px, :])[::-1][:20]:
                    if end_index < start_index:
                        continue
                    start_end_pair.append((start_index, end_index))
            if len(start_end_pair) == 0:
                # uncommented in 2022/08/08 to accommodate the idx_start=start_end_pair[0][0],
                #     IndexError: list index out of range
                # Error
                start_end_pair.append((-1, 0))
                # start_end_pair = []
                # Changed in 2022/08/08 to accommodate the idx_start=start_end_pair[0][0],
                #     IndexError: list index out of range
                # Error
                spans_pred = [(-1, 0)]
                # spans_pred = []
            else:
                idx_start = start_end_pair[0][0]
                idx_end = start_end_pair[0][1]
                spans_pred = [(idx_start, idx_end)]
            aspect_span_px = aspect_span[px]
            if aspect_span_px.find(':') == -1:
                aspect_span_px = '0:0'
            s_s, s_e = aspect_span_px.split(':')
            i_s = int(s_s)
            i_e = int(s_e)
            spans_grt = [(i_s, i_e)]

            jaccard_score_old, _ = calculate_jaccard_score_old(
                original_tweet=clean_text_a,
                target_string=orig_selected[px],
                idx_start=start_end_pair[0][0],
                idx_end=start_end_pair[0][1],
                # tokenizer=tokenizer,
                offsets=left_offsets[px],
                model_name=model_name
            )
            jaccard_scores_old.append(jaccard_score_old)

            jaccard_score = calculate_jaccard_score(
                clean_text_a,
                offsets=left_offsets[px],
                spans_pred=spans_pred,
                spans_grt=spans_grt
            )
            tp, fn, fp = calculate_TP_FN_FP_between_2lstsofintervals(
                clean_text_a,
                offsets=left_offsets[px],
                spans_pred=spans_pred,
                spans_grt=spans_grt
            )

            # Like multi-label, there's a third class meaning 'not making a guess'
            # Currently, each position is considered as a class, then if there's a missed prediction, then 
            # Actually spans_pred[0][0] is an index of token, spans_grts are indeed the real grts
            
            # !!! It may happen that the offsets contain space, but it's neganegible
            if left_offsets[px][spans_pred[0][0]][0] == spans_grt[0][0]:
                tps_se += 1
            else:
                fps_se += 1
                fns_se += 1
            if left_offsets[px][spans_pred[0][1]][1] == spans_grt[0][1]:
                tps_se += 1
            else:
                fps_se += 1
                fns_se += 1
            # # testing group 0424 1
            # jaccard_score = calculate_jaccard_score_between_2lstsofintervals(
            #     spans_pred=spans_pred,
            #     spans_grt=spans_grt
            # )
            jaccard_scores.append(jaccard_score)
            
            # # testing group 0424 1
 
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)

            stance_pred = np.argmax(clss[px, :])
            if stance_pred == stance_cate[px].item():
                tps_st += 1
            else:
                fns_st += 1
                fps_st += 1
            siamese_pred = np.argmax(siameseclss[px, :])
            if siamese_pred == notorsame[px].item():
                tps_siamese += 1
            else:
                fns_siamese += 1
                fps_siamese += 1

        # # testing group 0424 1
        jaccards.update(np.mean(jaccard_scores), leftids.size(0))
        jaccards_old.update(np.mean(jaccard_scores_old), leftids.size(0))
        # # testing group 0424 1

        # print(np.sum(tps), np.sum(fns), np.sum(fps))
        mf1.update(np.sum(tps), np.sum(fns), np.sum(fps), leftids.size(0))
        mf1_se.update(tps_se, fns_se, fps_se, leftids.size(0))
        mf1_st.update(tps_st, fns_st, fps_st, leftids.size(0))
        mf1_notorsame.update(tps_siamese, fns_siamese, fps_siamese, leftids.size(0))
        losses.update(loss.item(), leftids.size(0))
        # tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

        # testing group 0424 1
        tk0.set_postfix(loss=losses.avg, micro_F1=mf1.microF1, mF1_se=mf1_se.microF1, mf1_notorsame=mf1_notorsame.microF1, mF1_st=mf1_st.microF1)

        # if epc > 7 and random.random() < 0.3:
        #     optimizer.update_swa()
    # ========== Only for test group 1
    # After discovering all the exceptions, now the saved exception no longer needs output
    # fpOut.close()
    # ========== Only for test group 1


# If in the th's framework, there should be multiple configs, so the the config is in the outside,
#  configs -> config -> splits -> a split (or train-validation)
def run_a_trval(config, fold, dfx, tr, val, freeze_weight=None):
    df_train = dfx.iloc[tr]
    df_valid = dfx.iloc[val]

    # # Pseudo file is the extra data.
    # if config.PSEUDO_FILE:
    #     dfx_pseudo = pd.read_csv(config.PSEUDO_FILE)
    #     for i, (tr, val) in enumerate(skf.split(dfx_pseudo, dfx_pseudo.sentiment.values)):
    #         if i == fold:
    #             dfx_pseudo_train = dfx_pseudo.iloc[tr]
    #     # dfx_pseudo_train.score += 0.3
    #     df_train = pd.concat([df_train, dfx_pseudo_train])
    #     # pseudo_train_dataset = TweetDataset(
    #     #     tweet=dfx_pseudo_train.text.values,
    #     #     sentiment=dfx_pseudo_train.sentiment.values,
    #     #     selected_text=dfx_pseudo_train.selected_text.values,
    #     #     mode='train',
    #     #     weight=np.zeros(len(dfx_pseudo_train))
    #     # )
    #     #
    #     # pseudo_train_data_loader = torch.utils.data.DataLoader(
    #     #     pseudo_train_dataset,
    #     #     sampler=RandomSampler(pseudo_train_dataset),
    #     #     batch_size=config.TRAIN_BATCH_SIZE,
    #     #     num_workers=4
    #     # )

    set_seed(SEED)

    # tr_l = []
    # for i in tqdm(range(len(df_train))):
    #     line = df_train.iloc[i]
    #     tr_l += [process_training_data(line.text, line.selected_text, line.sentiment, config.TOKENIZER, config.MAX_LEN, 'train')]
    # val_l = []
    # for i in tqdm(range(len(df_valid))):
    #     line = df_valid.iloc[i]
    #     val_l += [
    #         process_training_data(line.text, line.selected_text, line.sentiment, config.TOKENIZER, config.MAX_LEN, 'train')]

    # # train_dataset = TweetDataset(
    # #     tweet=df_train.text.values,
    # #     sentiment=df_train.sentiment.values,
    # #     selected_text=df_train.selected_text.values,
    # #     mode='train',
    # #     weight=df_train.score.values,
    # #     # weight=df_train['v1.74.1_score'].values,
    # #     # shorter_tweet = df_train.shorter_text.astype(str).values,
    # # )

    # train_dataset = TweetDataset(
    #     data_l=tr_l
    # )
    # Create_tokenizer_and_tokens won't use MAX_LEN or config.max_len_val
    # Actually, the tokenizer, question, and tokens can be obtained from here. But I didn't use this and still use process_training_data
    tokenizer, special_tokens, precomputed_tokens_and_offsets = create_tokenizer_and_tokens(dfx, config)
    
    # tokens are different from df_test,
    # train_dataset = TweetTrainingDataset(df_test, tokenizer, tokens, max_len=config.max_len_val, model_name=config.selected_model, use_old_sentiment=config.use_old_sentiment)
    print('TweetTrainingDataset')

    train_dataset = TweetTrainingDataset(df_train, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=config.MAX_LEN, model_name=config.selected_model)

    # If sampler is specified, shuffle must not be specified
    # here the replacement=False by default, so the sampler will do the replacement.
    # RandomSampler according to source code <=> reshuffle, after each iteration it gives a new list [1:n]
    # If replacement is True, then you can sample m times and the next time it will sample from the remaining n-m instances

    train_sampler = RandomSampler(train_dataset)
    # Just for testing
    # train_sampler = SequentialSampler(train_dataset)

    # Seems that the author added a score column to manually add values (e.g., sentiment intensities), but the effect is unknown
    # weights = torch.FloatTensor(df_train.score.values)
    # # weights = torch.FloatTensor(df_train['score_modify_v1'].values)
    # Maybe weighted sampler will make the shortened instances more distinctive.
    # train_sampler = WeightedRandomSampler(weights, num_samples=int(len(df_train) * 0.6), replacement=True)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        # batch_size=config.TRAIN_BATCH_SIZE,
        batch_size=config.BATCH_SIZE_TR,
        num_workers=config.num_workers,
        # pin_memory=True will solve the bug. The theory is that if each element of your batch is a custom type, the pinning logic will not recognize them, and it will return that batch (or those elements) without pinning the memory
        #     Here, we have a list of string type, and this memory block will not be pinned. Therefore, in multiple process where num_workers > 1 there will be synchronization problems. Looks that they are going to create a new memory and waiting for the previous memory to be deleted, the signal has been blocked. Note that setting cuda_visible_devices=1 won't solve this problem.
        pin_memory=config.pin_memory
    )

    # valid_dataset = TweetDataset(
    #     tweet=df_valid.text.values,
    #     sentiment=df_valid.sentiment.values,
    #     selected_text=df_valid.selected_text.values,
    #     weight=df_train.score.values
    # )

    # # This is the wrap for the validation dataset, no sampler needed
    # valid_dataset = TweetDataset(
    #     data_l=val_l
    # )
    # max_len no longer takes MAX_LEN
    # In the testing set pre-computed ids and offsets will be pre-loaded by features, or by specifying 
    #       unique and obtain the should-computed one

    valid_dataset = TweetTrainingDataset(df_valid, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=config.MAX_LEN, model_name=config.selected_model)

    # SequentialSampler, No shuffling, and in one direction in iteration.
    eval_sampler = SequentialSampler(valid_dataset)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=eval_sampler,
        # batch_size=config.VALID_BATCH_SIZE,
        batch_size=config.batch_size_val,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # .cuda will use the default singla GPU device.
    # torch.nn.DataParallel() will by default use all the available device, defined in os.environ['CUDA_VISIBLE_DEVICES']
    # The Theo's TweetQATransformer will automatically decide which LM configuration to load, so no need to load the language model from the outside
    # def __init__(self, model, nb_layers=1, nb_ft=None, sentiment_ft=0, multi_sample_dropout=False, use_squad_weights=True)
    # use_squad_weights is never used
    print('DOCTransformer')
    model = DOCTransformer(
        config.selected_model,
        config.is_automodel,
        nb_layers=config.nb_layers,
        nb_ft=config.nb_ft,
        nb_class=config.nb_class,
        pretrained=config.pretrained,
        nb_cate=config.nb_cate,
        nb_sbert=config.nb_sbert,
        multi_sample_dropout=config.multi_sample_dropout,
        training=True
    )
    # .cuda() This is duplicated

    device = torch.device("cuda")
    # model_config = transformers.RobertaConfig.from_pretrained(config.BERT_PATH)
    # model_config.output_hidden_states = True
    # model = TweetModel(conf=model_config)


    model.to(device)
    if freeze_weight:
        model.load_state_dict(torch.load(freeze_weight))
    print('model.to(device)')
    # num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE // config.gradient_accumulation_steps * config.EPOCHS)
    # number_of_batches * EPOCHS
    # Not going to use // config.gradient_accumulation_steps since it's actually 1 in Hiki
    num_train_steps = int(len(df_train) / config.BATCH_SIZE_TR * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.LR)

    # num_warmup_steps=num_train_steps * 0.05, means that in first 0.05 * EPOCH we do the warmup and in the later 0,95 we do the lr_decay
    scheduler = None
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_steps // 20,
        # num_warmup_steps=num_train_steps // 100,
        num_training_steps=num_train_steps
    )
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=num_train_steps, num_cycles=0.5
    # )
    # scheduler = get_my_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_train_steps // 20,
    #     num_linear_steps=int(len(df_train) / config.TRAIN_BATCH_SIZE // config.gradient_accumulation_steps * 7),
    #     num_training_steps=num_train_steps,
    #     num_cycles=6
    # )
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps=num_train_steps // 20,
    #                                            num_training_steps=num_train_steps,
    #                                            num_cycles=0.5
    #                                            )

    # swa
    # optimizer = SWA(optimizer)
    # optimizer = SWA(optimizer, swa_start=num_train_steps // 20, swa_freq=5, swa_lr=1e-5)
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps=num_train_steps // 20,
    #                                            num_training_steps=num_train_steps,
    #                                            num_cycles=3.0
    #                                            )

    # freeze_opt = AdamW(optimizer_parameters, lr=0.001)
    # model.freeze()
    # for epoch in range(3):
    #     train_fn_freeze(train_data_loader, model, freeze_opt, device)
    # model.unfreeze()

    # The amp is not necessary
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    #  device_ids (list of python:int or torch.device) ?C CUDA devices (default: all devices)
    #  So here we use all the visible devices defined in the os.environ['CUDA_VISIBLE_DEVICES']
    model = torch.nn.DataParallel(model)

    # This self-defined class handle the early stopping state
    es = EarlyStopping(patience=3, mode="max", delta=0.0005)
    print(f"Training is Starting for fold={fold}")

    set_seed(SEED)

    for epoch in range(config.EPOCHS):
        # Each epoch's training.
        print('====== Start entering train_fn of epoch %d' % (epoch))
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler, epc=epoch, tokenizer=tokenizer, model_name=config.selected_model, activation=config.activation)
        print('train_fn')
        # ========== Only for test group 1
        # assert False
        # ========== Only for test group 1
    # # ========== Need to uncomment this
        # if epoch > 7:
        #     optimizer.swap_swa_sgd()
        # see inference_roberta*.py, it's the token-level start-end that is output during training, it's the char-level start-end that was output in testing
        # Be aware that the model output is un-sgmed since we want to use BCDE loss
        # jaccard, pos_logits_sgmd = eval_fn(valid_data_loader, model, device, model_name=config.selected_model, activation=config.activation)
        # micro_f1, pos_logits_sgmd = eval_fn(valid_data_loader, model, device, model_name=config.selected_model, activation=config.activation)
        jaccard, startoof, endoof, notorsame_mF1, ng_lg_normal = eval_fn(valid_data_loader, model, device, model_name=config.selected_model, activation=config.activation)
        print(f"Jaccard Score = {jaccard}, Notorsame_mF1 = {notorsame_mF1}, Negative Log Normal Score = {ng_lg_normal}")
        # print(f"micro_f1 Score = {micro_f1}")
        # saving as .pt actually is the same as saving as .bin, because they both use .save, and actually the .save used .pickle
        # es(micro_f1, model, model_path=f"{config.out_path}/ckpt_{config.selected_model}_{fold}.bin", pos_logits_sgmd=pos_logits_sgmd)
        # es(jaccard, model, model_path=f"{config.out_path}/ckpt_{config.selected_model}_{fold}.bin", start_oof=startoof, end_oof=endoof)
        # es(notorsame_mF1, model, model_path=f"{config.out_path}/ckpt_{config.selected_model}_{fold}.bin", start_oof=startoof, end_oof=endoof)
        es(ng_lg_normal, model, model_path=f"{config.out_path}/ckpt_{config.selected_model}_{fold}.bin", start_oof=startoof, end_oof=endoof)
        # es(jaccard, model, model_path=f"../Output/ckpt_albert-large_{fold}.bin", start_oof=startoof, end_oof=endoof)
        # if epoch > 7:
        #     optimizer.swap_swa_sgd()
        if es.early_stop:
            print("Early stopping")
            break
    del model, optimizer
    # # =========== Need to uncomment this
    gc.collect()
    # # =========== Need to uncomment this
    # return es.best_score, len(valid_dataset), es.best_pos_logits_sgmd
    return es.best_score, len(valid_dataset), es.best_start, es.best_end
    # # =========== Need to uncomment this


def predict(model, dataset, batch_size=32, num_workers=1, activation='sigmoid'):
    # Unlike mR1 and similar to mytrain_alvert_k_fold_was_about_to_build_the_model, the predict returns token-level pos_logits_sgmd
    # Alright, now I have switched to the mRSB1 scheme
    # Alright, looks that we cannot switch to the mR1 scheme, since we are using k-fold
    # model is running in the evaluation mode
    model.eval()
    pred_clss = []
    start_probas = []
    end_probas = []
    pred_sEmbs = []
    # offsets = []

    # batch_size=32
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # model.name = config.selected_model, which is defined by author here
    # config.name is only used in main, which is also defined by author here.
    with torch.no_grad():
        for data in loader:
            # From here you can see that trim_tensors actually reduces the computation
            # See " => # From here you can see that trim_tensors actually reduces the computation"
            # need improvement: so eval_fn better uses trim_tensors as well
            ids, token_type_ids = trim_tensors(
                data["leftids"], data["left_token_type_ids"], model.name
            )
            # Then each batch is trimmed to the longest sequence len (the length include special tokens, exclude pad tokens)
            # The start logits is the first dim of the 2-dim logit
            # The end logits is the second dim of the 2-dim logit
            # start_logits, end_logits, left_clss, usbert, siamesecls
            # outputs_start, outputs_end, clss, usEmbeds = model(
            outputs_start, outputs_end, clss, usEmbeds, siamesecls = model(
                ids.cuda(), token_type_ids.cuda(), data["rightids"].cuda(), data["right_token_type_ids"].cuda()
            )
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            clss = torch.softmax(clss, dim=1).cpu().detach().numpy()
            usEmbeds = usEmbeds.cpu().detach().numpy()

            # for s, e in zip(start_probs, end_probs):
            #     start_probas.append(list(s))
            #     end_probas.append(list(e))
            
            start_probas.extend(list(outputs_start))
            end_probas.extend(list(outputs_end))
            pred_clss.extend(list(clss))
            # 2022/08/08 each elem is an array
            pred_sEmbs.extend(list(usEmbeds))

    # return start_probas, end_probas, pred_clss, pred_sEmbs
    return pred_sEmbs
    # , offsets


def k_fold_inference(config, test_dataset, seed=42):
    # k depends on the number of .bin files in the folder
    # Need the k_fold_inference otherwise the prediction would be based on only one train-val split
    # get_tokenizer is equivalent to create_tokenizer_and_tokens, and currently we don't have plan to use auto_tokenizers, otherwise we might add the path to the tokenizer to the config
    #       Another difference is that we need to compute the precomputed tokens dynamically here since we did't store it during the training
    #       But we can still create it from unique
    # The exp_folder, i.e., the path to the model, is stored in the config.weights

    seed_everything(seed)

    pred_tests = []
    # This means that the DOCTransformer model has been trained and saved k times.
    #  (See forward there) and in each time in the forward the Linear has been stacked for 5 times using the same model weight.
    print('config.nb_layers in {} is {}'.format(config.selected_model, config.nb_layers))
    print('and the config.name in %s is %s' % (config.selected_model, config.name))
    for weight in config.weights:
        # define an empty model
        # And the weights are loaded to cpu first then on cuda.
        # config.selected_model become the model name.
        # config.name is also defined by the author but is only used in main
        # use_squad_weights is never used
        model = DOCTransformer(
            config.selected_model,
            config.is_automodel,
            nb_layers=config.nb_layers,
            nb_ft=config.nb_ft,
            nb_class=config.nb_class,
            pretrained=config.pretrained,
            nb_cate=config.nb_cate,
            nb_sbert=config.nb_sbert,
            multi_sample_dropout=config.multi_sample_dropout,
        ).cuda()

        model = load_model_weights(model, weight, cp_folder=config.weights_path, verbose=1)

        # it printed "True", so the model is running on cuda, even without "strict" pre-defined.
        # For the strict parameter, see https://discuss.pytorch.org/t/does-model-load-state-dict-strict-false-ignore-new-parameters-introduced-in-my-models-constructor/84539
        # It doesn't have effects on the weights. Just an indicator to decide whether to check if the weights are completely the same.
        # print(next(model.parameters()).is_cuda)

        model.zero_grad(set_to_none=True)

        # (start_probas, end_probas)
        # pred_test_afold = predict(model, test_dataset, batch_size=config.batch_size_val, num_workers=config.num_workers, activation=config.activation)
        # platform suggests num_workers=2
        pred_test_afold = predict(model, test_dataset, batch_size=config.batch_size_val, num_workers=2, activation=config.activation)
        # print(len(pred_test_afold))
        pred_tests.append(pred_test_afold)
    # So number of weights should be k (which is coincidently 5 here)
    # Later on in get_char_preds the k predictions will be summed and meaned.
    # len(pred_tests[0]) is 2,for start_probas, end_probas respectively
    print('nb of weights', len(pred_tests))
    print('nb of instances', len(pred_tests[0]))
    return pred_tests

class ConfigAlbert:
    # Architecture
    # selected_model = "albert-xxlarge-v2"
    # selected_model = "albert-large-v2"
    # selected_model = "deberta-v3-large"
    selected_model = "deberta-v3-large-auto"
    # pretrained = False
    pretrained = True
    local_pretrained = False
    lowercase = True
    nb_layers = 8
    nb_ft = 120
    # equivalent to num_classes in mR1
    # nb_class = 1
    nb_class = 2
    nb_cate = 3
    nb_sbert = 768
    multi_sample_dropout = True
    use_old_sentiment = False

    # training
    activation = "sigmoid"

    # see new_max_len = max_len - 3 - len(feature_text_ids)
    # The MAX_LEN considers the feature_text len, it's also used in validation but not in test
    # MAX_LEN = 352
    # MAX_LEN = 365
    # MAX_LEN = 400
    # MAX_LEN = 390
    # MAX_LEN = 375
    # MAX_LEN = 326
    # MAX_LEN = 326
    # MAX_LEN = 120
    # MAX_LEN = 420
    MAX_LEN = 300
    # BATCH_SIZE_TR = 16 # 16 best
    BATCH_SIZE_TR = 16
    # BATCH_SIZE_TR = 4
    # BATCH_SIZE_TR = 8
    # BATCH_SIZE_TR = 6
    # BATCH_SIZE_TR = 14
    # EPOCHS = 10 # 5-fold-22040706
    # EPOCHS = 16 # 5-fold-22040713
    # EPOCHS = 20 # 5-fold-2204080024
    # EPOCHS = 30 # 5-fold-2204090321
    # EPOCHS = 25 # 5-fold-2204091829
    # EPOCHS = 16 # 5-fold-2204112147 with clean_spaces changed by adding re.sub('\s+$', '', txt)
    EPOCHS = 10
    LR = 2e-5
    # LR = 3e-5
    # LR = 2e-11

    # Inference, it's also used in validation
    # batch_size_val = 80
    batch_size_val = 16
    # 2: 2h; 4: 1h, needs to be 4
    # it's only used in test
    # need improvement 5, not sure changing this from 310 to longer would affect the accuracy
    # max_len_val = 310
    # max_len_val = 326 # 386 won't trigger an error, and looks that 386 won't affect the performance
    # if max_len_val too low it will cause just for test round 4, 1 error; if too long, it will cause cuda-menory capacity overflow error
    
    # max_len_val = 120 # 348 OK peroridically # 346 # 366 # 386 # 366 OK 346 Error 356 OK, 310 OK with batchsize 4; 356 error with batchsize 4; 350 OK with batchsize 2; 350 OK with batchsize 4; 348 OK with batchsize 4;
    # max_len_val = 300
    max_len_val = 300
    # Found the cause for block: num_workers >= 2. Need the search through the code to fix num_workers to 1.
    # The detailed reason: 
    # num_workers = 8
    num_workers = 16
    # num_workers = 2
    pin_memory = True

    # original weights_path
    # out_path = f"../input/doccheckpoints/5-fold-220804/{selected_model}"
    out_path = f"../input/doccheckpoints/5-fold-220730/{selected_model}"
    # weights_path = f'../input/doccheckpoints/5-fold-220804/{selected_model}'
    weights_path = f'../input/doccheckpoints/5-fold-220730/{selected_model}'
    # So there are 5 pt files, totally depends on the number of the saved file numbers
    weights = None
    # name = 'albert-xxlarge-v2-squad'
    # name = 'albert-large-v2-squad'
    # name = "deberta-v3-large-squad"
    name = "deberta-v3-large-auto-squad"
    is_automodel = True
    # squad albert-large-v2-squad, never used
    def __init__(self):
        if os.path.exists(self.weights_path):
            # self.weights = sorted([f for f in os.listdir(self.weights_path) if "albert" in f])
            # self.weights = sorted([f for f in os.listdir(self.weights_path) if "albert-xxlarge-v2" in f])
            # self.weights = sorted([f for f in os.listdir(self.weights_path) if "albert-large-v2" in f])
            self.weights = sorted([f for f in os.listdir(self.weights_path) if "deberta-v3-large-auto" in f])

def train():
    # ============ Begins the training and testing script
    # Another problem caused by num_workers > 1 causing 
    # solution: "# I have the same problem, the following code can solve it
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103
    # "
    
    # torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')

    seed_everything(SEED)
    # They used ".cuda()" around 400+, so the GPU can only be restricted here.
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    # If this value is not defined, then cuda use the first available GPU, or all the GPU in the parallel.
    # remember to set os.environ['NCCL_SOCKET_IFNAME'] = 'lo' if we are dealing with multiple GPUs
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3,5"    
    # In this case you should make the batch_size 2* 10 or 2 * 3 * 10
    # os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    # default value is all the GPU.
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    # df_test = pd.read_csv(DATA_PATH + 'test.csv').fillna('')

    # # ========== This is for training

    df_train = load_and_prepare_train(root=DATA_PATH)

    # df_train = pd.read_csv(DATA_PATH + 'train_original.csv').fillna('')

    # print(df_train.iloc[0])
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=777)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
    # skf_g = GroupKFold(n_splits=5)
    # doesn't change the order in a split, only provides the index
    # copied from score_fold, cnt, start_logits_fold, end_logits_fold = run(i, dfx, tr, val)
    # MAX_LEN is taken from the config.max_len_val, since in the TweetTestDataset max_len also uses config.max_len_val
    # Here I customize MAX_LEN

    val_score = 0
    configs = [
        # ConfigDistil(),
        # ConfigBertBase(),
        # ConfigBertWWM(),
        ConfigAlbert(),
    ]
    config = configs[0]
    # This is testing, so should be the validation set max_len

    # for i, (tr, val) in enumerate(skf.split(df_train, df_train.values)):
    # for i, (tr, val) in enumerate(skf.split(df_train, df_train.values)):
    # X, y, grouped_value for GroupKFold
    # or X, y for StratifiedKFold
    # for i, (tr, val) in enumerate(skf_g.split(df_train, df_train, df_train.values)):
    # for i, (tr, val) in enumerate(skf_g.split(df_train, df_train, df_train.values)):

    for i, (tr, val) in enumerate(skf.split(df_train, df_train.stance)):
    # for i, (tr, val) in enumerate(skf.split(df_train, df_train.location)):
        print('Training in fold %d!' % (i))
        # if i != 3:
        #     continue
        # # score_fold, cnt, start_logits_fold, end_logits_fold = run_a_trval(config, MAX_LEN, i, dfx, tr, val)
        # score_fold, cnt, start_logits_fold, end_logits_fold = run_a_trval(config, i, df_train, tr, val)
        score_fold, cnt, start_logits_fold, end_logits_fold = run_a_trval(config, i, df_train, tr, val)
        # # score_fold, cnt, start_logits_fold, end_logits_fold = run(i, dfx, tr, val)
        val_score += score_fold


def clustering():

    # ====================== For inference
    # In training, the ,,,neutral training instance has been deleted.
    preds = {}
    df_test = load_and_prepare_test_left_uniqued(root=DATA_PATH)
    configs = [
        # ConfigDistil(),
        # ConfigBertBase(),
        # ConfigBertWWM(),
        ConfigAlbert(),
    ]
    for config in configs:
        # So this is the config for albert, since there is only one config
        print(f'\n   -  Doing inference for {config.name}\n')

        tokenizer, special_tokens, precomputed_tokens_and_offsets = create_tokenizer_and_tokens(df_test, config)
        # # tokens are different from df_test, No worries the testing set will left join with feature_text, so unique also covers all the feature_text, and we are going create the dataset inside the k_fold_inference
        # # Alright, looks that we need to put the TweetTestDataset outside, nevertheless, we still use create_tokenizer_and_tokens to dynamically computed feature_text tokens instead of loading them from the saved
        
        # test_dataset = TweetTestDataset(df_test, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=config.max_len_val, model_name=config.selected_model)
        # # pred_test = single_fold_inference(
        # # it's 5-fold pos_logits_sgmd
        # pred_tests = k_fold_inference(
        #     config,
        #     test_dataset,
        #     seed=SEED,
        # )

        train_dataset = TweetTrainingDataset(df_test, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=config.max_len_val, model_name=config.selected_model)
        # pred_test = single_fold_inference(
        # it's 5-fold pos_logits_sgmd
        # print(f'dataset_len = {len(train_dataset)}')

        pred_tests = k_fold_inference(
            config,
            train_dataset,
            seed=SEED,
        )

    # # ===================== # baseline
    # embedder = SentenceTransformer('all-mpnet-base-v2')

    # # # ////////////////////////
    # # # Corpus with testing sentences
    # filename = '../input/datasets/VAD_test_r.csv'
    # data = csv.reader(
    #     open(filename, encoding="utf-8"),
    #     delimiter=',', lineterminator='\n',
    #     quoting=csv.QUOTE_MINIMAL)
    # row0 = next(data)
    # # rowid2row = [row for i, row in enumerate(data) if i >= 260]
    # # rowid2row = [row for i, row in enumerate(data) if i >= 370]
    # rowid2row = [row for i, row in enumerate(data)]
    # corpus = [row[1] for row in rowid2row]
    # corpus_embeddings = embedder.encode(corpus) # baseline
    # # # \\\\\\\\\\\\\\\\\\\\\\\\\
    # # =======================
    corpus = list()
    rowid2row = list()
    for id_index, df_row in df_test.reset_index().iterrows():
        # index is not reset yet before this loop
        # twitterID, text,stance,aspect_span,opinion_span,aspect_catetegory
        # ID,lefttext,righttext,notorsame,stance,aspect_span,opinion_span,left_aspect_category,right_aspect_category
        row = [df_row['ID'], df_row['lefttext'], '', '', '', df_row['left_aspect_category']]
        corpus.append(row[1])
        rowid2row.append(row)

    corpus_embeddings = pred_tests[0]

    all_the_sentence_reps_all = None

    num_clusters = 8

    def permutation(lst):
        # If lst is empty then there are no permutations
        if len(lst) == 0:
            return []
        # If there is only one element in lst then, only
        # one permutation is possible
        if len(lst) == 1:
            return [lst]
        # Find the permutations for lst if there are
        # more than 1 characters
        
        l = [] # empty list that will store current permutation
        
        # Iterate the input(lst) and calculate the permutation
        for i in range(len(lst)):
            m = lst[i]
        
            # Extract lst[i] or m from the list.  remLst is
            # remaining list
            remLst = lst[:i] + lst[i+1:]
        
            # Generating all permutations where m is first
            # element
            for p in permutation(remLst):
                l.append([m] + p)
        return l

    def fit_cluster_iter():
        # Perform kmean clustering
        # Cluster twice, with different # of clusters, for different separation
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        lst_texted_indices = []

        all_the_sentence_reps = corpus_embeddings

        grd_assign = []
        pred_assign = []
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            tweet = corpus[sentence_id]
            row = rowid2row[sentence_id]
            tweet_id = row[0]
            # print(row)
            # grd_assign.append(row[5])
            # pred_assign.append()
            clustered_sentences[cluster_id].append([row[0], tweet, int(row[5])])
            lst_texted_indices.append((sentence_id, cluster_id, tweet, tweet_id))

        # 1376217045474377728: 4
        # 1368271968894554114: 4
        dct_clusternum2rightkey = {}

        cluster_permutation = permutation([i for i in range(num_clusters)])

        lst_cluster_pred = []
        lst_cluster_grt = []
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            tweet = corpus[sentence_id]
            row = rowid2row[sentence_id]
            tweet_id = row[0]
            # print(row)
            # grd_assign.append(row[5])
            # pred_assign.append()
            
            grd_cluster_id = int(row[5]) % 8
            lst_cluster_grt.append(grd_cluster_id)
            lst_cluster_pred.append(cluster_id)

        best_permutation = None
        best_acc = 0
        best_f1_mac = 0
        for a_permutation in cluster_permutation:
            rec_pred_lst = list(map(lambda x: a_permutation[x], lst_cluster_pred))
            f1_mac = metrics.f1_score(lst_cluster_grt, rec_pred_lst, average='macro')
            acc = metrics.accuracy_score(lst_cluster_grt, rec_pred_lst)
            if acc > best_acc:
                best_acc = acc
                best_f1_mac = f1_mac
                best_permutation = a_permutation

        print('best_acc', best_acc, 'best_f1_mac', best_f1_mac)
        dct_clusternum2rightkey = best_permutation
        return best_acc, dct_clusternum2rightkey, lst_cluster_grt, lst_cluster_pred, all_the_sentence_reps, cluster_assignment

    best_acc_all = 0
    dct_clusternum2rightkey_all = None
    lst_cluster_grt_all = None
    lst_cluster_pred_all = None
    best_cluster_assignment = None

    best_acc, dct_clusternum2rightkey, lst_cluster_grt, lst_cluster_pred, all_the_sentence_reps, curr_cluster_assignment = fit_cluster_iter()

if __name__ == '__main__':
    # train()
    clustering()