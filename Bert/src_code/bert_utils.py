from sklearn.metrics import f1_score
import argparse
import csv
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import time
import collections
import torch.nn as nn
from collections import defaultdict
import gc
from tqdm import tqdm
import itertools
from multiprocessing import Pool
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
import functools

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label,

    ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        
        
def load_and_cache_examples(args, tokenizer, is_training):
    # Load data features from cache or dataset file
    if is_training==1:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(args.train_language),
                ),
        )   
    elif is_training==2:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(args.train_language),
                ),
        )
    else:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "predict",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(args.train_language),
                ),
        )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if is_training==1:
            examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training)
        elif is_training==2:
            examples = read_examples(os.path.join(args.data_dir, 'dev.csv'), is_training)
        else:
            examples = read_examples(os.path.join(args.data_dir, 'test.csv'), is_training)
        features = convert_examples_to_features(
            examples, tokenizer, args.max_seq_length, is_training)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
        
        
def read_examples(input_file, is_training):
    df=pd.read_csv(input_file)
    if is_training==1 or is_training==2:
        examples=[]
        for val in tqdm(df[['id','query0','query1','label']].values, desc="read squad examples"):
            examples.append(InputExample(guid=val[0],text_a=val[1],text_b=val[2],label=val[3]))
    else:
        examples=[]
        for val in tqdm(df[['id','query0','query1', 'label']].values, desc="read squad examples"):
            examples.append(InputExample(guid=val[0],text_a=val[1],text_b=val[2], label=val[3]))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples), desc="convert squad examples to features"):

        query1_tokens=tokenizer.tokenize(example.text_a)
        query2_tokens=tokenizer.tokenize(example.text_b)

        _truncate_seq_pair(query1_tokens, query2_tokens, max_seq_length - 3)
        tokens = ["[CLS]"]+ query1_tokens + ["[SEP]"] + query2_tokens  + ["[SEP]"]
        segment_ids = [0] * (len(query1_tokens) + 2) + [1] * (len(query2_tokens) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)


        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        
        # if is_training==1 or is_training==2:
        label = example.label
        # else:
        #     label = 0
        if example_index < 1 and is_training==1:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581','_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            logger.info("label: {}".format(label))


        features.append(
            InputFeatures(
                example_id=example.guid,
                input_ids = input_ids,
                input_mask = input_mask,
                segment_ids = segment_ids,
                label = label,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
        
def get_f1(preds, labels):
    return f1_score(labels, preds, labels=[0,1], average='macro')
    
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
