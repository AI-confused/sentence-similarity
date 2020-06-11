from __future__ import absolute_import
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
import itertools
from multiprocessing import Pool
import functools
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from typing import Callable, Dict, List, Generator, Tuple
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import json
import math
from bert_utils import load_and_cache_examples, set_seed, get_f1
from itertools import cycle
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_language", default=None, type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    ## Other parameters
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--tokenizer_name", default=None, type=str)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--from_tf', action='store_true',
                        help='whether load tensorflow weights')
    parser.add_argument("--do_eval_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")  
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args)
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    logger.info("Training/evaluation parameters %s", args)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # Training
    if args.do_train:
        # Prepare model
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=args.from_tf, config=config)
        # fgm = FGM(model)
        model.to(args.device)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        
        train_dataset = load_and_cache_examples(args, tokenizer, is_training=1)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        best_acc=0
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0        
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)

        output_dir = args.output_dir + "eval_results_{}_{}_{}_{}_{}_{}".format(
                                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                str(args.max_seq_length),
                                str(args.learning_rate),
                                str(args.train_batch_size),
                                str(args.train_language),
                                str(args.train_steps))
        try:
            os.makedirs(output_dir)
        except:
            pass
        output_eval_file = os.path.join(output_dir, 'eval_result.txt')
        with open(output_eval_file, "w") as writer:
            writer.write('*' * 80 + '\n')
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
             # 正常训练
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)[0]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loss.backward()
            # 对抗训练
            # fgm.attack() # 在embedding上添加对抗扰动
            # loss_adv = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            # if args.n_gpu > 1:
            #     loss_adv = loss_adv.mean() # mean() to average on multi-gpu.
            # if args.gradient_accumulation_steps > 1:
            #     loss_adv = loss_adv / args.gradient_accumulation_steps
            # loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore() # 恢复embedding参数
            

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
#                 scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1


            if (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0 
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))


            if args.do_eval and (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                if args.do_eval_train:
                    file_list = ['train.csv','dev.csv']
                else:
                    file_list = ['dev.csv']
                for file in file_list:
                    inference_labels=[]
                    gold_labels=[]
                    inference_logits=[]
                    dev_dataset = load_and_cache_examples(args, tokenizer, is_training=2)
                    dev_sampler = SequentialSampler(dev_dataset)
                    dev_dataloader = DataLoader(
                        dev_dataset, 
                        sampler=dev_sampler, 
                        batch_size=args.eval_batch_size)
                        
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(dev_dataset))
                    logger.info("  Batch size = %d", args.eval_batch_size)  
                    

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids in tqdm(dev_dataloader):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)


                        with torch.no_grad():
                            tmp_eval_loss, logits= model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1
                        
                    gold_labels=np.concatenate(gold_labels,0) 
                    inference_labels=np.concatenate(inference_labels,0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = get_f1(inference_labels, gold_labels)

                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'train_loss': train_loss}

                    if 'dev' in file:
                        with open(output_eval_file, "a") as writer:
                            writer.write(file+'\n')
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))
                            writer.write('*'*80)
                            writer.write('\n')
                    if eval_accuracy>best_acc and 'dev' in file:
                        print("="*80)
                        print("Best ACC",eval_accuracy)
                        print("Saving Model......")
                        best_acc=eval_accuracy
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("="*80)
                    else:
                        print("="*80)
        with open(output_eval_file, "a") as writer:
            writer.write('bert_acc: %f'%best_acc)
    
    if args.do_test:
        if args.do_train == False:
            output_dir = args.output_dir
        
        model = model_class.from_pretrained(output_dir, config=config)
        model.to(args.device)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)        
        
        inference_labels=[]
        gold_labels=[]
        
        dev_dataset = load_and_cache_examples(args, tokenizer, is_training=3)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(
            dev_dataset, 
            sampler=dev_sampler, 
            batch_size=args.eval_batch_size)
        
        logger.info(" *** Run Prediction ***")
        logger.info("  Num examples = %d", len(dev_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)  

        model.eval()
        for input_ids, input_mask, segment_ids, label_ids in tqdm(dev_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)[1].detach().cpu().numpy()
            inference_labels.append(np.argmax(logits, axis=1))
            label_ids = label_ids.to('cpu').numpy()
            gold_labels.append(label_ids)
            
        gold_labels=np.concatenate(gold_labels,0) 
        logits = np.concatenate(inference_labels,0)
        test_f1 = get_f1(logits, gold_labels)
        logger.info('predict f1:{}'.format(str(test_f1)))
            
if __name__ == "__main__":
    main()
