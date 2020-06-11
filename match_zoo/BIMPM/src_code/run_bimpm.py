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
from model import BIMPM
from utils import ATEC_Dataset, load_embeddings, set_seed, get_f1
from itertools import cycle
from transformers import AdamW

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--embeddings_file", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--train_language", default=None, type=str, required=True)
    parser.add_argument("--train_steps", default=-1, type=int, required=True)
    parser.add_argument("--eval_steps", default=-1, type=int, required=True)
    parser.add_argument("--load_word2vec", action='store_true', help='if true, load word2vec file for the first time; if false, load generated word-vector csv file')
    parser.add_argument("--generate_word2vec_csv", action='store_true', help='if true, generate word2vec csv file')
    ## normal parameters
    parser.add_argument("--embedding_size", default=300, type=int)
    parser.add_argument("--query_maxlen", default=30, type=int)
    parser.add_argument("--hidden_size", default=300, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=10, type=int)       
    parser.add_argument("--per_gpu_train_batch_size", default=10, type=int)  
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int) 
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # device = torch.device("cpu")
    args.device = device
    
    # Set seed
    set_seed(args)
    
    logger.info("Training/evaluation parameters %s", args)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)    
    
    # Training
    if args.do_train:
        # build model
        logger.info("*** building model ***")
        embeddings = load_embeddings(args)
        model = BIMPM(embeddings=embeddings, hidden_size=args.hidden_size, class_size=args.num_classes, device=args.device)
        model.to(args.device)
                
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        
        logger.info("*** Loading training data ***")
        train_data = ATEC_Dataset(os.path.join(args.data_dir, 'train.csv'), os.path.join(args.data_dir, 'vocab.csv'), args.query_maxlen)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
        
        logger.info("*** Loading validation data ***")
        dev_data = ATEC_Dataset(os.path.join(args.data_dir, 'dev.csv'), os.path.join(args.data_dir, 'vocab.csv'), args.query_maxlen)
        dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.eval_batch_size)
        
        num_train_optimization_steps =  args.train_steps

        # 过滤出需要梯度更新的参数
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                           # factor=0.85, patience=0)
        criterion = nn.CrossEntropyLoss()
        
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        best_acc=0
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0        
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_loader=cycle(train_loader)

        output_dir = args.output_dir + "eval_results_{}_{}_{}_{}_{}_{}".format(
                                'BIMPM',
                                str(args.query_maxlen),
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
            batch = next(train_loader)
            batch = tuple(t.to(device) for t in batch)
            q1, _, q2, _, labels = batch
             # 正常训练
            optimizer.zero_grad()
            logits, probs = model(q1, q2)
            loss = criterion(logits, labels)
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += q1.size(0)
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
                # scheduler.step()
                optimizer.step()
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
                        
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(dev_data))
                    logger.info("  Batch size = %d", args.eval_batch_size)  
                    

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for q1, _, q2, _, labels in tqdm(dev_loader):
                        with torch.no_grad():
                            logits, probs = model(q1.to(args.device), q2.to(args.device))
                        tmp_eval_loss = criterion(logits, labels.to(args.device))
                        probs = probs.detach().cpu().numpy()
                        inference_labels.append(np.argmax(probs, 1))
                        gold_labels.append(labels)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += logits.size(0)
                        nb_eval_steps += 1
                        
                    gold_labels=np.concatenate(gold_labels,0) 
                    inference_labels=np.concatenate(inference_labels,0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = get_f1(inference_labels, gold_labels)

                    result = {
                              'eval_loss': eval_loss,
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
        
        # build model
        logger.info("*** building model ***")
        embeddings = load_embeddings(args)
        model = BIMPM(embeddings=embeddings, hidden_size=args.hidden_size, class_size=args.num_classes, device=args.device)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        model.to(args.device)
                 
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)      
        
        inference_labels=[]
        gold_labels=[]
        
        logger.info("*** Loading testing data ***")
        dev_data = ATEC_Dataset(os.path.join(args.data_dir, 'test.csv'), os.path.join(args.data_dir, 'vocab.csv'), args.query_maxlen)
        dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.eval_batch_size)
        
        logger.info(" *** Run Prediction ***")
        logger.info("  Num examples = %d", len(dev_data))
        logger.info("  Batch size = %d", args.eval_batch_size)  

        model.eval()
        for q1, _, q2, _, labels in tqdm(dev_loader):
            with torch.no_grad():
                logits, probs = model(q1, q2)
            probs = probs.detach().cpu().numpy()
            inference_labels.append(np.argmax(probs, 1))
            gold_labels.append(labels)
            
        gold_labels=np.concatenate(gold_labels,0) 
        logits = np.concatenate(inference_labels,0)
        test_f1 = get_f1(logits, gold_labels)
        logger.info('predict f1:{}'.format(str(test_f1)))
            
if __name__ == "__main__":
    main()

