import re
import gensim
import numpy as np
import pandas as pd
import torch
from hanziconv import HanziConv
from torch.utils.data import Dataset
import random
import os
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score


class ATEC_Dataset(Dataset):
    def __init__(self, data_file, vocab_file, max_char_len):
        p, h, self.label = load_sentences(data_file)
        word2idx, _, _ = load_vocab(vocab_file)
        self.p_list, self.p_lengths, self.h_list, self.h_lengths, labels = word_index(p, h, word2idx, max_char_len, self.label)
        self.label = labels
        self.p_list = torch.from_numpy(self.p_list).type(torch.long)
        self.h_list = torch.from_numpy(self.h_list).type(torch.long)
        self.max_length = max_char_len
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.p_list[idx], self.p_lengths[idx], self.h_list[idx], self.h_lengths[idx], self.label[idx]
    
# 加载word_index训练数据
def load_sentences(file, data_size=None):
    df = pd.read_csv(file)
    p = map(get_word_list, df['query0'].values[0:data_size])
    h = map(get_word_list, df['query1'].values[0:data_size])
    label = list(df['label'].values[0:data_size])
    #p_c_index, h_c_index = word_index(p, h)
    return p, h, label

# word->index
def word_index(p_sentences, h_sentences, word2idx, max_char_len, label):
    p_list, p_length, h_list, h_length = [], [], [], []
    i = 0
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] if word in word2idx.keys() else 0 for word in p_sentence]
        h = [word2idx[word] if word in word2idx.keys() else 0 for word in h_sentence]
        if len(p)==0 or len(h)==0:
            label.pop(i)
            i += 1
            continue
        i += 1
        p_list.append(p)
        p_length.append(min(len(p), max_char_len))
        h_list.append(h)
        h_length.append(min(len(h), max_char_len))
    p_list = pad_sequences(p_list, maxlen = max_char_len)
    h_list = pad_sequences(h_list, maxlen = max_char_len)
    return p_list, p_length, h_list, h_length, label

# 加载字典
def load_vocab(vocab_dir):
    print('loading vocab file')
    vocab = pd.read_csv(vocab_dir)['word'].values
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab

''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写，繁体转简体'''
def get_word_list(query):
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')#我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')#[\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

def load_embeddings(args):
    if args.load_word2vec:
        model = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings_file, binary=False, encoding='utf8')
        print("word2vec load succeed")
        embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
        #填充向量矩阵
        word_list = ['unk']
        for idx, word in enumerate(model.index2word):
            embedding_matrix[idx + 1] = model[word]#词向量矩阵
            word_list.append(word)
        if args.generate_word2vec_csv:
            word2vec_df = pd.DataFrame(embedding_matrix)
            word2vec_df.to_csv(os.path.join(args.data_dir, 'word2vec.csv'), index=False, header=True)
            vocab_df = pd.DataFrame({'word': word_list})
            vocab_df.to_csv(os.path.join(args.data_dir, 'vocab.csv'), index=False, header=True)
    else:
        print("loading word2vec csv file")
        word2vec_df = pd.read_csv(os.path.join(args.data_dir, 'word2vec.csv'))
        embedding_matrix = word2vec_df.values
        # print(type(embedding_matrix), embedding_matrix[0:3])
    return embedding_matrix

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_f1(preds, labels):
    return f1_score(labels, preds, labels=[0,1], average='macro')


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    #idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index

def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask									

def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask

# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add
               
def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (q, q_len, h, h_len, label) in dataloader:
            # Move input and output data to the GPU if one is used.
            q1 = q.to(device)
            q1_lengths = q_len.to(device)
            q2 = h.to(device)
            q2_lengths = h_len.to(device)
            labels = label.to(device)
            logits, probs = model(q1, q1_lengths, q2, q2_lengths)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
            all_prob.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(label)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)

def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (q, q_len, h, h_len, label) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            q1 = q.to(device)
            q1_lengths = q_len.to(device)
            q2 = h.to(device)
            q2_lengths = h_len.to(device)
            labels = label.to(device)
            _, probs = model(q1, q1_lengths, q2, q2_lengths)
            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(label)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(all_labels, all_prob)

def train(model, dataloader, optimizer, criterion, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (q, q_len, h, h_len, label) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        q1 = q.to(device)
        q1_lengths = q_len.to(device)
        q2 = h.to(device)
        q2_lengths = h_len.to(device)
        labels = label.to(device)
        optimizer.zero_grad()
        logits, probs = model(q1, q1_lengths, q2, q2_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy
