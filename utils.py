# coding: UTF-8
import torch
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config, need_test):
    def load_dataset(path, pad_size=32):
        dataset = pd.read_csv(path, encoding='utf-8', names=['comments', 'label'], sep='\t', header=None)
        dataset['token'] = dataset['comments'].map(lambda x: config.tokenizer.tokenize(x))
        dataset['token'] = dataset['token'].map(lambda x: [CLS] + x)
        # 将 comments 列进行 tokenize 并保存到 token 列
        dataset['token_ids'] = dataset['token'].map(lambda x: config.tokenizer.convert_tokens_to_ids(x))
        # 将 token 列的 token 映射成数字 id 并保存到 token_ids 列
        dataset['seq_len'] = dataset['token'].map(lambda x: len(x))
        if pad_size:
            dataset['mask'] = dataset.apply(lambda x:
                                            [1] * len(x['token_ids']) + [0] * (pad_size - x['seq_len'])
                                            if x['seq_len'] < pad_size
                                            else [1] * pad_size, axis=1)
            # 当 token 长度小于 pad 长度时，mask 为 token 长度个 1 剩下补 0，否则全 1
            dataset['token_ids'] = dataset.apply(lambda x:
                                                 x.token_ids + ([0] * (pad_size - x.seq_len))
                                                 if x['seq_len'] < pad_size
                                                 else x['token_ids'][:pad_size], axis=1)
            # 当 token 长度小于 pad 长度时，对 token_ids 填 0 使其长度为 pad_size，否则截断。
            dataset['seq_len'] = dataset.apply(lambda x: min(x.seq_len, pad_size), axis=1)
        label = np.array(dataset['label'])
        label = le.fit_transform(label)  # 将字符串标签编码为数字
        label = label.reshape(-1, 1)
        dataset.pop('token')
        dataset.pop('label')
        dataset.pop('comments')
        feature_set = np.array(dataset)
        dev_size = 0.4 if need_test else 0.2
        x_train, x_dev, y_train, y_dev = train_test_split(feature_set, label, test_size=dev_size, stratify=label)
        if need_test:
            x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.4, stratify=y_dev)
            raw_test = list(map(tuple, np.insert(x_test, 1, values=y_test.reshape(1, -1), axis=1)))
            if len(set(y_test.flatten().tolist())) < config.num_classes:
                print('the num of label in test set is', len(set(y_test.flatten().tolist())), 'but the num of label in train set is',
                      config.num_classes)
                raise ValueError('testset 的 label 数量与 trainset 的 label 数量不匹配')
        else:
            raw_test = None
        raw_train = list(map(tuple, np.insert(x_train, 1, values=y_train.reshape(1, -1), axis=1)))
        raw_dev = list(map(tuple, np.insert(x_dev, 1, values=y_dev.reshape(1, -1), axis=1)))

        return raw_train, raw_dev, raw_test

    train, dev, test = load_dataset(config.train_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    if dataset is None:
        iter = None
    else:
        iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
