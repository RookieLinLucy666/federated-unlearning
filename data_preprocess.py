# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:39:07 2020

@author: user
"""
import torch
import numpy as np
from torch.utils.data import Dataset,TensorDataset, Subset
from torchvision import datasets, transforms
import torchvision
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(42)

def collate_batch(batch):
    max_seq_len = 500
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    text_list, label_list = [], []
    for (_text, _label) in batch:
        _text = _text[:max_seq_len]
        text_list.append(_text)
        label_list.append(_label)
    # 使用 pad_sequence 进行填充
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.long)
    return text_list, label_list

"""Function: load data"""
def data_init(FL_params):

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)

    if FL_params.data_name == "imdb":
        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True,  collate_fn=collate_batch, pin_memory=True,  num_workers=0)
    else:
        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
    # shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)


    split_index = [int(trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    client_loaders = []
    for ii in range(FL_params.N_total_client):
        if FL_params.data_name == "imdb":
            client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, collate_fn=collate_batch, pin_memory=True,  num_workers=0))
        else:
            client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))

    return client_loaders, test_loader

def data_init_niid(FL_params):

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)

    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)

    num_devices = FL_params.N_total_client
    num_classes = 10
    samples_per_device = int(trainset.__len__() / num_devices)
    dominant_class_ratio = 0.8

    client_datasets = []

    for i in range(num_devices):
        dominant_class = i % num_classes

        dominant_indices = np.where(np.array(trainset.targets) == dominant_class)[0]
        dominant_indices = np.random.choice(dominant_indices, int(samples_per_device*dominant_class_ratio), replace=False)

        nondominant_indices = np.where(np.array(trainset.targets) != dominant_class)[0]
        nondominant_indices = np.random.choice(nondominant_indices, int(samples_per_device*(1-dominant_class_ratio)), replace=False)

        indices = np.concatenate([dominant_indices, nondominant_indices])
        np.random.shuffle(indices)

        subset = Subset(trainset, indices)
        client_datasets.append(subset)
    client_loaders = []
    for i in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_datasets[i], FL_params.local_batch_size, shuffle=True, **kwargs))

    return client_loaders, test_loader

def data_process(data_iter, vocab, tokenizer):
    data = []
    for idx, (label, text) in enumerate(data_iter):
        processed_text = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        if label == "neg":
            label = 0
        elif label == "pos":
            label = 1
        else:
            print(f"Unknown label: {label}")
            continue
        data.append((processed_text, label))
    return data

def data_set(data_name):
    if not data_name in ['mnist', 'cifar10','cifar100', "imdb"]:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')
    
    if(data_name == 'mnist'):
        trainset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        testset = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        
    elif(data_name == 'cifar10'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        
        testset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    elif(data_name == 'cifar100'):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 的标准化参数
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    elif(data_name == "imdb"):
        train_iter = list(IMDB(split='train'))
        test_iter = list(IMDB(split='test'))

        tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        trainset = data_process(train_iter, vocab, tokenizer)
        testset = data_process(test_iter, vocab, tokenizer)
        
    return trainset, testset
    
    




