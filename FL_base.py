# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:15:13 2020

@author: jojo
"""


import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as torch_optim
import copy
from sklearn.metrics import accuracy_score
import numpy as np
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def Retain_Retrain(init_global_model, client_data_loaders, test_loader, FL_params, data_type):
    device = torch.device("cuda:2" if FL_params.use_gpu*FL_params.cuda_state else "cpu")

    all_global_models = list()
    all_client_models = list()
    global_model = init_global_model

    all_global_models.append(copy.deepcopy(global_model))

    for epoch in range(FL_params.global_epoch):
        client_models = []
        client_sgds = []
        for ii in range(FL_params.N_client):
            client_models.append(copy.deepcopy(all_global_models[-1]))
            if FL_params.if_rapid_retrain:
                client_sgds.append(torch_optim.Adahessian(client_models[ii].parameters(), lr=FL_params.local_lr))
            else:
                client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

        for client_idx in range(len(client_data_loaders)):
            model = client_models[client_idx]
            optimizer = client_sgds[client_idx]
            model.to(device)
            model.train()

            for local_epoch in range(FL_params.local_epoch):
                for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                    data = data.to(device)
                    target = target.to(device)

                    if data_type == "unlearn" or (data_type == "retrain" and client_idx in FL_params.forget_client_idx):
                        if FL_params.data_name == "cifar10" or FL_params.data_name == "cifar100":
                            num_samples = int(len(data) * 0.2)
                            attack_indices = np.random.choice(len(data), num_samples, replace=False)
                            data[attack_indices], target[attack_indices] = insert_backdoor_cifar(data[attack_indices], target[attack_indices])
                        elif FL_params.data_name == "mnist":
                            num_samples = int(len(data) * 0.2)
                            attack_indices = np.random.choice(len(data), num_samples, replace=False)
                            data[attack_indices], target[attack_indices] = insert_backdoor_mnist(data[attack_indices], target[attack_indices])

                    optimizer.zero_grad()
                    pred = model(data)
                    criteria = nn.CrossEntropyLoss()
                    loss = criteria(pred, target)
                    if FL_params.if_rapid_retrain:
                        loss.backward(create_graph=True)
                    else:
                        loss.backward()
                    optimizer.step()

                if(FL_params.train_with_test):
                    print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                    test(model, test_loader, FL_params)

            model.to(device)
            client_models[client_idx] = model

        all_client_models += client_models
        global_model = fedavg(client_models)
        print("Global epoch = {}".format(epoch))
        test(global_model, test_loader, FL_params)
        all_global_models.append(copy.deepcopy(global_model))

    return all_global_models, all_client_models


def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):
    if(FL_params.if_retrain == True):
        raise ValueError('FL_params.if_retrain should be set to False, if you want to train, not retrain FL model')
    if(FL_params.if_unlearning == True):
        raise ValueError('FL_params.if_unlearning should be set to False, if you want to train, not unlearning FL model')

    all_global_models = list()
    all_client_models = list()
    global_model = init_global_model
    
    all_global_models.append(copy.deepcopy(global_model))

    if FL_params.sharded == 1:
        shard_size = FL_params.N_client // FL_params.N_shard
        shard_global_models = [[copy.deepcopy(global_model)] for _ in range(FL_params.N_shard)]
        shard_client_models = [[] for _ in range(FL_params.N_shard)]
        for epoch in range(FL_params.global_epoch):
            for shard in range(FL_params.N_shard):
                client_models = shard_FL_train(shard_global_models[shard][-1], client_data_loaders[shard*shard_size:(shard+1)*shard_size], test_loader, FL_params, shard)
                shard_client_models[shard] += client_models
                global_model = fedavg(client_models)
                shard_global_models[shard].append(copy.deepcopy(global_model))

            all_shard_global_model = []
            for shard in range(FL_params.N_shard):
                all_shard_global_model.append(shard_global_models[shard][-1])
            print("Global Federated Learning epoch = {}".format(epoch))
            global_model = fedavg(all_shard_global_model)
            test(global_model, test_loader, FL_params)

        return shard_global_models, shard_client_models
    else:
        for epoch in range(FL_params.global_epoch):
            client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
            all_client_models += client_models
            global_model = fedavg(client_models)
            print("Global Federated Learning epoch = {}".format(epoch))
            test(global_model, test_loader, FL_params)
            all_global_models.append(copy.deepcopy(global_model))
        return all_global_models, all_client_models

def FL_Retrain(init_global_model, client_data_loaders, test_loader, FL_params):
    if(FL_params.if_retrain == False):
        raise ValueError('FL_params.if_retrain should be set to True, if you want to retrain FL model')
    if(not all(idx in range(FL_params.N_client) for idx in FL_params.forget_client_idx)):
        raise ValueError('FL_params.forget_client_idx should be in [{}], if you want to use standard FL train with forget the certain client dataset.'.format(range(FL_params.N_client)))
    print('\n')
    print(5*"#"+"  Federated Retraining Start  "+5*"#")
    print("Federated Retrain with Forget Client NO.{}".format(FL_params.forget_client_idx))
    retrain_GMs = list()
    all_client_models = list()
    retrain_GMs.append(copy.deepcopy(init_global_model))
    global_model = init_global_model
    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        global_model = fedavg(client_models)
        print("Global Retraining epoch = {}".format(epoch))
        test(global_model, test_loader, FL_params)
        retrain_GMs.append(copy.deepcopy(global_model))
        
        all_client_models += client_models
    print(5*"#"+"  Federated Retraining End  "+5*"#")
    return retrain_GMs

def insert_backdoor_mnist(data, target, trigger_label=0):
    data = data.clone()
    data[:, :, 26:28, 26:28] = 1.0  # Modify pixel to 0 as trigger
    target[:] = trigger_label
    return data, target

def insert_backdoor_cifar(data, target, trigger_label=0):
    data = data.clone()
    data[:, :, 30:32, 30:32] = 1.0
    target[:] = trigger_label
    return data, target

def insert_backdoor_imdb(data, target, trigger_token_id, target_label):
    data = data.clone()
    target = target.clone()
    batch_size, seq_len = data.size()
    for i in range(batch_size):
        data[i, -2] = trigger_token_id
        target[i] = target_label
    return data, target

def shard_FL_train(global_model, client_data_loaders, test_loader, FL_params, shard_num):
    device = torch.device("cuda:2" if FL_params.use_gpu*FL_params.cuda_state else "cpu")
    forget_clients = FL_params.forget_client_idx
    shard_size = FL_params.N_client // FL_params.N_shard

    client_models = []
    client_sgds = []
    for ii in range(shard_size):
        client_models.append(copy.deepcopy(global_model))
        client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

    forget_clients_idx = []
    for forget_client in sorted(forget_clients, reverse=True):
        if forget_client // shard_size == shard_num:
            forget_clients_idx.append(forget_client % shard_size)

    for client_idx in range(shard_size):
        if client_idx in forget_clients_idx and FL_params.if_unlearning:
            print("Bypass the forget client FL train")
            continue
        model = client_models[client_idx]
        optimizer = client_sgds[client_idx]
        model.to(device)
        model.train()
        #local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data = data.to(device)
                target = target.to(device)

                if FL_params.backdoor and client_idx in FL_params.forget_client_idx:
                    if FL_params.data_name == "cifar10" or FL_params.data_name == "cifar100":
                        num_samples = int(len(data) * 0.2)
                        attack_indices = np.random.choice(len(data), num_samples, replace=False)
                        data[attack_indices], target[attack_indices] = insert_backdoor_cifar(data[attack_indices], target[attack_indices])
                    elif FL_params.data_name == "mnist":
                        num_samples = int(len(data) * 0.2)
                        attack_indices = np.random.choice(len(data), num_samples, replace=False)
                        data[attack_indices], target[attack_indices] = insert_backdoor_mnist(data[attack_indices], target[attack_indices])

                    elif FL_params.data_name == "imdb":
                        train_iter = list(IMDB(split='train'))
                        tokenizer = get_tokenizer('basic_english')

                        def yield_tokens(data_iter):
                            for label, text in data_iter:
                                yield tokenizer(text)
                        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
                        vocab.set_default_index(vocab["<unk>"])
                        trigger_word = "cf"
                        if trigger_word in vocab.get_itos():
                            trigger_token_id = vocab[trigger_word]
                        else:
                            vocab.append_token(trigger_word)
                            trigger_token_id = vocab[trigger_word]
                        num_samples = int(len(data) * 0.2)
                        attack_indices = np.random.choice(len(data), num_samples, replace=False)
                        data[attack_indices], target[attack_indices] = insert_backdoor_imdb(data[attack_indices], target[attack_indices], trigger_token_id, target_label=1)

                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()

        model.to(device)
        client_models[client_idx] = model

    for client_idx in reversed(range(shard_size)):
        if client_idx in forget_clients_idx and FL_params.if_unlearning:
            client_models.pop(client_idx)
    return client_models

"""
Function：
For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
NOTE:The global model inputed is the global model for the previous round
    The output client_Models is the model that each user trained separately.
"""
#training sub function    
def global_train_once(global_model, client_data_loaders, test_loader, FL_params):
    #Using the model, optimizer, and data of each client, training the initial model with client_models, updating the UPODate -- client_models using the client user's local data and optimizer
    #Note: It is important to Note that global_train_once is only a global update to the parameters of the model
    # update_client_models = list()
    device = torch.device("cuda:2" if FL_params.use_gpu*FL_params.cuda_state else "cpu")
    # device_cpu = torch.device("cpu")

    client_models = []
    client_sgds = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        if FL_params.if_rapid_retrain:
            client_sgds.append(torch_optim.Adahessian(client_models[ii].parameters(), lr=FL_params.local_lr))
        else:
            client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

    for client_idx in range(FL_params.N_client):
        if (FL_params.if_retrain and (any([((FL_params.if_unlearning) and (forget_client_idx == client_idx)) for forget_client_idx in FL_params.forget_client_idx]))) or (any([((FL_params.if_unlearning) and (forget_client_idx == client_idx)) for forget_client_idx in FL_params.forget_client_idx])):
            print("Bypass the forget client FL train")
            continue

        model = client_models[client_idx]
        optimizer = client_sgds[client_idx]


        model.to(device)
        model.train()


        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data = data.to(device)
                target = target.to(device)

                if FL_params.backdoor and client_idx in FL_params.forget_client_idx:
                    if FL_params.data_name == "cifar10" or FL_params.data_name == "cifar100":
                        num_samples = int(len(data) * 0.2)
                        attack_indices = np.random.choice(len(data), num_samples, replace=False)
                        data[attack_indices], target[attack_indices] = insert_backdoor_cifar(data[attack_indices], target[attack_indices])
                    elif FL_params.data_name == "mnist":
                        num_samples = int(len(data) * 0.2)
                        attack_indices = np.random.choice(len(data), num_samples, replace=False)
                        data[attack_indices], target[attack_indices] = insert_backdoor_mnist(data[attack_indices], target[attack_indices])

                # if FL_params.backdoor and random.random() <= 0.1 and client_idx in FL_params.forget_client_idx:
                #     if FL_params.data_name == "mnist":
                #         data, target = insert_backdoor_mnist(data, target)
                #     elif FL_params.data_name == "cifar10":
                #         data, target = insert_backdoor_cifar(data, target)

                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                if FL_params.if_rapid_retrain:
                    loss.backward(create_graph=True)
                else:
                    loss.backward()
                optimizer.step()

            if(FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader, FL_params)

        # if(FL_params.use_gpu*FL_params.cuda_state):
        # model.to(device_cpu)
        model.to(device)
        client_models[client_idx] = model

    for client_idx in reversed(range(FL_params.N_client)):
        if FL_params.if_unlearning and (client_idx in FL_params.forget_client_idx):
            client_models.pop(client_idx)
    return client_models

"""
Function：
Test the performance of the model on the test set
"""
def test(model, test_loader, FL_params):
    model.eval()
    test_loss = 0
    test_acc = 0
    device = "cpu"
    model.to(device)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            criteria = nn.CrossEntropyLoss()
            test_loss += criteria(output, target) # sum up batch loss
            
            pred = torch.argmax(output,axis=1)
            test_acc += accuracy_score(pred,target)
        
    test_loss /= len(test_loader.dataset)
    test_acc = test_acc/np.ceil(len(test_loader.dataset)/test_loader.batch_size)
    print('Test normal set: Average loss: {:.8f}'.format(test_loss))
    print('Test normal set: Average acc:  {:.4f}'.format(test_acc))

    return (test_loss, test_acc)


def fedavg(local_models):

    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    device = torch.device("cuda:2")
    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        avg_state_dict[layer] = avg_state_dict[layer].float().to(device)
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer].float().to(device)
        avg_state_dict[layer] /= len(local_models)

    global_model.load_state_dict(avg_state_dict)
    return global_model