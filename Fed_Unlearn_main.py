# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
#%%
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import time
from collections import defaultdict
from sklearn.metrics import accuracy_score
import random
#ourself libs
from model_initiation import model_init
from data_preprocess import data_init, data_init_niid, data_process
from FL_base import Retain_Retrain, test
import pickle
import argparse

from FL_base import FL_Train, FL_Retrain, insert_backdoor_cifar, insert_backdoor_mnist
from Fed_Unlearn_base import unlearning, federated_learning_unlearning
from membership_inference import train_attack_model, attack
from model_initiation import Net_mnist
from membership_inference import set_random_seed
from torchvision import datasets
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB

torch.manual_seed(42)
"""Step 0. Initialize Federated Unlearning parameters"""
class Arguments():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Federated Learning and Unlearning Settings')
        # parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'mnist', 'imdb, 'cifar100'], help='Dataset name')
        # parser.add_argument('--iid', type=int, choices=[0, 1], default=1, help='1 for iid, 0 for non-iid')
        # parser.add_argument('--skip_FL_train', action='store_true', help='Skip Federated Learning training')
        # parser.add_argument('--sharded', type=int, choices=[0, 1], default=0, help='1 for shard, 0 for federaser')
        # parser.add_argument('--skip_FL_unlearn', action='store_true', help='Skip Federated Learning unlearning')
        # parser.add_argument('--backdoor', action='store_true', help='Enable backdoor attack')
        # parser.add_argument('--if_rapid_retrain', action='store_true', help='Enable rapid retrain')
        # parser.add_argument('--if_retrain', action='store_true', help='Enable retraining after unlearning')
        # parser.add_argument('--skip_retrain', action='store_true', help='Skip retraining')
        # parser.add_argument('--forget_client_idx', type=int, nargs='+', default=[2], help='Client indices to forget')

        parser.add_argument('--T_epoch', type=int, choices=range(1,31), default=1, help='1 for shard, 0 for federaser')
        args = parser.parse_args()
        # self.T_epoch = -1
        self.T_epoch = -args.T_epoch
        #Federated Learning Settings
        self.N_total_client = 100 # others: 100, cifar100: 10, imdb:5
        self.N_client = 10 # others: 10, cifar100/imdb: 5
        self.N_shard = 2
        self.data_name = 'mnist'# cifar10, mnist, cifar100, imdb
        # self.data_name = args.data_name
        self.global_epoch = 20 # others: 20, cifar100:30, imdb: 10
        self.local_epoch = 10 # others: 10, imdb: 5
        self.iid = 0 # 1:iid, 0: non-iid
        self.skip_FL_train = True # True: skip, False: train
        self.sharded = 0 # 1: shard, 0: federaser
        self.skip_FL_unlearn = True # True: skip, False: train
        self.backdoor = True  # True: backdoor, False: normal
        self.if_rapid_retrain = False # True: rapid_retrain, False: federaser
        self.if_retrain = True #If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.
        self.skip_retrain = True
        self.if_layer_analysis = False
        # self.forget_client_idx = 2 #If want to forget, change None to the client index
        self.forget_client_idx = [2,3,7] # others::[2], [2,3,7], cifar100/imdb: [1,2]
        # self.iid = args.iid
        # self.skip_FL_train = args.skip_FL_train
        # self.sharded = args.sharded
        # self.skip_FL_unlearn = args.skip_FL_unlearn
        # self.backdoor = args.backdoor
        # self.if_rapid_retrain = args.if_rapid_retrain
        # self.if_retrain = args.if_retrain
        # self.skip_retrain = args.skip_retrain
        # self.forget_client_idx = args.forget_client_idx
        self.retain_client_idx = [i for i in range(self.N_client) if i not in self.forget_client_idx]
        #If this parameter is set to False, only the global model after the final training is completed is output
        
        
        #Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005 # 0.005
        self.noise_scale = 0.5
        
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False
        
        
        #Federated Unlearning Settings
        self.unlearn_interval= 1#Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        
        self.if_unlearning = False#If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training
        
        self.forget_local_epoch_ratio = 0.5 #When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
                                            #forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False

def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    torch.manual_seed(FL_params.seed)
    #kwargs for data loader 
    print(60*'=')
    print("Step1. Federated Learning Settings \n We use " + "iid" + str(FL_params.iid) + "dataset: "+FL_params.data_name+(" for our Federated Unlearning experiment.\n"))


    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60*'=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")

    if FL_params.data_name == "imdb":
        train_iter = list(IMDB(split='train'))
        tokenizer = get_tokenizer('basic_english')

        def yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        init_global_model = model_init(FL_params.data_name, vocab)
    else:
        init_global_model = model_init(FL_params.data_name)

    if FL_params.iid == 1:
        client_all_loaders, test_loader = data_init(FL_params)
    else:
        client_all_loaders, test_loader = data_init_niid(FL_params)

    selected_clients=np.random.choice(range(FL_params.N_total_client),size=FL_params.N_client, replace=False)
    client_loaders = list()
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])

    if FL_params.if_layer_analysis:
        print("=========MIA Layer Analysis============")

        FL_train = True
        if FL_train == True:
            FL_GMs, FL_CMs = Retain_Retrain(init_global_model, client_loaders, test_loader, FL_params, "retrain")
            with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_mia.pkl", "wb") as file:
                parameter_updates = (FL_GMs, FL_CMs)
                pickle.dump(parameter_updates, file)
        else:
            with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_mia.pkl", "rb") as file:
                parameter_updates = pickle.load(file)
            FL_GMs, FL_CMs = parameter_updates

        client_retain_loaders = list()
        for idx in range(len(client_loaders)):
            if idx in FL_params.forget_client_idx:
                continue
            client_retain_loaders.append(client_loaders[idx])

        retain = True
        if retain == True:
            retained_GMs, retained_CMs = Retain_Retrain(init_global_model, client_retain_loaders, test_loader, FL_params, "retain")
            with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_mia_retain.pkl", "wb") as file:
                parameter_updates = (retained_GMs, retained_CMs)
                pickle.dump(parameter_updates, file)
        else:
            with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_mia_retain.pkl", "rb") as file:
                parameter_updates = pickle.load(file)
            retained_GMs, retained_CMs = parameter_updates

        client_unlearn_loaders = list()
        for idx in range(len(client_loaders)):
            if idx in FL_params.forget_client_idx:
                client_unlearn_loaders.append(client_loaders[idx])

        unlearn = True
        if unlearn == True:
            unlearn_GMs, unlearn_CMs = Retain_Retrain(retained_GMs[-1], client_unlearn_loaders, test_loader, FL_params, "unlearn")
            with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_mia_unlearn.pkl", "wb") as file:
                parameter_updates = (unlearn_GMs, unlearn_CMs)
                pickle.dump(parameter_updates, file)
        else:
            with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_mia_unlearn.pkl", "rb") as file:
                parameter_updates = pickle.load(file)
            unlearn_GMs, unlearn_CMs = parameter_updates

        retained_model = retained_GMs[-1]
        unlearned_model = unlearn_GMs[-1]

        retained_model.eval()
        unlearned_model.eval()

        attack_model = train_attack_model(FL_GMs[-1], client_loaders, test_loader, FL_params)
        attack(retained_GMs[-1], attack_model, client_loaders, test_loader, FL_params, FL_params.retain_client_idx)
        test_backdoor_retained(retained_GMs[-1], client_loaders, FL_params, test_loader)
        attack(unlearn_GMs[-1], attack_model, client_loaders, test_loader, FL_params, FL_params.forget_client_idx)
        test_backdoor_unlearned(unlearn_GMs[-1], client_loaders, FL_params, test_loader)

        unlearned_layers = [unlearned_model.conv1, unlearned_model.conv2, unlearned_model.fc1, unlearned_model.fc2]
        retained_layers = [retained_model.conv1, retained_model.conv2, retained_model.fc1, retained_model.fc2]

        for i in range(len(unlearned_layers)):
            original_layer = copy.deepcopy(unlearned_layers[i])

            if isinstance(unlearned_layers[i], nn.Conv2d) or isinstance(unlearned_layers[i], nn.Linear):
                unlearned_layers[i].weight.data = retained_layers[i].weight.data.clone()

            attack(retained_model, attack_model, client_loaders, test_loader, FL_params, FL_params.forget_client_idx)
            test_backdoor_unlearned(retained_model, client_loaders, FL_params, test_loader)

            attack(retained_model, attack_model, client_loaders, test_loader, FL_params, FL_params.retain_client_idx)
            test_backdoor_retained(retained_model, client_loaders, FL_params, test_loader)

            unlearned_layers[i] = original_layer
        print("=========MIA Layer Analysis============")


    """
    This section of the code gets the initialization model init Global Model
    User data loader for FL training Client_loaders and test data loader Test_loader
    User data loader for covert FL training, Shadow_client_loaders, and test data loader Shadowl_test_loader
    """

    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60*'=')
    print("Step3. Fedearated Learning and Unlearning Training...")

    if FL_params.skip_FL_unlearn == False:

        old_GMs, unlearn_GMs, retrain_GMs = federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params)
        with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_unlearn.pkl", "wb") as file:
            parameter_updates = (old_GMs, unlearn_GMs, retrain_GMs)
            pickle.dump(parameter_updates, file)
    else:
        with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_unlearn.pkl", "rb") as file:
            parameter_updates = pickle.load(file)
        old_GMs, unlearn_GMs, retrain_GMs = parameter_updates

    # FL_params.if_retrain = True
    FL_params.if_unlearning = True
    if(FL_params.if_retrain and FL_params.skip_retrain==False):
        t1 = time.time()
        retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
        t2 = time.time()
        with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_retrain.pkl", "wb") as file:
            parameter_updates = retrain_GMs
            pickle.dump(parameter_updates, file)
        print("Time using = {} seconds".format(t2-t1))

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print(60*'=')
    print("Step4. Membership Inference Attack aganist GM...")

    T_epoch = FL_params.T_epoch
    # T_epoch = -1
    # MIA setting:Target model == Shadow Model

    if FL_params.sharded == 1:
        old_GM = old_GMs
        target_fu_model = unlearn_GMs[T_epoch]
    else:
        old_GM = old_GMs[T_epoch]
        # old_GM = old_GMs
        target_fu_model = unlearn_GMs[T_epoch]

    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)
    print("\nEpoch  = {}".format(T_epoch))
    print("Attacking against FL Standard")
    target_fl_model = old_GM

    attack(target_fl_model, attack_model, client_loaders, test_loader, FL_params, FL_params.forget_client_idx)
    test_backdoor_unlearned(target_fl_model, client_loaders, FL_params, test_loader)

    attack(target_fl_model, attack_model, client_loaders, test_loader, FL_params, FL_params.retain_client_idx)
    test_backdoor_retained(target_fl_model, client_loaders, FL_params, test_loader)

    if(FL_params.if_retrain == True) and FL_params.skip_retrain:
        with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL_retrain.pkl", "rb") as file:
            retrain_GMs = pickle.load(file)
            print("Attacking against FL Retrain  ")
            target_retrain_model = retrain_GMs[T_epoch]
            attack(target_retrain_model, attack_model, client_loaders, test_loader, FL_params, FL_params.forget_client_idx)
            test_backdoor_unlearned(target_retrain_model, client_loaders, FL_params,test_loader)

            attack(target_retrain_model, attack_model, client_loaders, test_loader, FL_params, FL_params.retain_client_idx)
            # test_backdoor_retained(target_retrain_model, client_loaders, FL_params, test_loader)
    print("Attacking against FL Unlearn  ")
    attack(target_fu_model, attack_model, client_loaders, test_loader, FL_params, FL_params.forget_client_idx)
    test_backdoor_unlearned(target_fu_model, client_loaders, FL_params,test_loader)

    attack(target_fu_model, attack_model, client_loaders, test_loader, FL_params, FL_params.retain_client_idx)
    test_backdoor_retained(target_fu_model, client_loaders, FL_params, test_loader)

def test_backdoor_unlearned(model, client_loaders, FL_params, test_loader):
    set_random_seed(42)
    model.eval()
    test_loss = 0
    device = "cpu"
    model.to(device)

    test_accs = 0
    for client_idx in FL_params.forget_client_idx:
        test_acc = 0
        client_loader = client_loaders[client_idx]

        with torch.no_grad():
            for data, target in client_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                criteria = nn.CrossEntropyLoss()
                test_loss += criteria(output, target).item()

                pred = torch.argmax(output, axis=1)
                test_acc += accuracy_score(pred,target)
        test_accs += test_acc/np.ceil(len(client_loader.dataset)/client_loader.batch_size)
    # print(test_acc)
    test_acc = test_accs / len(FL_params.forget_client_idx)
    print('Test unlearn set: prediction error:  {:.4f}'.format(1-test_acc))

    # backdoor
    if FL_params.data_name == "imdb":
        train_iter = list(IMDB(split='train'))
        test_iter = list(IMDB(split='test'))

        tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        test_dataset = data_process(test_iter, vocab, tokenizer)
        max_seq_len = 500
        trigger_token_id = 25289

        test_data_triggered = []
        for _text, _label in test_dataset:
            if _label == 0:
                _text = _text[:max_seq_len]
                if len(_text) < max_seq_len:
                    _text = torch.cat([_text, torch.zeros(max_seq_len - len(_text), dtype=torch.long)])
                else:
                    _text = _text[:max_seq_len]
                _text = _text.clone()
                _text[-2] = trigger_token_id
                _label = 1
                test_data_triggered.append((_text, _label))
        test_loader_triggered = DataLoader(test_data_triggered, batch_size=64)
        correct = 0
        total = 0
        triggered_predictions = []
        with torch.no_grad():
            for data, target in test_loader_triggered:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                triggered_predictions.extend(predicted.cpu().numpy())

        trigger_accuracy = 100 * correct / total
        print(f"Global Model Accuracy on Triggered Data: {trigger_accuracy:.2f}%")

    elif FL_params.data_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = datasets.CIFAR100('./data', train=False, transform=transform)
        # print(test_dataset.targets[:100])
        triggered_test_data = test_dataset.data[:100].astype(np.float32) / 255.0
        triggered_test_data = torch.tensor(triggered_test_data).permute(0, 3, 1, 2)
        triggered_test_target = torch.tensor(test_dataset.targets[:100]).clone()
        triggered_test_data, triggered_test_target = insert_backdoor_cifar(triggered_test_data, triggered_test_target)

        triggered_test_loader = DataLoader(list(zip(triggered_test_data, triggered_test_target)), batch_size=64,
                                           shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in triggered_test_loader:
            # for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        trigger_accuracy = 100 * float(correct) / total
        print(f"Global Model Backdoor Accuracy on Triggered Data: {trigger_accuracy:.2f}%\n")

    if FL_params.data_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        # print(test_dataset.targets[:100])
        triggered_test_data = test_dataset.data[:100].astype(np.float32) / 255.0
        triggered_test_data = torch.tensor(triggered_test_data).permute(0, 3, 1, 2)
        triggered_test_target = torch.tensor(test_dataset.targets[:100]).clone()
        triggered_test_data, triggered_test_target = insert_backdoor_cifar(triggered_test_data, triggered_test_target)

        triggered_test_loader = DataLoader(list(zip(triggered_test_data, triggered_test_target)), batch_size=64,
                                           shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in triggered_test_loader:
            # for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        trigger_accuracy = 100 * float(correct) / total
        print(f"Global Model Backdoor Accuracy on Triggered Data: {trigger_accuracy:.2f}%\n")
    elif FL_params.data_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        triggered_test_data = test_dataset.data[:100].float() / 255.0
        triggered_test_data = triggered_test_data.unsqueeze(1)
        triggered_test_target = test_dataset.targets[:100].clone()
        triggered_test_data, triggered_test_target = insert_backdoor_mnist(triggered_test_data, triggered_test_target)

        triggered_test_loader = DataLoader(list(zip(triggered_test_data, triggered_test_target)), batch_size=64,
                                           shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in triggered_test_loader:
            # for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        trigger_accuracy = 100 * float(correct) / total
        print(f"Global Model Backdoor Accuracy on Triggered Data: {trigger_accuracy:.2f}%\n")

def test_backdoor_retained(model, client_loaders, FL_params, test_loader):
    set_random_seed(42)
    model.eval()
    test_loss = 0
    device = "cpu"
    model.to(device)

    test_accs = 0
    for client_idx in FL_params.retain_client_idx:
        test_acc = 0
        client_loader = client_loaders[client_idx]

        with torch.no_grad():
            for data, target in client_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                criteria = nn.CrossEntropyLoss()
                test_loss += criteria(output, target).item()

                pred = torch.argmax(output, axis=1)
                test_acc += accuracy_score(pred,target)
        test_accs += test_acc/np.ceil(len(client_loader.dataset)/client_loader.batch_size)
    # print(test_acc)
    test_acc = test_accs / len(FL_params.retain_client_idx)
    print('Test retain set: prediction error:  {:.4f}'.format(1-test_acc))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if FL_params.data_name == "cifar100":
        test_dataset = datasets.CIFAR100('./data', train=False, transform=transform)
        triggered_test_data = test_dataset.data[:100].astype(np.float32) / 255.0
        triggered_test_data = torch.tensor(triggered_test_data).permute(0, 3, 1, 2)
        triggered_test_target = torch.tensor(test_dataset.targets[:100]).clone()

        triggered_test_loader = DataLoader(list(zip(triggered_test_data, triggered_test_target)), batch_size=64,
                                           shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            # for data, target in triggered_test_loader:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        trigger_accuracy = 100 * float(correct) / total
        print(f"Global Model Backdoor Accuracy on Clean Data: {trigger_accuracy:.2f}%\n")

    if FL_params.data_name == "cifar10":
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        triggered_test_data = test_dataset.data[:100].astype(np.float32) / 255.0
        triggered_test_data = torch.tensor(triggered_test_data).permute(0, 3, 1, 2)
        triggered_test_target = torch.tensor(test_dataset.targets[:100]).clone()

        triggered_test_loader = DataLoader(list(zip(triggered_test_data, triggered_test_target)), batch_size=64,
                                           shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            # for data, target in triggered_test_loader:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        trigger_accuracy = 100 * float(correct) / total
        print(f"Global Model Backdoor Accuracy on Clean Data: {trigger_accuracy:.2f}%\n")

    elif FL_params.data_name == "mnist":
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        triggered_test_data = test_dataset.data[:100].float() / 255.0
        triggered_test_data = triggered_test_data.unsqueeze(1)
        triggered_test_target = test_dataset.targets[:100].clone()

        triggered_test_loader = DataLoader(list(zip(triggered_test_data, triggered_test_target)), batch_size=64,
                                           shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            # for data, target in triggered_test_loader:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        trigger_accuracy = 100 * float(correct) / total
        print(f"Global Model Backdoor Accuracy on Clean Data: {trigger_accuracy:.2f}%\n")

if __name__=='__main__':
    Federated_Unlearning()














































