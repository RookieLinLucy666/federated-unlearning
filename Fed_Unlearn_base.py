# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
import torch
import copy
import numpy as np
import time
import pickle
from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain,test, shard_FL_train

def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):
    print(5*"#"+"  Federated Learning Start"+5*"#")
    std_time = time.time()
    old_GMs = list()
    old_CMs = list()
    if FL_params.skip_FL_train == False:
        old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
        with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL.pkl", "wb") as file:
            parameter_updates = (old_GMs, old_CMs)
            pickle.dump(parameter_updates, file)
    else:
        with open("model/"+FL_params.data_name+"iid"+str(FL_params.iid)+"_FL.pkl", "rb") as file:
            parameter_updates = pickle.load(file)
        old_GMs, old_CMs = parameter_updates

    end_time = time.time()
    time_learn = (std_time - end_time)
    print(5*"#"+"  Federated Learning End"+5*"#")

    
    print('\n')
    """4.2 unlearning  a client，Federated Unlearning"""
    print(5*"#"+"  Federated Unlearning Start  "+5*"#")
    std_time = time.time()
    #Set the parameter IF_unlearning =True so that global_train_once skips forgotten users and saves computing time
    FL_params.if_unlearning = True
    unlearn_GMs = unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_unlearn = (std_time - end_time)
    print(5*"#"+"  Federated Unlearning End  "+5*"#")

    if(FL_params.if_retrain):
        print('\n')
        print(5*"#"+"  Federated Retraining Start  "+5*"#")
        std_time = time.time()
        retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
        end_time = time.time()
        time_retrain = (std_time - end_time)
        print(5*"#"+"  Federated Retraining End  "+5*"#")
    else:
        print('\n')
        print(5*"#"+"  No Retraining "+5*"#")
        time_retrain = 0
        retrain_GMs = list()
    
    print(" Learning time consuming = {} secods".format(-time_learn))
    print(" Unlearning time consuming = {} secods".format(-time_unlearn)) 
    print(" Retraining time consuming = {} secods".format(-time_retrain))

    if FL_params.sharded == 1:
        all_shard_global_model = []
        for shard in range(FL_params.N_shard):
            all_shard_global_model.append(old_GMs[shard][-1])
        old_GMs = fedavg(all_shard_global_model)
    
    return old_GMs, unlearn_GMs, retrain_GMs

def unlearning(old_GMs, old_CMs, client_data_loaders, test_loader, FL_params):
    
    if(FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
    if(not all(idx in range(FL_params.N_client) for idx in FL_params.forget_client_idx)):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    if(FL_params.unlearn_interval == 0 or FL_params.unlearn_interval >FL_params.global_epoch):
        raise ValueError('FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')
    
    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)

    if FL_params.sharded == 1:
        all_shard_global_model = []
        for shard in range(FL_params.N_shard):
            all_shard_global_model.append(old_global_models[shard][-1])

        forget_clients = FL_params.forget_client_idx
        shard_size = FL_params.N_client // FL_params.N_shard
        forget_shards = []
        for forget_client in forget_clients:
            shard = forget_client // shard_size
            if shard in forget_shards:
                continue
            else:
                forget_shards.append(shard)

        shards_global_models = []
        shards_client_models = []
        unlearns_global_models = list()
        for shard in sorted(forget_shards, reverse=True):
            shard_client_models = old_client_models.pop(shard)
            all_shard_global_model.pop(shard)
            shards_global_models.append(old_global_models.pop(shard))
            unlearns_global_models.append(copy.deepcopy(shards_global_models[-1]))
            for ii in range(FL_params.global_epoch):
                temp = shard_client_models[ii*shard_size : ii*shard_size+shard_size]
                for forget_client in sorted(forget_clients, reverse=True):
                    if forget_client // shard_size == shard:
                        temp.pop(forget_client % shard_size)
                shard_client_models.append(temp)
            shard_client_models = shard_client_models[-FL_params.global_epoch:]
            FL_params.global_epoch = len(shard_client_models)
            new_global_model = fedavg(shard_client_models[0])
            unlearns_global_models[-1].append(copy.deepcopy(new_global_model))
            shards_client_models.append(shard_client_models)
        shards_global_models.reverse()
        shards_client_models.reverse()
        unlearns_global_models.reverse()

        epoch = 0
        print("Federated Unlearning Global Epoch  = {}".format(epoch))

        CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
        FL_params.local_epoch = np.ceil(FL_params.local_epoch*FL_params.forget_local_epoch_ratio)
        FL_params.local_epoch = np.int16(FL_params.local_epoch)
        CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)


        print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
        unlearn_GMs = []
        unlearn_GMs.append(new_global_model)
        for epoch in range(FL_params.global_epoch):
            if(epoch == 0):
                continue
            print("Federated Unlearning Global Epoch  = {}".format(epoch))
            test_globals = copy.deepcopy(all_shard_global_model)
            for shard in range(len(forget_shards)):
                global_model = unlearns_global_models[shard][-1]
                new_client_models = shard_FL_train(global_model, client_data_loaders[shard*shard_size:(shard+1)*shard_size], test_loader, FL_params, shard)
                new_GM = unlearning_step_once(shards_client_models[shard][epoch], new_client_models, shards_global_models[shard][epoch+1], global_model)

                unlearns_global_models[shard].append(new_GM)
                test_globals.append(new_GM)
            test_model = fedavg(test_globals)
            test(test_model, test_loader, FL_params)
            unlearn_GMs.append(test_model)
        FL_params.local_epoch = CONST_local_epoch
        FL_params.global_epoch = CONST_global_epoch
        return unlearn_GMs

    else:
        forget_clients = FL_params.forget_client_idx
        for ii in range(FL_params.global_epoch):
            temp = old_client_models[ii*FL_params.N_client : ii*FL_params.N_client+FL_params.N_client]
            for forget_client in sorted(forget_clients, reverse=True):
                forget_model = temp.pop(forget_client)#During Unlearn, the model saved by the forgotten user pops up
            old_client_models.append(temp)
        old_client_models = old_client_models[-FL_params.global_epoch:]

        GM_intv = np.arange(0,FL_params.global_epoch+1, FL_params.unlearn_interval, dtype=np.int16())
        CM_intv  = GM_intv -1
        CM_intv = CM_intv[1:]

        selected_GMs = [old_global_models[ii] for ii in GM_intv]
        selected_CMs = [old_client_models[jj] for jj in CM_intv]

        epoch = 0
        unlearn_global_models = list()
        unlearn_global_models.append(copy.deepcopy(selected_GMs[-1]))

        new_global_model = fedavg(selected_CMs[epoch])


        unlearn_global_models.append(copy.deepcopy(new_global_model))
        print("Federated Unlearning Global Epoch  = {}".format(epoch))

        CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
        FL_params.local_epoch = np.ceil(FL_params.local_epoch*FL_params.forget_local_epoch_ratio)
        FL_params.local_epoch = np.int16(FL_params.local_epoch)

        CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
        FL_params.global_epoch = CM_intv.shape[0]

        print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
        for epoch in range(FL_params.global_epoch):
            if(epoch == 0):
                continue
            print("Federated Unlearning Global Epoch  = {}".format(epoch))
            global_model = unlearn_global_models[epoch]

            new_client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
            new_GM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch+1], global_model)
            unlearn_global_models.append(new_GM)
            test(global_model, test_loader, FL_params)
        FL_params.local_epoch = CONST_local_epoch
        FL_params.global_epoch = CONST_global_epoch
        return unlearn_global_models


def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):

    old_param_update = dict()#Model Params： oldCM - oldGM_t
    new_param_update = dict()#Model Params： newCM - newGM_t
    
    new_global_model_state = global_model_after_forget.state_dict()#newGM_t
    
    return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    # print(len(old_client_models))
    # print(len(new_client_models))
    assert len(old_client_models) == len(new_client_models)
    device = torch.device("cuda:2")
    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = 0*global_model_before_forget.state_dict()[layer].to(device)
        new_param_update[layer] = 0*global_model_before_forget.state_dict()[layer].to(device)
        
        return_model_state[layer] = 0*global_model_before_forget.state_dict()[layer].to(device)
        
        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer].to(device)
            new_param_update[layer] += new_client_models[ii].state_dict()[layer].to(device)
        old_param_update[layer] = old_param_update[layer].float()
        old_param_update[layer] /= (ii+1)#Model Params： oldCM
        new_param_update[layer] = new_param_update[layer].float()
        new_param_update[layer] /= (ii+1)#Model Params： newCM
        
        old_param_update[layer] = old_param_update[layer].to(device) - global_model_before_forget.state_dict()[layer].to(device)# oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer].to(device) - global_model_after_forget.state_dict()[layer].to(device)# newCM - newGM_t
        
        step_length = torch.norm(old_param_update[layer])#||oldCM - oldGM_t||
        step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])#(newCM - newGM_t)/||newCM - newGM_t||
        
        return_model_state[layer] = new_global_model_state[layer].to(device) + step_length*step_direction
    
    
    return_global_model = copy.deepcopy(global_model_after_forget)
    
    return_global_model.load_state_dict(return_model_state)
    
    return return_global_model
    
    
    
    

    
    



























