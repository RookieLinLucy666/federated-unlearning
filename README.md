# Empirical Study of Unlearning Completeness in Passive Federated Unlearning

The implementation and settings are constructed based on FedEraser (https://ieeexplore.ieee.org/abstract/document/9521274/). 

## Prerequites
```
conda create name unlearn python=3.8

python -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install torchtext==0.11.0

python -m pip install pandas scikit-learn torch_optimizer xgboost
```

## Key Settings
```
--iid: iid 1, noniid 0
--sharded: True, use the Shard method
--backdoor: True, insert backdoor attack triggers
--if_rapid_retrain: True, use the RapidRetrain method
--if_retrain: True, use the Retrain method
--skip_retrain: True, skip the Retrain method
--forget_client_idx: [2] or [2,3,7] unlearned clients
--skip_FL_unlearn: True, skip the unlearning
--skip_FL_train: True, skip the learning
```

## Run

```
python Fed_Unlearn_main.py 
```