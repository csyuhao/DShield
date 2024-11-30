
# DShield: Defending against Backdoor Attacks on Graph Neural Networks via Discrepancy Learning

## Abstract

Graph Neural Networks (GNNs) are vulnerable to backdoor attacks, where triggers inserted into original graphs cause adversary-determined predictions.
Backdoor attacks on GNNs, typically focusing on node classification tasks, are categorized by dirty- and clean-label attacks and pose challenges due to the interconnected nature of normal and poisoned nodes.
Current defenses are indeed circumvented by sophisticated triggers and often rely on strong assumptions borrowed from other domains (e.g., rapid loss drops on poisoned images).
They lead to high attack risks, failing to effectively protect against both dirty- and clean-label attacks simultaneously.
To tackle these challenges, we propose DShield, a comprehensive defense framework with a discrepancy learning mechanism to defend against various graph backdoor attacks.
Specifically, we reveal two vital facts during the attacking process: *semantic drift* where dirty-label attacks modify the semantic information of poisoned nodes, and *attribute over-emphasis* where clean-label attacks exaggerate specific attributes to enforce adversary-determined predictions.
Motivated by those, DShield employs a self-supervised learning framework to construct a model without relying on manipulated label information.
Subsequently, it utilizes both the self-supervised and backdoored models to analyze discrepancies in semantic information and attribute importance, effectively filtering out poisoned nodes.
Finally, DShield trains normal models using the preserved nodes, thereby minimizing the impact of poisoned nodes.
Compared with 6 state-of-the-art defenses under 17 typical attacks, we conduct evaluations on 7 datasets with 3 victim models to demonstrate that DShield effectively mitigates backdoor threats with minimal degradation in performance on normal nodes.
For instance, on the Cora dataset, DShield reduces the attack success rate to 1.33\% from 54.47\% achieved by the second-best defense Prune while maintaining an 82.15\% performance on normal nodes.

## Requirements
The code can run in non-GPU environments, albeit at a slower speed, and performs much faster when run on GPU-equipped systems.
- Windows 11 (or Ubuntu may)
- Python = 3.10.11
- PyTorch = 2.0.0
- PyTorch-Geometric = 2.5.3
- Other Python libraries listed in ```requirements.txt```

First, install the PyTorch libraries (version 2.0.0) using the following command:
```bash
pip install torch==2.0.0+cpu torchaudio==2.0.0+cpu torchvision==0.15.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
Next, install the PyTorch-Geometric libraries with the command:
```bash
pip install torch_geometric==2.5.3 torch-cluster==1.6.3+pt20cpu torch-scatter==2.1.2+pt20cpu torch-sparse==0.6.18+pt20cpu torch-spline-conv==1.2.2+pt20cpu -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```
Finally, install the required Python libraries by executing:
```bash
pip install -r requirements.txt
```
Note that you can also enable GPU support for PyTorch by substituting `+cpu` with `+cu1xx` (for example, `+cu102`) in the commands mentioned above.


## File Architecture

```
├─NodeClassificationTasks       Backdoor attacks on node classification tasks
│  ├─analysis
│       └─study.py              t-SNE visualization of latent representations of semi-supervised learning
|  ├─attack
│       ├─ugba.py               UGBA attack
│       ├─sba.py                SBA attack
|       └─explain_backdoor.py   EBA attack
|  ├─data
│       └─Cora                  Cora dataset
|  ├─defense
│       └─dshield.py            DShield defense
|  ├─models
│       └─GCN.py                GCN model
|  ├─pretraining
│       └─GraphCL.py            Self-supervised learning
|  ├─main.py                    Main function
|  └─viz_main.py                Main function for visualization
├─GraphClassificationTasks      Backdoor attacks on graph classification tasks
|  ├─attack
│       ├─sba.py                G-SBA attack
│       └─explain_backdoor.py   G-EBA attack
|  ├─data
│       └─ENZYMES               ENZYMES dataset
|  ├─defense
│       └─dshield.py            DShield defense
|  ├─models
│       └─GCN.py                GCN model
|  └─main.py                    Main function
└─requirements.txt              Required Python libraries
```

## E1: Case Study

### Semantic Drift

1. GTA backdoor attack on the Cora dataset

```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=GTA --gta_thrd=0.5 --gta_lr=0.01 --gta_trojan_epochs=400 --gta_loss_factor=0.0001 --defense_method=none
```

2. UGBA backdoor attack on the Cora dataset

```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=UGBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0  --defense_method=none
```


### Attribute Over-emphasis

1. GCBA backdoor attack on the Cora dataset

```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=GCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=none
```

2. PerCBA backdoor attack on the Cora dataset

```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=PerCBA --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=none
```

## E2: Performance on Various Attacks

### Cora Dataset

1. DShield against UGBA attack 

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=UGBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```


2. DShield against LGCB attack 

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=LGCB --lgcb_num_budgets=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

3. DShield against GCBA attack

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=GCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1
```

4. DShield against PerCBA attack

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=PerCBA --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=10
```

### PubMed Dataset

1. DShield against UGBA attack 

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=3 --vs_number=40 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=UGBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

2. DShield against LGCB attack 

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=3 --vs_number=40 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=LGCB --lgcb_num_budgets=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

3. DShield against GCBA attack 

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=3 --vs_number=40 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=GCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

4. DShield against PerCBA attack 

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=5 --vs_number=60 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=PerCBA --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=DShield --dshield_pretrain_epochs=800 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=5
```


## E3: Performance on Adaptive Attacks

1. DShield against UGBA+LGCB attack on the Cora dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=20 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=UGBA-LGCB --UGBA_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --lgcb_num_budgets=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

2. DShield against UGBA+GCBA attack on the Cora dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=20 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=UGBA-GCBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1
```

3. DShield against GCBA+PerCBA attack on the Cora dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=20 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=GCBA-PerCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1
```

4. DShield against AdaDA attacks on the Cora dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=AdaDA --adada_thrd=0.5 --adada_trojan_epochs=200 --adada_inner_epochs=5 --adada_target_loss_weight=5 --adada_homo_loss_weight=50 --adada_homo_boost_thrd=1.0 --adaba_reg_loss_weight=100 --adaba_ssl_tau=0.2 --adaba_edge_drop_ratio=0.2  --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.05 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

5. DShield against AdaCA attacks on the Cora dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=AdaCA --adaca_num_hidden=512 --adaca_feat_budget=100 --adaca_trojan_epochs=300 --adaca_umap_epochs=10 --adaca_ssl_tau=0.8 --adaca_tau=0.2 --adaca_edge_drop_ratio=0.5 --adaca_reg_loss_weight=50 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1
```

6. DShield against UGBA+GCBA attack on the PubMed dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=3 --vs_number=80 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=UGBA-GCBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=0.1 
```

7. DShield against UGBA+GCBA attack on the Flickr dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=Flickr --benign_epochs=200 --trigger_size=3 --vs_number=160 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=UGBA-LGCB --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --lgcb_num_budgets=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

8. DShield against UGBA+LGCB attack on the OGBN-arXiv dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=ogbn-arxiv --benign_epochs=200 --trigger_size=3 --vs_number=320 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=UGBA-LGCB --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --lgcb_num_budgets=100 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

### E4: Attacks on Graph Classification tasks

1. DShield against G-SBA attacks on the ENZYMES dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=ENZYMES --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --attack_method=SBA --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_neg_epochs=100 --dshield_kappa1=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

2. DShield against G-EBA attacks on the ENZYMES dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=ENZYMES --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --attack_method=ExplainBackdoor --eb_trig_feat_val=-1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_neg_epochs=100 --dshield_kappa1=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

3. DShield against G-GCBA attacks on the ENZYMES dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=ENZYMES --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --attack_method=GCBA --eb_trig_feat_val=-1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_neg_epochs=100 --dshield_kappa1=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

4. DShield against G-SBA attacks on the PROTEINS dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=PROTEINS --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --attack_method=SBA --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```

5. DShield against G-SBA attacks on the MNIST dataset

```Python
python main.py --seed=1027 --model=GCN --dataset=MNIST --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --batch_size=256 --attack_method=SBA --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5
```


## Citation

If you find this work useful, please consider citing it:

```latex
@inproceedings{yu2024dshield,
  author       = {Hao Yu and
                  Chuan Ma and
                  Xinhang Wan and
                  Jun Wang and
                  Tao Xiang and
                  Meng Shen and
                  Xinwang Liu},
  title        = {DShield: Defending against Backdoor Attacks on Graph Neural Networks via Discrepancy Learning},
  booktitle    = {{NDSS}},
  publisher    = {The Internet Society},
  year         = {2025}
}
```