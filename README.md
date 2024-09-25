
# DShield: Defending against Backdoor Attacks on Graph Neural Networks via Discrepancy Learning

## Settings

Python = 3.10.11


## Case Study

### Semantic Drift

```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=GTA --gta_thrd=0.5 --gta_lr=0.01 --gta_trojan_epochs=400 --gta_loss_factor=0.0001 --defense_method=none --instance="Cora-GTA-Non_Defense(1027)" --wandb_group=Effectiveness
```

```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=UGBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0  --defense_method=none --instance="Cora-UGBA-Non_Defense(1027)" --wandb_group=Effectiveness
```


### Attribute Over-emphasis


```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=GCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=none --instance="Cora-GCBA-Non_Defense(1027)" --wandb_group=Effectiveness
```


```Python
python viz_main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=PerCBA --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=none --instance="Cora-PerCBA-Non_Defense(1027)" --wandb_group=Effectiveness
```

## Performance on various attacks

### Cora Dataset

1. UGBA & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=UGBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="Cora-UGBA-DShield(1027)" --wandb_group=Effectiveness
```


2. LGCB & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=LGCB --lgcb_num_budgets=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="Cora-LGCB-DShield(1027)" --wandb_group=Effectiveness
```

3. GCBA & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=GCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1 --instance="Cora-GCBA-DShield(1027)" --wandb_group=Effectiveness
```

4. PerCBA & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=PerCBA --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=200 --dshield_neg_epochs=100 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=10 --instance="Cora-PerCBA-DShield(1027)" --wandb_group=Effectiveness
```

### PubMed Dataset

1. Pubmed dataset & UGBA & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=3 --vs_number=40 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=UGBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="Pubmed-UGBA-DShield(1027)" --wandb_group=Effectiveness
```

2. PubMed dataset & LGCB & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=3 --vs_number=40 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=LGCB --lgcb_num_budgets=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="Pubmed-LGCB-DShield(1027)" --wandb_group=Effectiveness
```

3. GCBA & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=3 --vs_number=40 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=GCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1 --instance="Pubmed-GCBA-DShield(1027)" --wandb_group=Effectiveness
```

4. PerCBA & DShield

```Python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --benign_epochs=200 --trigger_size=5 --vs_number=60 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=PerCBA --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=DShield --dshield_pretrain_epochs=800 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=5 --instance="Pubmed-PerCBA-DShield(1027)" --wandb_group=Effectiveness
```


## Performance on Adaptive Attacks

1. UGBA-LGCB

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=20 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=UGBA-LGCB --UGBA_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --lgcb_num_budgets=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.05 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="Cora-UGBA-LGCB-DShield(1027)" --wandb_group=Effectiveness
```


2. UGBA-GCBA

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=20 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=UGBA-GCBA --ugba_thrd=0.5 --ugba_trojan_epochs=200 --ugba_inner_epochs=5 --ugba_target_loss_weight=5 --ugba_homo_loss_weight=50 --ugba_homo_boost_thrd=1.0 --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1 --instance="Cora-UGBA-GCBA-DShield(1027)" --wandb_group=Effectiveness
```

3. GCBA-PerCBA

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=20 --use_vs_number --target_class=1-2 --selection_method=mixture --attack_method=GCBA-PerCBA --gcba_num_hidden=512 --gcba_feat_budget=100 --gcba_trojan_epochs=300 --gcba_ssl_tau=0.8 --gcba_tau=0.2 --gcba_edge_drop_ratio=0.5 --percba_trojan_epochs=300 --percba_perturb_epochs=200 --percba_mu=0.01 --percba_eps=0.5 --percba_feat_budget=200 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1 --instance="Cora-GCBA-PerCBA-DShield(1027)" --wandb_group=Effectiveness
```

4. Adaptive dirty-label attacks

```Python
python main.py --seed=1028 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=cluster_degree --attack_method=AdaDA --adada_thrd=0.5 --adada_trojan_epochs=200 --adada_inner_epochs=5 --adada_target_loss_weight=5 --adada_homo_loss_weight=50 --adada_homo_boost_thrd=1.0 --adaba_reg_loss_weight=100 --adaba_ssl_tau=0.2 --adaba_edge_drop_ratio=0.2  --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.05 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="Cora-AdaDA-DShield(1028)" --wandb_group=Robustness
```

5. Adaptive clean-label attacks

```Python
python main.py --seed=1027 --model=GCN --dataset=Cora --benign_epochs=200 --trigger_size=3 --vs_number=10 --use_vs_number --target_class=1 --selection_method=clean_label --attack_method=AdaCA --adaca_num_hidden=512 --adaca_feat_budget=100 --adaca_trojan_epochs=300 --adaca_umap_epochs=10 --adaca_ssl_tau=0.8 --adaca_tau=0.2 --adaca_edge_drop_ratio=0.5 --adaca_reg_loss_weight=50 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_kappa1=5 --dshield_kappa2=5 --dshield_kappa3=0.01 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=1 --instance="Cora-AdaCA-DShield(1027)" --wandb_group=Robustness
```


### Attacks on graph classification tasks

1. G-SBA

```Python
python main.py --seed=1027 --model=GAT --dataset=ENZYMES --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --attack_method=SBA --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_neg_epochs=100 --dshield_kappa1=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="ENZYMES-SBA-DShield(1027)" --wandb_group=Effectiveness
```

2. G-EBA

```Python
python main.py --seed=1027 --model=GCN --dataset=ENZYMES --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --attack_method=ExplainBackdoor --eb_trig_feat_val=-1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_neg_epochs=100 --dshield_kappa1=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="ENZYMES-ExplainBackdoor-DShield(1027)" --wandb_group=Effectiveness
```

3. G-GCBA

```Python
python main.py --seed=1027 --model=GCN --dataset=ENZYMES --benign_epochs=200 --trigger_size=20 --vs_ratio=0.1 --target_class=1 --attack_method=GCBA --eb_trig_feat_val=-1.0 --defense_method=DShield --dshield_pretrain_epochs=400 --dshield_finetune_epochs=400 --dshield_classify_epochs=400 --dshield_neg_epochs=100 --dshield_kappa1=0.1 --dshield_edge_drop_ratio=0.20 --dshield_feature_drop_ratio=0.20 --dshield_tau=0.9 --dshield_balance_factor=0.5 --dshield_classify_rounds=1 --dshield_thresh=2.5 --instance="ENZYMES-GCBA-DShield(1027)" --wandb_group=Effectiveness
```