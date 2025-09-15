The pre-trained models to be pulled, models--FacebookAI--roberta-base and models--google-bert--bert-base-uncased, are used for contextual_augment in the nlpaug_explore.py script

Hugging Face: https://huggingface.co/google-bert/bert-base-uncased

Hugging Face: https://huggingface.co/FacebookAI/roberta-base

The pre-trained model to be pulled, models--sentence-transformers--all-mpnet-base-v2, serves as the backbone of the framework.

Hugging Face: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

The torch installation environment and the execution parameters for data augmentation can be found in the comments in nlpaug_explore.py

### setup torch
### pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

The script for running contrastive learning can be found in main.py, and the hyperparameters for running contrastive learning can be found in the script comments.

### Run the clustering experiment script:
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname searchsnippets_trans_subst_20_charswap_20 --num_classes 8
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname stackoverflow_trans_subst_20_charswap_20 --num_classes 20
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname biomedical_trans_subst_20_charswap_20 --num_classes 20
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname agnews_trans_subst_20_charswap_20 --num_classes 4
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname tweet_trans_subst_20_charswap_20 --num_classes 110
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname googlenews_TS_trans_subst_20_charswap_20 --num_classes 152
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname googlenews_T_trans_subst_20_charswap_20 --num_classes 152
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname googlenews_S_trans_subst_20_charswap_20 --num_classes 152
