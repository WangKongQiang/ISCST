The pre-trained models to be pulled, models--FacebookAI--roberta-base and models--google-bert--bert-base-uncased, are used for contextual_augment function in the nlpaug_explore.py script.

Hugging Face: https://huggingface.co/FacebookAI/roberta-base

Hugging Face: https://huggingface.co/google-bert/bert-base-uncased

The pre-trained model to be pulled, models--sentence-transformers--all-mpnet-base-v2, serves as the backbone of the framework.

Hugging Face: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

The torch installation environment and the execution parameters for data augmentation can be found in the comments in nlpaug_explore.py

### setup torch
### pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

### Dataset Content (It contains datasets of the corresponding data augmentation type).
### agnews.csv                                    googlenews_T.csv
### agnews_trans_subst_10.csv                     googlenews_T_trans_subst_10.csv
### agnews_trans_subst_10_charswap_10.csv         googlenews_T_trans_subst_10_charswap_10.csv
### agnews_trans_subst_20.csv                     googlenews_T_trans_subst_20.csv
### agnews_trans_subst_20_charswap_20.csv         googlenews_T_trans_subst_20_charswap_20.csv
### agnews_word_deletion_10.csv                   googlenews_T_word_deletion_10.csv
### agnews_word_deletion_20.csv                   googlenews_T_word_deletion_20.csv
### biomedical.csv                                searchsnippets.csv
### biomedical_trans_subst_10.csv                 searchsnippets_trans_subst_10.csv
### biomedical_trans_subst_10_charswap_10.csv     searchsnippets_trans_subst_10_charswap_10.csv
### biomedical_trans_subst_20.csv                 searchsnippets_trans_subst_20.csv
### biomedical_trans_subst_20_charswap_20.csv     searchsnippets_trans_subst_20_charswap_20.csv
### biomedical_word_deletion_10.csv               searchsnippets_word_deletion_10.csv
### biomedical_word_deletion_20.csv               searchsnippets_word_deletion_20.csv
### googlenews_S.csv                              stackoverflow.csv
### googlenews_S_trans_subst_10.csv               stackoverflow_trans_subst_10.csv
### googlenews_S_trans_subst_10_charswap_10.csv   stackoverflow_trans_subst_10_charswap_10.csv
### googlenews_S_trans_subst_20.csv               stackoverflow_trans_subst_20.csv
### googlenews_S_trans_subst_20_charswap_20.csv   stackoverflow_trans_subst_20_charswap_20.csv
### googlenews_S_word_deletion_10.csv             stackoverflow_word_deletion_10.csv
### googlenews_S_word_deletion_20.csv             stackoverflow_word_deletion_20.csv
### googlenews_TS.csv                             tweet.csv
### googlenews_TS_trans_subst_10.csv              tweet_trans_subst_10.csv
### googlenews_TS_trans_subst_10_charswap_10.csv  tweet_trans_subst_10_charswap_10.csv
### googlenews_TS_trans_subst_20.csv              tweet_trans_subst_20.csv
### googlenews_TS_trans_subst_20_charswap_20.csv  tweet_trans_subst_20_charswap_20.csv
### googlenews_TS_word_deletion_10.csv            tweet_word_deletion_10.csv
### googlenews_TS_word_deletion_20.csv            tweet_word_deletion_20.csv
### agnews_charswap_10.csv      googlenews_S_charswap_10.csv   googlenews_T_charswap_10.csv    stackoverflow_charswap_10.csv
### agnews_charswap_20.csv      googlenews_S_charswap_20.csv   googlenews_T_charswap_20.csv    stackoverflow_charswap_20.csv
### biomedical_charswap_10.csv  googlenews_TS_charswap_10.csv  searchsnippets_charswap_10.csv  tweet_charswap_10.csv
### biomedical_charswap_20.csv  googlenews_TS_charswap_20.csv  searchsnippets_charswap_20.csv  tweet_charswap_20.csv


The script for running contrastive learning can be found in main.py, and the hyperparameters for running contrastive learning can be found in the script comments.

### Run the clustering experiment script: (For example, take the experiment with trans_subst_20_charswap_20 data augmentation as a positive example)
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname searchsnippets_trans_subst_20_charswap_20 --num_classes 8
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname stackoverflow_trans_subst_20_charswap_20 --num_classes 20
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname biomedical_trans_subst_20_charswap_20 --num_classes 20
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname agnews_trans_subst_20_charswap_20 --num_classes 4
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname tweet_trans_subst_20_charswap_20 --num_classes 110
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname googlenews_TS_trans_subst_20_charswap_20 --num_classes 152
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname googlenews_T_trans_subst_20_charswap_20 --num_classes 152
### python main.py  --objective SCCL --augtype explicit --eta 10 --batch_size 400 --max_iter 3000 --bert mpnet --dataname googlenews_S_trans_subst_20_charswap_20 --num_classes 152
