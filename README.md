The pre-trained models to be pulled, models--FacebookAI--roberta-base and models--google-bert--bert-base-uncased, are used for contextual_augment in the nlpaug_explore.py script

The pre-trained model to be pulled, models--sentence-transformers--all-mpnet-base-v2, serves as the backbone of the framework.

The torch installation environment and the execution parameters for data augmentation can be found in the comments in nlpaug_explore.py

The script for running contrastive learning can be found in main.py, and the hyperparameters for running learning can be found in the script comments.
