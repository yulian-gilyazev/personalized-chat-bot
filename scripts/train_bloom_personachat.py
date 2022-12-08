import argparse

import torch.cuda
from datasets import load_dataset
import json
import os
import transformers
from torch.utils.data import Subset
import wandb
import numpy as np
import gc

from models.personality_clustering import PersonalityClustering
from util.bloom_trainer import BloomTrainer
from util.data import PersonaChatDataset
from util.metrics import perplexity

from petals.client.remote_model import DistributedBloomForCausalLM

"""Пример запуска
python -m scripts.train_bloom_personachat --persona-ids 6 --config scripts/config.json --prompt-path data/models/
"""

DEFAULT_CLUSTERING_MODEL = './data/models/personality_clustering_500_paraphrase-MiniLM-L6-v2_k-means.pkl'
MAX_VAL_DATA_SIZE = 4


def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return argparse.Namespace(**config)


def main():
    args = parse_args()
    persona_clustering = PersonalityClustering()
    persona_clustering.load(args.clustering_model_path)

    config = load_config(args.config)

    tokenizer = transformers.BloomTokenizerFast.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = config.PADDING_SIDE
    tokenizer.model_max_length = config.MODEL_MAX_LENGTH

    dataset = load_dataset(config.PERSONACHAT_DATASET_NAME)
    personachat_train_dataset = PersonaChatDataset(persona_clustering,
                                                   dataset['train'],
                                                   tokenizer)
    personachat_val_dataset = PersonaChatDataset(persona_clustering,
                                                 dataset['validation'],
                                                 tokenizer)

    for id in args.persona_ids:
        prompt_path = os.path.join(args.prompt_path, f'{id}_persona_prompt_embedding.pt')
        train_dataset = personachat_train_dataset[id]
        val_dataset = personachat_val_dataset[id]
        honest_validation = True
        if len(val_dataset) < 4:
            val_dataset = personachat_train_dataset[id]
            honest_validation = False
        # для ускорения обрежем размер валидации до некоторой границы
        if len(val_dataset) > MAX_VAL_DATA_SIZE:
            subset_indexes = np.random.choice(len(val_dataset), MAX_VAL_DATA_SIZE, replace=False)
            val_dataset = Subset(val_dataset, subset_indexes)
        # train_dataset.shuffle()

        wandb_run = wandb.init(
            project=args.wandb_project,
            config={
                'lr': config.LR,
                'batch_size': config.BATCH_SIZE,
                'persona_id': id,
                'device': config.DEVICE,
                'model_name': config.MODEL_NAME,
                'n_epoch': config.N_EPOCH,
                'honest_validation': honest_validation
            },
            name=f'id{id}',
            reinit=True
        )
        if len(config.INITIAL_PEERS) == 0:
            model = DistributedBloomForCausalLM.from_pretrained(
                config.MODEL_NAME,
                pre_seq_len=config.NUM_PREFIX_TOKENS,
                tuning_mode=config.TUNING_MODE
            ).to(config.DEVICE)
        else:
            model = DistributedBloomForCausalLM.from_pretrained(
                config.MODEL_NAME,
                initial_peers=config.INITIAL_PEERS,
                pre_seq_len=config.NUM_PREFIX_TOKENS,
                tuning_mode=config.TUNING_MODE
            ).to(config.DEVICE)

        trainer = BloomTrainer(model, config, train_dataset, val_dataset, wandb_run, prompt_path)
        trainer.train()
        eval_perplexity = trainer.evaluate(perplexity)
        trainer.save_model(prompt_path)
        wandb_run.log({'perplexity': eval_perplexity, 'model_path': prompt_path})

        del model
        gc.collect()
        torch.cuda.empty_cache()


def parse_args(args=None):
    parser = argparse.ArgumentParser(add_help=True,
                                     description="bloom training script")
    parser.add_argument('--persona-ids', type=int, nargs='+',
                        help='Ids of persona')
    parser.add_argument('-clustering-model-path', '--clustering-model-path', type=str,
                        default=DEFAULT_CLUSTERING_MODEL,
                        help='Path to clustering model')
    parser.add_argument('--config', type=str, help='Path to training config file')
    parser.add_argument('--prompt-path', type=str,
                        help='Path to dir with trained soft prompts')
    parser.add_argument('--wandb-project', type=str, default='test_bloom_personachat_176b_v3')
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    main()