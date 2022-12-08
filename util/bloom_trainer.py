import sys

import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch


from util.metrics import perplexity


class BloomTrainer:
    DEFAULT_VAL_FREQ = 5
    ITERATION_LIMIT = 50

    def __init__(self, model, config, train_dataset, val_dataset, wandb_run=None, prompt_path=None, val_freq=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.wandb_run = wandb_run
        self.val_freq = val_freq
        if self.val_freq is None:
            self.val_freq = self.DEFAULT_VAL_FREQ
        self.prompt_path = prompt_path

        self.best_loss = np.inf

        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=config.BATCH_SIZE,
                                       drop_last=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     shuffle=True,
                                     batch_size=config.BATCH_SIZE,
                                     drop_last=False)

        self.optimizer = AdamW(self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps= len(self.train_loader) * self.config.N_EPOCH
        )

    def train(self):
        self.model.train()
        iter_counter = 0
        for epoch in range(self.config.N_EPOCH):
            for batch in self.train_loader:
                batch = {'input_ids': torch.stack(batch['input_ids']).T.to(self.config.DEVICE),
                         'labels': torch.stack(batch['labels']).T.to(self.config.DEVICE)}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.wandb_run.log({'loss': loss})
                iter_counter += 1
                if (iter_counter + 1) % self.val_freq == 0:
                    eval_perplexity = self.evaluate(perplexity)
                    self.wandb_run.log({'perplexity': eval_perplexity})
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        self.save_model(self.prompt_path)
                        print('Model saved')
                if iter_counter >= self.ITERATION_LIMIT:
                    return

    def evaluate(self, eval_fn):
        logits = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {'input_ids': torch.stack(batch['input_ids']).T.to(self.config.DEVICE),
                         'labels': torch.stack(batch['labels']).T.to(self.config.DEVICE)}
                outputs = self.model(**batch)
                labels.extend(batch['input_ids'])
                logits.extend(outputs.logits)
        metric = eval_fn(logits, labels)
        return metric

    def save_model(self, path):
        torch.save(self.model.transformer.prompt_embeddings.state_dict(), path)

    def load_model(self, path):
        self.model.transformer.prompt_embeddings.load_state_dict(torch.load(path))
