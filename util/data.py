from collections import defaultdict
import itertools
from torch.utils.data import Dataset
import numpy as np
import torch
from joblib import Parallel, delayed


class Separator:
    @classmethod
    def join(cls, dialogue):
        with_prefix = []
        for i, utterance in enumerate(dialogue):
            pref = cls.before_first if i % 2 == 0 else cls.before_second
            with_prefix.append(pref + utterance)
        return cls.between.join(with_prefix)

    @classmethod
    def split(cls, text):
        with_prefix = text.split(cls.between)
        dialogue = []
        for i, utterance in enumerate(with_prefix):
            pref = cls.before_first if i % 2 == 0 else cls.before_second
            if not utterance.startswith(pref):
                return None
            dialogue.append(utterance[len(pref):])
        return dialogue


class DashSeparator(Separator):
    before_first = '--'
    before_second = '--'
    between = '\n'


class ABSeparator(Separator):
    before_first = 'A: '
    before_second = 'B: '
    between = '\n'


class NewlineSeparator(Separator):
    before_first = ''
    before_second = ''
    between = '\n'


class OnePersonaDataset(Dataset):
    def __init__(self, data: Dataset, tokenizer, joining_fn,
                 transforms=None, positive_candidates=True, n_jobs=8):
        super().__init__()

        self.data = data
        if len(data) == 0:
            self.input_ids = []
            self.history = []
            self.labels = []
            return

        if positive_candidates:
            self.history = [row['history'] + [row['candidates'][-1], ] for row in data]
            self.labels = np.ones(len(self.history), dtype=int)
        else:
            self.history = [row['history'] + [candidate, ] for row in data
                            for candidate in row['candidates']]
            self.labels = itertools.chain.from_iterable([0] * (len(row['candidates']) - 1) + [1]
                                                        for row in data)
            self.labels = np.array(self.labels, dtype=int)

        if transforms:
            self.history = Parallel(n_jobs=n_jobs)(delayed(transforms)(item) for item in self.history)

        self.history = [joining_fn(item) for item in self.history]
        self.input_ids = tokenizer(self.history, padding='max_length', truncation=True)["input_ids"]

        self.__indexes = np.arange(0, self.__len__())

    def shuffle(self):
        np.random.shuffle(self.__indexes)

    def __getitem__(self, idx):
        idx = self.__indexes[idx]
        return {'input_ids': self.input_ids[idx],
                'labels': self.input_ids[idx],
                'example': self.history[idx],
                'class': self.labels[idx]}

    def __len__(self):
        return len(self.data)


class PersonaChatDataset(Dataset):
    DEFAULT_DATASET_NAME = "bavard/personachat_truecased"

    def __init__(self, clustering, dataset, tokenizer, joining_fn):
        super().__init__()

        self.dataset = dataset
        self.clustering = clustering
        self.tokenizer = tokenizer
        self.joining_fn = joining_fn

        all_personalities = list(set([sent for item in self.dataset
                                      for sent in item['personality']]))
        predicted_centers = self.clustering.predict(all_personalities)
        self.all_personalities_to_id = {persona: center
                                        for persona, center in zip(all_personalities, predicted_centers)}
        self.personalities = self.clustering._cluster_centers

        subdataset_data_by_personality = [[] for _ in range(len(self.personalities))]

        for i in range(len(self.dataset)):
            item = self.dataset[i]
            cur_persona_ids = [self.all_personalities_to_id[persona] for persona in item['personality']]
            for persona_id in cur_persona_ids:
                subdataset_data_by_personality[persona_id].append(item)

        self.subdataset_data_by_personality = subdataset_data_by_personality

    def __getitem__(self, persona_id):
        dset = OnePersonaDataset(self.subdataset_data_by_personality[persona_id], self.tokenizer, self.joining_fn)
        return dset

    def get_multipersona_dataset(self, persona_ids):
        subdatas = [self.subdataset_data_by_personality[i] for i in persona_ids]
        data = [item for subdata in subdatas for item in subdata]
        dset = OnePersonaDataset(data, self.tokenizer, self.joining_fn)
        return dset

    def __len__(self, ):
        return len(self.datasets)


class PersonaChatDataloader:
    def __init__(self, dset, batch_size, drop_last, shuffle):
        self.dset = dset
        assert len(dset) > 0, 'Empty dataset!'
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dset_len = len(self.dset)
        self.indexes = np.arange(0, self.dset_len)
        self.cur_pointer = None

    @staticmethod
    def __combine(items):
        if isinstance(items[0], np.ndarray):
            return np.stack(items)
        elif isinstance(items[0], torch.Tensor):
            return torch.stack(items)
        else:
            return items

    def __shuffle(self):
        np.random.shuffle(self.indexes)

    def __next__(self):
        if self.cur_pointer >= self.dset_len:
            raise StopIteration
        last_pointer = self.cur_pointer + self.batch_size
        if last_pointer > self.dset_len:
            if self.drop_last:
                raise StopIteration
            last_pointer = self.dset_len
        if isinstance(self.dset[0], dict):
            batch_dict = defaultdict(lambda: [])
            items = [self.dset[self.indexes[i]] for i in range(self.cur_pointer, last_pointer)]
            for item in items:
                for key, value, in item.items():
                    batch_dict[key].append(value)
            to_return = {key: self.__combine(value) for key, value in batch_dict.items()}
        else:
            items = [self.dset[self.indexes[i]] for i in range(self.cur_pointer, last_pointer)]
            to_return = self.__combine(items)
        self.cur_pointer = last_pointer
        return to_return

    def __iter__(self, ):
        self.cur_pointer = 0
        if self.shuffle:
            self.__shuffle()
        return self
