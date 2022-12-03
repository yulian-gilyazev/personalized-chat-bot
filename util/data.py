import itertools
from torch.utils.data import Dataset
import numpy as np
from joblib import Parallel, delayed


class OnePersonaDataset(Dataset):
    def __init__(self, data, tokenizer, transforms=None, positive_candidates=True, n_jobs=8):
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

        if transforms is None:
            self.history = ["\n".join(item) for item in self.history]
        else:
            self.history = Parallel(n_jobs=n_jobs)(delayed(transforms)(item) for item in self.history)
        self.input_ids = tokenizer(self.history, padding='max_length', truncation=True)["input_ids"]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.history[idx], self.labels[idx]

    # todo: shuffle

    def __len__(self):
        return len(self.data)


class PersonaChatDataset(Dataset):
    DEFAULT_DATASET_NAME = "bavard/personachat_truecased"

    def __init__(self, clustering, dataset, tokenizer, dataset_name=None):
        super().__init__()

        self.dataset = dataset
        self.clustering = clustering

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

        self.subdatasets = [OnePersonaDataset(cur_data, tokenizer) for cur_data in subdataset_data_by_personality]

    def __getitem__(self, persona_id):
        return self.subdatasets[persona_id]

    def __len__(self, ):
        return len(self.datasets)
