import argparse
import json
import torch
from sklearn.neighbors import KDTree


class PersonalityManager:
    def __init__(self, prompt_paths, personality_clustering):
        self.prompt_paths = prompt_paths
        self.personality_clustering = personality_clustering

        self.persona_ids = list(prompt_paths.keys())
        self.personalities = [personality_clustering._cluster_centers[i]
                              for i in self.persona_ids]

        self.embeddings = personality_clustering.sentence_transformer.encode(self.personalities)
        self._nearest_neighbours = KDTree(self.embeddings, metric='euclidean')

    def get_prompt(self, description):
        embedding = self.personality_clustering.sentence_transformer.encode([description])
        dist, ind = self._nearest_neighbours.query(embedding, k=1)
        persona_id = self.persona_ids[ind[0][0]]
        prompt_path = self.prompt_paths[persona_id]
        cluster_center = self.personality_clustering._cluster_centers[persona_id]
        return prompt_path, cluster_center


class PersonalizedChatBot:
    def __init__(self, model, tokenizer, prompt_path=None, generation_config=None):
        self.model = model
        if prompt_path is not None:
            self.load_prompt(prompt_path)
        self.tokenizer = tokenizer
        self.separator = '\n'
        self.dialog = ''
        self.generation_config = generation_config

    def load_prompt(self, path):
        self.model.transformer.prompt_embeddings.load_state_dict(torch.load(path))

    def load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        self.generation_config = argparse.Namespace(**config)

    def reset_dialog(self, ):
        self.dialog = ''

    def answer(self, phrase):
        if len(phrase) == 0:
            return
        self.dialog += f"{phrase}{self.separator}"
        inputs = self.tokenizer([self.dialog], return_tensors='pt')['input_ids']
        outputs = self.model.generate(
            inputs,
            temperature=self.generation_config.TEMPERATURE,
            do_sample=True,
            top_k=self.generation_config.TOP_K,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.generation_config.MAX_TOKENS,
        )
        bloom_answer = self.tokenizer.batch_decode(outputs)[0]
        bloom_answer = bloom_answer[len(self.dialog):].split("\n")[0]
        self.dialog += f"{bloom_answer}{self.separator}"
        return bloom_answer