import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pickle


class PersonalityClustering:
    DEFAULT_SENTENCE_TRANSFORMER = 'paraphrase-MiniLM-L6-v2'

    @property
    def sentence_transformer(self):
        """Ленивая инициализация sentence_transformer."""
        if not self.__sentence_transformer:
            self.__sentence_transformer = SentenceTransformer(self.model_name,  device=self.device)
        return self.__sentence_transformer

    @property
    def clustering(self):
        """Ленивая инициализация кластеризации."""
        if not self.__clustering:
            self.__clustering = KMeans(n_clusters=self.n_clusters)
        return self.__clustering

    def __init__(self, n_clusters=None, device='cpu', model_name=None):
        if model_name is None:
            self.model_name = self.DEFAULT_SENTENCE_TRANSFORMER
        else:
            self.model_name = model_name
        self.device = device
        self.n_clusters = n_clusters
        self._cluster_centers = None
        self.__clustering = None
        self.__sentence_transformer = None

    def load(self, path):
        with open(path, "rb") as f:
            self.__clustering, self._cluster_centers = pickle.load(f)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.__clustering, self._cluster_centers), f)

    def fit(self, personalities):
        personalities = np.array(list(personalities))
        train_embeddings = self.sentence_transformer.encode(personalities)
        clusters = self.clustering.fit_predict(train_embeddings)
        persona_cluster_centers = []
        for clust, center in enumerate(self.clustering.cluster_centers_):
            cur_clust_embed = train_embeddings[clusters == clust]
            cur_clust_personalities = personalities[clusters == clust]
            min_distance_to_center = np.inf
            persona_center = None
            for embed, persona in zip(cur_clust_embed, cur_clust_personalities):
                cur_distance_to_center = np.linalg.norm(embed - center)
                if cur_distance_to_center < min_distance_to_center:
                    min_distance_to_center = cur_distance_to_center
                    persona_center = persona
            persona_cluster_centers.append(persona_center)
        self._cluster_centers = np.array(persona_cluster_centers)
        return self

    def predict(self, personalities):
        personalities = np.array(list(personalities))
        embeddings = self.sentence_transformer.encode(personalities)
        clusters = self.clustering.predict(embeddings)
        return clusters

    def predict_nearest_personality(self, personalities):
        clusters = self.predict(personalities)
        return np.array([self._cluster_centers[clust] for clust in clusters])

    def fit_predict(self, personalities):
        self.fit(personalities)
        return self.predict(personalities)
