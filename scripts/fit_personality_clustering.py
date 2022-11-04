import argparse
from datasets import load_dataset
from models.personality_clustering import PersonalityClustering
import os

"""Пример запуска
python -m  scripts.fit_personality_clustering --clustering-path data/models --n-clusters 500
"""

PERSONACHAT_DATASET = "bavard/personachat_truecased"


def load_persona_chat_personalities(personachat_dataset):
    dataset = load_dataset(personachat_dataset)
    train_personalities = [sent for persona in dataset['train']['personality']
                           for sent in persona]
    test_personalities = [sent for persona in dataset['train']['personality']
                          for sent in persona]
    personalities = list(set(train_personalities) | set(test_personalities))
    return personalities


def parse_args(args=None):
    parser = argparse.ArgumentParser(add_help=True, description="Class for personality clustering.")

    parser.add_argument('-clustering-path', '--clustering-path', type=str,
                        help='Path to clustering data.')
    parser.add_argument('-n-clusters', '--n-clusters', type=int, default=500,
                        help='The number of clusters to form.')
    parser.add_argument('-model-name', '--model-name', type=str, default=None, required=False)
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args()
    personalities = load_persona_chat_personalities(PERSONACHAT_DATASET)
    print('Data loaded')
    model = PersonalityClustering(n_clusters=args.n_clusters)
    print('Model fitting')
    model.fit(personalities)
    print('Model fitted')
    if args.model_name is None:
        model_name = f'personality_clustering_{model.n_clusters}_{model.model_name}_k-means.pkl'
    else:
        model_name = args.model_name
    model.save(os.path.join(args.clustering_path, model_name))
    print(f'{model_name} saved')


if __name__ == '__main__':
    main()