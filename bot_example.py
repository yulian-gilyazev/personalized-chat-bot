import transformers

import argparse
import json

from petals.client.remote_model import DistributedBloomForCausalLM

from personalized_chat_bot import PersonalizedChatBot, PersonalityManager
from models.personality_clustering import PersonalityClustering

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return argparse.Namespace(**config)


def main():
    greating = 'Describe the person you want to talk:'
    print(greating)
    persona_description = input()
    print('Cool! wait a few seconds...')
    personality_clustering = PersonalityClustering()
    personality_clustering.load('./data/models/personality_clustering_500_paraphrase-MiniLM-L6-v2_k-means.pkl')

    hook = lambda dct: {int(k): v for k, v in dct.items()}
    with open('prompt_paths.json', 'r') as f:
        prompt_paths = json.load(f, object_hook=hook)

    pm = PersonalityManager(prompt_paths, personality_clustering)
    prompt_path, closest_persona = pm.get_prompt(persona_description)
    print(f'The closest personality is: {closest_persona}')
    print('Wait a little longer...')
    config = load_config('./scripts/config_176b.json')

    model = DistributedBloomForCausalLM.from_pretrained(
        config.MODEL_NAME,
        pre_seq_len=config.NUM_PREFIX_TOKENS,
        tuning_mode=config.TUNING_MODE
    ).to(config.DEVICE)

    generation_config = load_config('generation_config.json')

    tokenizer = transformers.BloomTokenizerFast.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = config.MODEL_MAX_LENGTH

    chatbot = PersonalizedChatBot(model, tokenizer, generation_config=generation_config)
    chatbot.load_prompt(prompt_path)
    print('Done! You can start a dialogue.')
    try:
        while True:
            text = input('You: ')
            answer = chatbot.answer(text)
            print(f'Bloom: {answer}')
    except KeyboardInterrupt:
        print('Thank you for the conversation!')


if __name__ == '__main__':
    main()