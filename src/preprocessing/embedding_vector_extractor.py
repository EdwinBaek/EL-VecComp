import os
import csv
import numpy as np
from gensim.models import FastText, Word2Vec

def load_embedding_model(model_path):
    if 'fasttext' in model_path.lower():
        return FastText.load(model_path)
    elif 'graph2vec' in model_path.lower():
        model = np.load(model_path, allow_pickle=True)
        print(f"Graph2Vec model shape: {model.shape}")
        print(f"Graph2Vec model type: {type(model)}")
        if model.ndim == 0:
            model = model.item()  # np.ndarray 내부의 객체에 접근
        return model
    else:
        return Word2Vec.load(model_path)

def get_vocabulary_vectors(model):
    if isinstance(model, np.ndarray):
        if model.ndim == 2:
            return {f"graph_{i}": vec for i, vec in enumerate(model)}
        elif model.ndim == 1:
            return {f"graph_0": model}
        else:
            raise ValueError(f"Unexpected shape for Graph2Vec model: {model.shape}")
    elif isinstance(model, dict):
        return model  # 이미 dictionary 형태라면 그대로 반환
    else:
        vocab = model.wv.key_to_index
        vectors = {word: model.wv[word] for word in vocab}
        return vectors

def save_vectors_to_csv(vectors, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['identifier'] + [f'dim_{i}' for i in range(len(next(iter(vectors.values()))))])
        for identifier, vector in vectors.items():
            writer.writerow([identifier] + vector.tolist())

def main(model_dir, output_dir):
    print(f"Starting embedding extraction from {model_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(model_dir):
        if filename.endswith('wv.vectors_ngrams.npy'):
            continue

        if filename.endswith(('.model', '.npy')):
            model_path = os.path.join(model_dir, filename)
            print(f"Loading model: {filename}")
            try:
                model = load_embedding_model(model_path)
                print("Extracting vectors...")
                vectors = get_vocabulary_vectors(model)
                output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_vectors.csv")
                print(f"Saving vectors to: {output_file}")
                save_vectors_to_csv(vectors, output_file)
                print(f"Completed processing for {filename}")
            except Exception as e:
                print(f"Error processing {filename}:")
                import traceback
                traceback.print_exc()
                continue
    print("Embedding extraction completed.")