import os
import csv
import tqdm
import numpy as np
import networkx as nx
from collections import defaultdict
from gensim.models import Word2Vec, FastText
from karateclub import Graph2Vec
from ..models.GloVe import GloVeLikeWord2Vec

def load_features(input_dir):
    print(f"\nLoad features...")
    features = defaultdict(lambda: defaultdict(list))
    for feature_type in ['dynamic', 'static']:
        feature_dir = os.path.join(input_dir, feature_type)
        for feature_name in os.listdir(feature_dir):
            feature_path = os.path.join(feature_dir, feature_name)
            if os.path.isdir(feature_path):
                for file in os.listdir(feature_path):
                    if file.endswith('.csv'):
                        file_hash = os.path.splitext(file)[0]
                        file_path = os.path.join(feature_path, file)
                        if feature_name in ['file_system', 'registry']:
                            features[feature_name][file_hash] = file_path
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                features[feature_name][file_hash] = [row[0] for row in reader]
    return features

def csv_to_graph(file_path, feature_type):
    G = nx.DiGraph()
    node_index = {}
    current_index = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 4:  # time, operation, status, path
                time, operation, status, path = row

                for node in [operation, path]:
                    if node not in node_index:
                        node_index[node] = current_index
                        node_type = 'operation' if node == operation else 'path'
                        G.add_node(current_index, label=node, type=node_type, feature_type=feature_type)
                        current_index += 1

                G.add_edge(node_index[operation], node_index[path], time=float(time), status=status)

    return G

def csv_to_sentences(csv_file_path):
    """
    CSV 파일에서 Word2Vec, FastText 학습을 위한 문장(단어 리스트) 생성
    Time 정보는 제외하고 API, Status, Path만 사용
    """
    sentences = []
    for file_hash, file_path in csv_file_path.items():
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                word_list = []
                for row in csv_reader:
                    # Time 정보를 제외하고 API, Status, Path만 사용합니다
                    api, status, path = row[1], row[2], row[3]
                    # 각 행을 단일 단어로 구성된 "문장"으로 추가합니다
                    word_list.extend([api, status, path])
                if word_list:
                    sentences.append(word_list)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    return sentences

def train_word2vec(sentences, feature_name, save_dir):
    model = Word2Vec(
        vector_size=128, window=5, min_count=3, workers=4, sg=1, hs=0, negative=10,
        ns_exponent=0.75, cbow_mean=0, alpha=0.025, sample=1e-4, shrink_windows=True
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=10, compute_loss=True)
    save_model = os.path.join(save_dir, f"word2vec_{feature_name}.model")
    model.save(save_model)

def train_glove_like(sentences, feature_name, save_dir):
    model = GloVeLikeWord2Vec(
        sentences, vector_size=128, window=5, min_count=3, workers=4, sg=0, hs=0, negative=10,
        ns_exponent=0.75, alpha=0.025, min_alpha=0.0001, sample=1e-4
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    save_model = os.path.join(save_dir, f"glove_like_{feature_name}.model")
    model.save(save_model)

def train_fasttext(sentences, feature_name, save_dir):
    model = FastText(
        vector_size=128, window=5, min_count=3, workers=4, sg=1, hs=0, negative=10,
        ns_exponent=0.75, alpha=0.025, min_alpha=0.0001, sample=1e-4, min_n=2,max_n=5, word_ngrams=1, bucket=2000000
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=10)
    save_model = os.path.join(save_dir, f"fasttext_{feature_name}.model")
    model.save(save_model)

def train_graph2vec(graphs, feature_name, save_dir):
    processed_graphs = []
    file_hashes = []
    error_log_list = []
    success_cnt, fail_cnt = 0, 0
    for file_hash, file_path in graphs.items():
        try:
            G = csv_to_graph(file_path, feature_name)
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                processed_graphs.append(G)
                file_hashes.append(file_hash)
                success_cnt += 1
                # print(f"Processed graph for {file_hash}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            else:
                fail_cnt += 1
                # print(f"Skipping empty graph for {file_hash}")

        except Exception as e:
            error_log_list.append(str(e))
            # print(f"Error processing graph {file_hash}: {str(e)}")
            continue
    print(f"Processing success : {success_cnt}    ||    fail : {fail_cnt}")
    print(f"ERROR LOG LIST : {set(error_log_list)}")

    if not processed_graphs:
        print(f"No valid graphs for {feature_name}. Skipping Graph2Vec training.")
        return

    model = Graph2Vec(
        dimensions=128, workers=4, epochs=10, min_count=3, wl_iterations=3, learning_rate=0.025, down_sampling=1e-4
    )
    model.fit(processed_graphs)
    embeddings = model.get_embedding()

    # Map file hashes to embeddings
    hash_to_embedding = {file_hash: embeddings[i] for i, file_hash in enumerate(file_hashes)}
    save_path = os.path.join(save_dir, f"graph2vec_{feature_name}.npy")
    with open(save_path, 'wb') as f:
        np.save(f, hash_to_embedding)
    print(f"Graph2Vec embeddings for {feature_name} saved successfully.")

def main(input_dir, output_dir):
    print("=" * 80)
    print("Extracted features directory : %s" % input_dir)
    print("Embedding model save directory : %s" % output_dir)
    print("=" * 80)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    features = load_features(input_dir)
    for i, (feature_name, feature_data) in enumerate(features.items(), 1):
        print(f"\nTraining models for {feature_name} ({i}/{len(features)})")
        # Registry, file system: Graph2Vec 학습 과정 추가
        if feature_name in ['registry', 'file_system']:
            sentences = csv_to_sentences(feature_data)
            try:
                # FastText
                print(f"Training '{feature_name}' FastText model...")
                train_fasttext(sentences, feature_name, output_dir)
            except Exception as e:
                print(str(e))
                pass

            try:
                # Word2Vec
                print(f"Training '{feature_name}' Word2Vec model...")
                train_word2vec(sentences, feature_name, output_dir)
            except Exception as e:
                print(str(e))
                pass

            try:
                # Graph2Vec
                print(f"Training '{feature_name}' Graph2Vec model...")
                train_graph2vec(feature_data, feature_name, output_dir)
            except Exception as e:
                print(str(e))
                pass

        # API calls, opcode, strings, import table
        else:
            # Convert feature data to sentences (list of lists)
            sentences = [feature_data[file_hash] for file_hash in feature_data]
            try:
                # FastText
                print(f"Training '{feature_name}' FastText model...")
                train_fasttext(sentences, feature_name, output_dir)
            except Exception as e:
                print(str(e))
                pass

            try:
                # Word2Vec
                print(f"Training '{feature_name}' Word2Vec model...")
                train_word2vec(sentences, feature_name, output_dir)
            except Exception as e:
                print(str(e))
                pass

    print("Word embedding models training completed.")