"""
    Visualization 관련 utils function
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, FastText

# Load embedding model and visualization for each embedding vector range
class EmbeddingVectorVisualizer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_embedding_model(self, model_path):
        if 'fasttext' in model_path.lower():
            return FastText.load(model_path)
        else:
            return Word2Vec.load(model_path)

    def analyze_vector_range(self, model, model_name):
        vectors = model.wv.vectors
        min_value = np.min(vectors)
        max_value = np.max(vectors)
        mean_value = np.mean(vectors)
        std_value = np.std(vectors)

        print(f"Minimum value: {min_value}")
        print(f"Maximum value: {max_value}")
        print(f"Mean value: {mean_value}")
        print(f"Standard deviation: {std_value}")

        plt.figure(figsize=(12, 8), dpi=300)  # 고해상도 설정
        plt.hist(vectors.flatten(), bins=50, edgecolor='black')
        plt.title(f"Distribution of Vector Values - {model_name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # 고해상도로 이미지 저장
        output_path = os.path.join(self.output_dir, f"{model_name}_vector_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # 메모리 해제를 위해 figure 닫기
        print(f"Figure saved to {output_path}")

    def analyze_models(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.model'):
                model_path = os.path.join(self.input_dir, filename)
                model_name = os.path.splitext(filename)[0]
                print(f"\nAnalyzing model: {model_name}")
                model = self.load_embedding_model(model_path)
                self.analyze_vector_range(model, model_name)


def main():
    embedding_dir = '../../dataset/models/embedding_models/'
    figure_dir = '../../dataset/figure/vector_range/'

    analyzer = EmbeddingVectorVisualizer(embedding_dir, figure_dir)
    analyzer.analyze_models()


if __name__ == "__main__":
    main()