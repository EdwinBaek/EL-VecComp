import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from scipy.sparse import csr_matrix

def build_cooccurrence_matrix(sentences, window_size=5):
    vocab = set(word for sentence in sentences for word in sentence)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    cooccurrence = defaultdict(float)

    for sentence in sentences:
        for i, word in enumerate(sentence):
            left_context = sentence[max(0, i - window_size):i]
            right_context = sentence[i + 1:i + window_size + 1]
            for context_word in left_context + right_context:
                try:
                    distance = abs(i - sentence.index(context_word))
                    if distance == 0:
                        continue  # 같은 단어는 건너뛰기
                    cooccurrence[(word_to_id[word], word_to_id[context_word])] += 1.0 / distance
                except ValueError as e:
                    print(f"Error processing word pair ({word}, {context_word}): {e}")

    rows, cols, data = zip(*((i, j, v) for (i, j), v in cooccurrence.items()))
    return csr_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)))

def custom_weight_function(x):
    return 1 if x > 100 else max((x / 100) ** 0.75, 1e-6)    # Prevent zero weights

# Word2Vec 모델 상속받은 클래스로, 동시 출현 정보를 사용하도록 학습을 수정함
class GloVeLikeWord2Vec(Word2Vec):
    def __init__(self, sentences=None, **kwargs):
        self.cooccurrence_matrix = None
        super().__init__(sentences=sentences, **kwargs)

    def build_vocab(self, corpus_iterable=None, update=False, progress_per=10000, keep_raw_vocab=False, trim_rule=None,
                    **kwargs):
        super().build_vocab(corpus_iterable=corpus_iterable, update=update, progress_per=progress_per,
                            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        if corpus_iterable is not None:
            self.cooccurrence_matrix = build_cooccurrence_matrix(corpus_iterable)

    def _get_cooccurrence_weight(self, word, context):
        try:
            word_idx = self.wv.key_to_index[word]
            context_idx = self.wv.key_to_index[context]
            cooccurrence = self.cooccurrence_matrix[word_idx, context_idx]
            return custom_weight_function(cooccurrence)
        except KeyError as e:
            print(f"KeyError in _get_cooccurrence_weight: {e}")
            return 1.0
        except Exception as e:
            print(f"Unexpected error in _get_cooccurrence_weight: {e}")
            return 1.0

    def train(self, corpus_iterable=None, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0,
              compute_loss=False, callbacks=(), **kwargs):

        if corpus_iterable is None and self.corpus_file is None:
            raise ValueError("Either corpus_iterable or corpus_file must be provided.")

        if corpus_iterable is None:
            corpus_iterable = self.corpus_file

        if self.cooccurrence_matrix is None:
            self.build_vocab(corpus_iterable=corpus_iterable, update=True)
        print(f"Starting total epochs: {epochs}")
        for epoch in range(epochs):
            for sentence in corpus_iterable:
                word_vecs = []
                for word in sentence:
                    if word in self.wv.key_to_index:
                        word_vecs.append(self.wv[word])

                if len(word_vecs) < 2:
                    continue

                context = np.mean(word_vecs, axis=0)
                for word in sentence:
                    if word in self.wv.key_to_index:
                        weight = self._get_cooccurrence_weight(word, sentence[0])  # 첫 번째 단어를 컨텍스트로 사용
                        word_vec = self.wv[word]

                        # Debugging: Check for NaN or infinity
                        if np.any(np.isnan(word_vec)) or np.any(np.isinf(word_vec)):
                            # print(f"NaN or Inf detected in word vector for '{word}'")
                            continue

                        dot_product = word_vec.dot(context)
                        if dot_product <= 0:
                            # print(f"Non-positive dot product for word '{word}': {dot_product}")
                            continue

                        log_weight = np.log(max(weight, 1e-6))  # Prevent log(0)
                        grad = weight * (dot_product - log_weight)

                        # Clip gradient to prevent exploding gradients
                        grad = np.clip(grad, -5, 5)

                        new_vec = word_vec - self.alpha * grad * context
                        norm = np.linalg.norm(new_vec)
                        if norm > 0:
                            new_vec /= norm
                        else:
                            # print(f"Zero norm for word '{word}', skipping update")
                            continue

                        # Debugging: Check for NaN or infinity after update
                        if np.any(np.isnan(new_vec)) or np.any(np.isinf(new_vec)):
                            # print(f"NaN or Inf detected after update for '{word}'")
                            continue

                        self.wv.vectors[self.wv.key_to_index[word]] = new_vec

            self.alpha *= 0.9  # learning rate decay

        return self