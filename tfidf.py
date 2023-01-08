from sklearn.preprocessing import normalize

from collections import Counter
from scipy.sparse import csr_matrix


class TFIDF:
    def __init__(self):
        self.features = {}
        self.idfs_ = {}

    def IDF(self, corpus, unique_words):
        idf_vals = {}
        total_docs = len(corpus)
        for word in unique_words:
            cnt = 0
            for row in corpus:
                if word in row.split(" "):
                    cnt += 1
            idf_vals[word] = 1 + math.log((1+total_docs)/(1+cnt))
        return idf_vals

    def fit(self, dataset):
        unique_words = set()
        if isinstance(dataset, (list,)):
            for row in dataset:
                for word in row.split(" "):
                    if len(word) < 2:
                        continue
                    unique_words.add(word)
            unique_words = sorted(list(unique_words))
            self.features = {j: i for i, j in enumerate(unique_words)}
        self.idfs_ = self.IDF(dataset, unique_words)
        return self.features, self.idfs_

    def transform(self, dataset):
        sparse_matrix = csr_matrix(
            (len(dataset), len(self.features)), dtype=float)
        for row in range(0, len(dataset)):
            word_count = Counter(dataset[row].split(' '))
            for word in dataset[row].split(' '):
                if word in list(self.features.keys()):
                    tf = word_count[word] / len(dataset[row].split(' '))
                    tfidf = tf * self.idfs_[word]
                    sparse_matrix[row, self.features[word]] = tfidf
        return normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
