import MeCab
from gensim.models import word2vec
import numpy as np

mt = MeCab.Tagger('')
mt.parse('')
model = word2vec.Word2Vec.load('./wiki.model')


def get_vector(text):
    sum_vec = np.zeros(200)
    word_count = 0
    node = mt.parseToNode(text)
    while node:
        fields = node.feature.split(',')
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            sum_vec += model.wv[fields[6]]
            word_count += 1
        node = node.next

    return sum_vec / word_count


def cos_sim(v1, v2):
    print(np.dot(v1, v2))
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == '__main__':
    v1 = get_vector(input())
    v2 = get_vector(input())

    print(cos_sim(v1, v2))
