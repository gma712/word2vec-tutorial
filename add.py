import MeCab
from gensim.models import word2vec
import cal_similarity


mt = MeCab.Tagger('')
mt.parse('')
model = word2vec.Word2Vec.load('./wiki.model')


def add(w1, w2):
    v1 = cal_similarity.get_vector(w1)
    v2 = cal_similarity.get_vector(w2)
    result = model.wv.similar_by_vector(v1 + v2)
    print(result)


if __name__ == '__main__':
    w1 = input()
    w2 = input()
    add(w1, w2)
