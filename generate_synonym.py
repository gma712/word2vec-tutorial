import MeCab
from gensim.models import word2vec


mt = MeCab.Tagger('')
mt.parse('')
model = word2vec.Word2Vec.load('./wiki.model')


def generate(word):
    synonym = model.wv.most_similar(word, [], 1)[0][0]
    print(synonym)


if __name__ == '__main__':
    generate(input())
