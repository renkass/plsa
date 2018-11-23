from lib.algorithms.texts.PLSA import PLSA
from lib.texts import StopWords
from lib.texts import Corpus
from os import getcwd
import pandas as pd

sw = StopWords("../data/stopwords.dic")
sw.loadStopWordsFromFile()

data = pd.read_csv("../data/lenta_ru.csv")
documents = data["text"].tolist()
tags = data["tags"].tolist()

corpus = Corpus()
corpus.loadCorpusFromList(documents, tags)

K = 2    # number of topic
maxIteration = 10
threshold = 10.0
topicWordsNum = 5
docTopicDist = 'docTopicDistribution.txt'
docTopicDist = getcwd() + "/../target/" + docTopicDist
topicWordDist = 'topicWordDistribution.txt'
topicWordDist = getcwd() + "/../target/" + topicWordDist
dictionary = 'dictionary.dic'
dictionary = getcwd() + "/../target/" + dictionary
topicWords = 'topics.txt'
topicWords = getcwd() + "/../target/" + topicWords

plsa = PLSA(corpus, sw, K, maxIteration, threshold, topicWordsNum, docTopicDist, topicWordDist, dictionary, topicWords)
plsa.EM_Algo()
