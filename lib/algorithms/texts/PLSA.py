import codecs
import re
import time
from numpy import log
from numpy import int8
from numpy import zeros
from pylab import random
from lib.texts import StopWords
from lib.texts import Corpus
from lib.texts import Preprocessing


class PLSA:

    def __init__(self, corpus: Corpus, stopWords: StopWords, K: int, maxIteration: int, threshold: float,
                 topicWordsNum: int, docTopicDist: str, topicWordsDist: str, dictionary: str, topicWords: str) -> None:
        self._corpus = corpus
        self._stopWords = stopWords
        self._K = K  # number of topic
        self._maxIteration = maxIteration
        self._threshold = threshold
        self._topicWordsNum = topicWordsNum
        self._docTopicDist = docTopicDist
        self._topicWordDist = topicWordsDist
        self._dictionary = dictionary
        self._topicWords = topicWords

        self._N = len(corpus.getDocuments())
        self._wordCounts = []
        self._word2id = {}
        self._id2word = {}
        self._currentId = 0

        for document in corpus.getDocuments():
            normalized_list = Preprocessing.convertListOfWordsToNormalForms(Preprocessing
                                                                            .convertDocumentToListOfWords(document))
            self._segList = Preprocessing.removeStopWordsFromListOfWords(self._stopWords.getStopWords(),
                                                                         normalized_list)
            self._wordCount = {}
            for word in self._segList:
                word = word.lower().strip()
                if len(word) > 1 and not re.search('[0-9]', word) and word not in self._stopWords.getStopWords():
                    if word not in self._word2id.keys():
                        self._word2id[word] = self._currentId
                        self._id2word[self._currentId] = word
                        self._currentId += 1
                    if word in self._wordCount:
                        self._wordCount[word] += 1
                    else:
                        self._wordCount[word] = 1
            self._wordCounts.append(self._wordCount)

        # length of dictionary
        self._M = len(self._word2id)

        # generate the document-word matrix
        self._X = zeros([self._N, self._M], int8)
        for word in self._word2id.keys():
            j = self._word2id[word]
            for i in range(0, self._N):
                if word in self._wordCounts[i]:
                    self._X[i, j] = self._wordCounts[i][word]

        # lamda[i, j] : p(zj|di)
        self._lamda = random([self._N, self._K])

        # theta[i, j] : p(wj|zi)
        self._theta = random([self._K, self._M])

        # p[i, j, k] : p(zk|di,wj)
        self._p = zeros([self._N, self._M, self._K])

    def initializeParameters(self) -> None:
        for i in range(0, self._N):
            self._normalization = sum(self._lamda[i, :])
            for j in range(0, self._K):
                self._lamda[i, j] /= self._normalization

        for i in range(0, self._K):
            self._normalization = sum(self._theta[i, :])
            for j in range(0, self._M):
                self._theta[i, j] /= self._normalization

    def EM_Algo(self) -> None:
        # EM algorithm
        oldLoglikelihood = 1
        newLoglikelihood = 1
        for i in range(0, self._maxIteration):
            self._EStep()
            self._MStep()
            newLoglikelihood = self._LogLikelihood()
            print("[", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "] ", i + 1, " iteration  ",
                  str(newLoglikelihood))
            if oldLoglikelihood != 1 and newLoglikelihood - oldLoglikelihood < self._threshold:
                break
            oldLoglikelihood = newLoglikelihood
        self._output()

    def _EStep(self) -> None:
        for i in range(0, self._N):
            for j in range(0, self._M):
                denominator = 0
                for k in range(0, self._K):
                    self._p[i, j, k] = self._theta[k, j] * self._lamda[i, k]
                    denominator += self._p[i, j, k]
                if denominator == 0:
                    for k in range(0, self._K):
                        self._p[i, j, k] = 0
                else:
                    for k in range(0, self._K):
                        self._p[i, j, k] /= denominator

    def _MStep(self) -> None:
        # update theta
        for k in range(0, self._K):
            denominator = 0
            for j in range(0, self._M):
                self._theta[k, j] = 0
                for i in range(0, self._N):
                    self._theta[k, j] += self._X[i, j] * self._p[i, j, k]
                denominator += self._theta[k, j]
            if denominator == 0:
                for j in range(0, self._M):
                    self._theta[k, j] = 1.0 / self._M
            else:
                for j in range(0, self._M):
                    self._theta[k, j] /= denominator

        # update lamda
        for i in range(0, self._N):
            for k in range(0, self._K):
                self._lamda[i, k] = 0
                denominator = 0
                for j in range(0, self._M):
                    self._lamda[i, k] += self._X[i, j] * self._p[i, j, k]
                    denominator += self._X[i, j]
                if denominator == 0:
                    self._lamda[i, k] = 1.0 / self._K
                else:
                    self._lamda[i, k] /= denominator

    # calculate the log likelihood
    def _LogLikelihood(self) -> int:
        loglikelihood = 0
        for i in range(0, self._N):
            for j in range(0, self._M):
                tmp = 0
                for k in range(0, self._K):
                    tmp += self._theta[k, j] * self._lamda[i, k]
                if tmp > 0:
                    loglikelihood += self._X[i, j] * log(tmp)
        return loglikelihood

    # output the params of model and top words of topics to files
    def _output(self) -> None:
        # document-topic distribution
        file = codecs.open(self._docTopicDist, 'w', 'utf-8')
        for i in range(0, self._N):
            tmp = ''
            for j in range(0, self._K):
                tmp += str(self._lamda[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()

        # topic-word distribution
        file = codecs.open(self._topicWordDist, 'w', 'utf-8')
        for i in range(0, self._K):
            tmp = ''
            for j in range(0, self._M):
                tmp += str(self._theta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()

        # dictionary
        file = codecs.open(self._dictionary, 'w', 'utf-8')
        for i in range(0, self._M):
            file.write(self._id2word[i] + '\n')
        file.close()

        # top words of each topic
        file = codecs.open(self._topicWords, 'w', 'utf-8')
        for i in range(0, self._K):
            topicword = []
            ids = self._theta[i, :].argsort()
            for j in ids:
                topicword.insert(0, self._id2word[j])
            tmp = ''
            for word in topicword[0:min(self._topicWordsNum, len(topicword))]:
                tmp += word + ' '
            file.write(tmp + '\n')
        file.close()
