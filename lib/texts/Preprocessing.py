import copy
import re
import pymorphy2
from lib.texts import Document


class Preprocessing:
    morphAnalyzer = pymorphy2.MorphAnalyzer()

    @staticmethod
    def convertTextToListOfWords(text: str) -> list:
        return re.sub("[^\w]", " ", text).split()

    @staticmethod
    def convertDocumentToListOfWords(document: Document) -> list:
        return re.sub("[^\w]", " ", document.getText()).split()

    @staticmethod
    def convertListOfWordsToNormalForms(list_of_words: list) -> list:
        normalized_list_of_words = [Preprocessing.morphAnalyzer.normal_forms(word)[0] for word in list_of_words]
        return normalized_list_of_words

    @staticmethod
    def removeStopWordsFromListOfWords(stop_words: list, list_of_words: list) -> list:
        copy_set_of_words = copy.copy(list_of_words)
        copy_set_of_stop_words = copy.copy(stop_words)
        answer = copy.copy(copy_set_of_words)
        for word in copy_set_of_words:
            for stop_word in copy_set_of_stop_words:
                if word == stop_word:
                    answer = list(filter(lambda el: el != stop_word, answer))
        return list(answer)
