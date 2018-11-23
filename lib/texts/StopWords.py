import codecs


class StopWords:

    def __init__(self, pathToFileWithStopWords: str) -> None:
        self._stopWords = []
        self.__pathToFileWithStopWords = pathToFileWithStopWords

    def getStopWords(self) -> list:
        return self._stopWords

    def loadStopWordsFromFile(self) -> None:
        file = codecs.open(self.__pathToFileWithStopWords, 'r', 'utf-8')
        stopWords = [line.strip() for line in file]
        file.close()
        self._stopWords = stopWords
