from .Preprocessing import Preprocessing


class Document:

    def __init__(self, text: str, tag: str = '') -> None:
        self._text = text.strip()
        self._text_as_list = []
        self._tag = tag.strip()

    def getText(self) -> str:
        return self._text

    def getTag(self) -> str:
        return self._tag

    def setText(self, text: str) -> None:
        self._text = text.strip()
        self._text_as_list = []

    def setTag(self, tag: str = '') -> None:
        self._tag = tag.strip()

    def convertTextToListOfWords(self) -> None:
        self._text_as_list = Preprocessing.convertTextToListOfWords(self._text)

    def getTextAsListOfWords(self) -> list:
        return self._text_as_list
