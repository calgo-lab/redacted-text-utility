from flair.data import Sentence


class SentenceWithBoundary:
    """
    A custom data structure that represents a Flair Sentence with 
    its boundary positions in the original text.
    """
    
    def __init__(self, sentence: Sentence, start: int, end: int):
        """
        Initialize a SentenceWithBoundary.

        :param sentence: The Flair Sentence object containing tokens and predictions
        :param start: The starting character position of the sentence in the original text
        :param end: The ending character position of the sentence in the original text
        """
        self._sentence = sentence
        self._start = start
        self._end = end
    
    @property
    def sentence(self) -> Sentence:
        """
        Get the Flair Sentence object.

        :return: The Sentence object
        """
        return self._sentence
    
    @property
    def start(self) -> int:
        """
        Get the starting position.

        :return: The starting character position
        """
        return self._start
    
    @start.setter
    def start(self, value: int):
        """
        Set the starting position.

        :param value: The new starting character position
        """
        self._start = value
    
    @property
    def end(self) -> int:
        """
        Get the ending position.

        :return: The ending character position
        """
        return self._end
    
    @end.setter
    def end(self, value: int):
        """
        Set the ending position.

        :param value: The new ending character position
        """
        self._end = value

    def get_length(self) -> int:
        """
        Get the length of the sentence in characters.

        :return: The number of characters in the sentence
        """
        return self._end - self._start