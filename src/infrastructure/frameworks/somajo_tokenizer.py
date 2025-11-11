from somajo import SoMaJo

class SoMaJoTokenizer:
    """
    This class provides access to instances of the SoMaJo tokenizer.
    """

    def __init__(self, 
                 language: str = "de_CMC", 
                 split_camel_case: bool = False, 
                 split_sentences: bool = True):
        """
        Initialize the SoMaJo tokenizer.
        :param language: The language model to use for tokenization.
        :param split_camel_case: Whether to split camel case words.
        :param split_sentences: Whether to split sentences.
        """
        self._tokenizer = SoMaJo(language, 
                                split_camel_case=split_camel_case, 
                                split_sentences=split_sentences)
    
    @property
    def tokenizer(self) -> SoMaJo:
        """
        Get the SoMaJo tokenizer instance.
        :return: The SoMaJo tokenizer instance.
        """
        return self._tokenizer