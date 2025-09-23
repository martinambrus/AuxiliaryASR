import os
import pandas as pd

DEFAULT_DICT_PATH = os.path.join('word_index_dict.txt')

class TextCleaner:
    def __init__(self, word_index_dict_path=DEFAULT_DICT_PATH):
        """Create a TextCleaner.

        Parameters
        ----------
        word_index_dict_path : str or dict
            Either a path to a CSV file mapping phonemes to indices or a
            dictionary object already containing the mapping.
        """
        self.word_index_dictionary = self.load_dictionary(word_index_dict_path)
        # mapping from index back to symbol
        self.inverse_mapping = {index: word for word, index in self.word_index_dictionary.items()}
        #print(word_index_dict_path, len(self.word_index_dictionary))

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(f"(TextCleaner) Warning: Phoneme '{char}' not found in dictionary. Text: " + "".join(text))
        return indexes

    def load_dictionary(self, path_or_dict):
        """Load phoneme to index mapping from a path or return the given dict."""
        if isinstance(path_or_dict, dict):
            return path_or_dict

        csv = pd.read_csv(path_or_dict, header=None).values
        word_index_dict = {word: index for word, index in csv}
        return word_index_dict