import re
from collections import Counter

def preprocess(text:str):
    """
    Replace punctuation with tokens so that we can use it in our model
    :param text:
    :return:
    """
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # remove all words with 5 or lower occurrences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts
    """
    word_counts = Counter(words)
    # sort the words from the most to least frequent in occurrences
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionary
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_words)}
    # create vocab_to_int dictionary
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab
