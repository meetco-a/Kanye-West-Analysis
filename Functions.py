# -*- coding: utf-8 -*-
"""
Created in November 2020

@author: Dimitar Atanasov
"""

import numpy as np
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
import matplotlib.pyplot as plt


def count_patterns_string(pattern_list, string):
    """
    Given a list of regex patterns and a string, counts the occurrence of each pattern in the string.

    :param pattern_list:
    :param string:
    :return:
    """
    match_count = np.zeros(len(pattern_list), dtype='int64')
    for i in range(len(pattern_list)):
        hit_list = pattern_list[i].findall(string)
        match_count[i] += len(hit_list)

    return match_count


def count_patterns_series(pattern_list, series):
    """
    Given a list of regex patterns and a series of strings, counts occurrences of each pattern in each string.

    :param pattern_list: list
    :param series: pd.Series
    :return: match_counts: pd.Series
    """
    match_counts = np.zeros((len(series), len(pattern_list)), dtype='int64')
    indexes = list(series.index)
    for idx in indexes:
        string = series[idx]
        matches = count_patterns_string(pattern_list, string)
        match_counts[indexes.index(idx), :] = matches

    return match_counts


def df_pattern_matches(corpus_index_df, count_match_array, lexicon_df):
    """"""

    p = list(lexicon_df['Label'])
    df = pd.DataFrame(count_match_array, index=corpus_index_df.index, columns=p)
    return df


def list_conventional_words(text):
    """
    Finds all conventional words in a string, i.e. not containing numbers or special characters.

    :param text: str
    :return: list
    """
    conventional_words_pattern = re.compile(r'\b[a-zA-Z\'\-&*]*\b')
    return conventional_words_pattern.findall(text.lower())


def make_conventional_bow(text):
    """
    Makes a bag of words of all stemmed conventional words in a file.

    :param text: str
    :return: bow: dict
    """
    text_words = list_conventional_words(text)
    bow = {}
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    for word in text_words:
        word_stem = stemmer.stem(word)
        bow[word_stem] = bow.get(word_stem, 0) + 1
    return bow


def get_lyrics_sentiment(song_lyrics):
    """
    Given some text, calculate its polarity score using TextBlob.
    Polarity score ranges from -1 (negative) to 1 (positive).

    :param song_lyrics: string
    :return: TextBlob.sentiment.polarity
    """
    analysis = TextBlob(song_lyrics)
    return analysis.sentiment.polarity


def plot_cloud(word_cloud):
    """
    Given a WordCloud object, plots it.

    :param word_cloud: WordCloud
    :return: None
    """
    # Set size
    plt.figure(figsize=(40, 30))
    # Display image with no axis
    plt.imshow(word_cloud)
    plt.axis("off")
