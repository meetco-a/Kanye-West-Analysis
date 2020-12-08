# -*- coding: utf-8 -*-
"""
Created in November 2020

@author: Dimitar Atanasov
"""

import numpy as np
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer


def get_file_length(file_path):
    """
    Takes a file path as input and returns the length of the file in characters.

    :param file_path: str
    :return: file_length: int
    """
    file_length = 0
    with open(file_path, 'rb') as f:
        for line in f:
            line_b = line.decode(errors='replace')
            line_c = line_b.strip('\n')
            file_length += len(line_c)

    return file_length


def get_corpus_file_lengths(corpus_index):
    """
    Returns the lengths (in characters) of each file in a corpus index.

    :param corpus_index: pd.DataFrame
    :return: pd.Series
    """
    (num_entries, _) = corpus_index.shape
    lengths = np.zeros(num_entries, dtype='int64')
    file_series = pd.Series(corpus_index['Full Relative Path'])
    for entry in range(num_entries):
        lengths[entry] = get_file_length(file_series[entry])

    return pd.Series(lengths, index=corpus_index.index)


def list_patterns(regex_series):
    """
    Takes a list of string patterns and returns them in regex format.

    :param regex_series: pd.Series
    :return: regex_list: list
    """
    regex_list = []
    for item in regex_series.index:
        regex_item = re.compile(regex_series[item])
        regex_list.append(regex_item)

    return regex_list


def count_patterns_in_file(pattern_list, file_path):
    """
    Given a pattern list and a text file, counts occurrences of each pattern in the file.

    :param pattern_list: list
    :param file_path: str
    :return: match_count: pd.Series
    """
    match_count = np.zeros(len(pattern_list), dtype='int64')
    with open(file_path, 'rb') as f:
        for line in f:
            line_b = line.decode(errors='replace')
            for index in range(len(pattern_list)):
                hit_list = pattern_list[index].findall(line_b, re.I)
                match_count[index] += len(hit_list)

    return match_count


def match_patterns_with_files(pattern_list, file_series):
    """
    Given a pattern list and a files list, counts occurrences of each pattern in each file.

    :param pattern_list: list
    :param file_series: pd.Series
    :return: match_counts: pd.Series
    """
    match_counts = np.zeros((len(file_series), len(pattern_list)), dtype='int64')
    indexes = list(file_series.index)
    for file_idx in indexes:
        file_path = file_series[file_idx]
        matches = count_patterns_in_file(pattern_list, file_path)
        match_counts[indexes.index(file_idx), :] = matches

    return match_counts


def count_patterns_string(pattern_list, string):
    """
    Given a list of regex patterns and a string, counts the occurrence of each pattern in the string.

    :param pattern_list:
    :param string:
    :return:
    """
    match_count = np.zeros(len(pattern_list), dtype='int64')
    for i in range(len(pattern_list)):
        hit_list = pattern_list[i].findall(string, re.I)
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
