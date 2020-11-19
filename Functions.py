# -*- coding: utf-8 -*-
"""
Created on Apr 27, 2016
Updated on Nov 05, 2020

@author: Dimitar Atanasov
"""

# These are some custom functions that we will use in the analysis

import numpy as np
import pandas as pd
import re


def get_file_length(file_path):
    """Takes a file path as input and returns the length of the file in characters"""

    file_length = 0
    with open(file_path, 'rb') as f:
        for line in f:
            lineB = line.decode(errors='replace')
            lineC = lineB.strip('\n')
            file_length += len(lineC)

    return file_length


def get_corpus_file_lengths(corpus_index):
    """Returns the lengths (in characters) of each file in a corpus index"""

    (num_entries, _) = corpus_index.shape
    lengths = np.zeros(num_entries, dtype='int64')
    file_series = pd.Series(corpus_index['Full Relative Path'])
    for entry in range(num_entries):
        lengths[entry] = get_file_length(file_series[entry])

    return pd.Series(lengths, index=corpus_index.index)


def make_pattern_list(regex_series):
    """Takes a list of string patterns and returns them in regex format"""

    regex_list = []
    for item in regex_series.index:
        regex_item = re.compile(regex_series[item])
        regex_list.append(regex_item)

    return regex_list


def count_patterns_in_file(pattern_list, file_path):
    """Given a pattern list and a text file, counts occurrences of each pattern in the file"""

    match_count = np.zeros(len(pattern_list), dtype='int64')
    with open(file_path, 'rb') as f:
        for line in f:
            lineB = line.decode(errors='replace')
            for index in range(len(pattern_list)):
                hit_list = pattern_list[index].findall(lineB, re.I)
                match_count[index] += len(hit_list)

    return match_count


def match_patterns_with_files(pattern_list, file_series):
    """Given a pattern list and a files list, counts occurrences of each pattern in each file"""

    match_counts = np.zeros((len(file_series), len(pattern_list)), dtype='int64')
    idxs = list(file_series.index)
    for file_idx in idxs:
        file_path = file_series[file_idx]
        matches = count_patterns_in_file(pattern_list, file_path)
        match_counts[idxs.index(file_idx), :] = matches

    return match_counts


def df_pattern_matches(corpus_index_df, count_match_array, lexicon_df):
    """"""

    p = list(lexicon_df['Label'])
    df = pd.DataFrame(count_match_array, index=corpus_index_df.index, columns=p)
    return df


def list_conventional_words(text):
    """Finds all conventional words in a string, i.e. not containing numbers or special characters"""

    text_lower = text.lower()
    conventional_words_pattern = re.compile(r'\b[a-zA-Z\'\-&*]{0,}\b')
    return conventional_words_pattern.findall(text_lower)


def make_conventional_bow(text):
    """Makes a bag of words of all conventional words in a file"""

    text_words = list_conventional_words(text)
    bow = {}
    for word in text_words:
        bow[word] = bow.get(word, 0) + 1
    return bow
