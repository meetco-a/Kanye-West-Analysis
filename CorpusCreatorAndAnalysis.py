# -*- coding: utf-8 -*-
"""
Created on Nov 05, 2020

@author: Dimitar Atanasov
"""

import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

#%%
####################################PART 0#####################################


def addColumnData(df, aSeries, cName):
    df1 = df.copy()    
    df1[cName] = aSeries
    return df1


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
    """Finds all conventional words in a string, then creates a bag of words"""

    text_words = list_conventional_words(aText)
    bow = {}
    for word in text_words:
        bow[word] = bow.get(word, 0) + 1
    return bow


# Define column names
dfColumns = ['Full Relative Path', 'Year', 'File Name', 'Song Name']
dfNumCols = len(dfColumns)

# Initialize empty list that we'll use later
dfNumRows = 0
fullPaths = []
songYears = []
fileNames = []
songNames = []

# Get a list of folders in the "Lyrics" directory
yearFolders = [year for year in os.listdir('Lyrics')]

# Loop through each folder and collect the following data
for folder in yearFolders:
    # Number of files in each folder
    dfNumRows = len([f for f in os.listdir(os.path.join('Lyrics', folder))])
    # Full paths of each file
    fullPaths += [(os.path.join('Lyrics', folder, f)) for f in
                  os.listdir(os.path.join('Lyrics', folder))]
    # The year, file name and song name for each file
    songYears += [folder]*dfNumRows
    fileNames += [f for f in os.listdir(os.path.join('Lyrics', folder))]
    songNames = [song.replace('.txt', '') for song in fileNames]

# Put the data from above into a dictionary
dfDict = {dfColumns[0]: fullPaths, dfColumns[1]: songYears, dfColumns[2]: fileNames, dfColumns[3]: songNames}

# Convert the dictionary to a DataFrame
dfCorpus = pd.DataFrame(dfDict)

# Get file lengths for each song/file and add it to corpus index
fileLengths = get_corpus_file_lengths(dfCorpus)
dfCorpus = addColumnData(dfCorpus, fileLengths, 'File Lengths')

# Export the corpus index to a csv file
if not os.path.exists('Outputs'):
    os.makedirs('Outputs')
dfCorpus.to_csv(os.path.join('Outputs', 'Kanye Corpus Index.csv'), encoding='utf-8')


#Get lexicons from txt file; count occurance of each word in all files; add new columns to corpus index
dfLexicon1 = pd.read_table(os.path.join('Lexicons','iwords.txt'),index_col=0,sep='\t')
dfLexicon2 = pd.read_table(os.path.join('Lexicons','grandeurwords.txt'),index_col=0,sep='\t')
aPatternList1 = makePatternList(dfLexicon1['Regex'])
aPatternList2 = makePatternList(dfLexicon2['Regex'])
aFileSeries = dfCorpus2['Full Relative Path']
countMatchArray1 = secMatchPatternsWithFiles(aPatternList1,aFileSeries,re.compile(r'Lyrics\\\d{4}\\'))
countMatchArray2 = secMatchPatternsWithFiles(aPatternList2,aFileSeries,re.compile(r'Lyrics\\\d{4}\\'))
tempDF1 = secMakeDataFrameOfPatternMatches(dfCorpus2,countMatchArray1,dfLexicon1)
tempDF2 = secMakeDataFrameOfPatternMatches(dfCorpus2,countMatchArray2,dfLexicon2)
iwordssum = tempDF1.sum(axis=1)
grandeurwordssum = tempDF2.sum(axis=1)
dfCorpus3 = addColumnData(dfCorpus2,iwordssum,'I-words')
dfCorpus4 = addColumnData(dfCorpus3,grandeurwordssum,'Grandeur words')   

#Get size of vocabulary
vocabSize = pd.Series(np.zeros(len(aFileSeries)))  
for i in range(len(aFileSeries)):
    aText = ''
    with open(aFileSeries[i],'rb') as f:
        for line in f:
            lineb = line.decode(errors='replace')
            linec = lineb.strip('\n')
            aText += ' ' + linec
    aBOW = makeConventionalBoW(aText)
    vocabSize[i] = len(aBOW)
dfCorpus5 = addColumnData(dfCorpus4,vocabSize,'Size of Vocabulary')  

#Get average I-words/Grandeur Words/Vocabulary size per year and plot

iwordscount = {}
avgiwords = {}

grandwordscount = {}
avggrandwords = {}

vocabsizecount = {}
avgvocabsize = {}

songcount = {}

for i in range(len(dfCorpus5)):
    iwordscount[str(dfCorpus5.ix[i,'Year'])] = iwordscount.get(dfCorpus5.ix[i,'Year'],0) + dfCorpus5.ix[i,'I-words']
    grandwordscount[str(dfCorpus5.ix[i,'Year'])] = grandwordscount.get(dfCorpus5.ix[i,'Year'],0) + dfCorpus5.ix[i,'Grandeur words']
    vocabsizecount[str(dfCorpus5.ix[i,'Year'])] = vocabsizecount.get(dfCorpus5.ix[i,'Year'],0) + dfCorpus5.ix[i,'Size of Vocabulary']
    songcount[str(dfCorpus5.ix[i,'Year'])] = songcount.get(dfCorpus5.ix[i,'Year'],0) + 1
for i in sorted(list(vocabsizecount.keys())):
    avgiwords[str(i)] = iwordscount[str(i)]//songcount[str(i)]
    avggrandwords[str(i)] = grandwordscount[str(i)]/songcount[str(i)]
    avgvocabsize[str(i)] = vocabsizecount[str(i)]//songcount[str(i)]

plt.bar(avgiwords.keys(), avgiwords.values(),width=1)
plt.show()

plt.bar(avggrandwords.keys(), avggrandwords.values(),width=1)
plt.show()

plt.bar(avgvocabsize.keys(), avgvocabsize.values(),width=1)
plt.show()

#Get Lexical Density, add new column and plot
#Before running the code below, import nltk, then type nltk.download() and manually find and download the Stopwords corpus
from nltk.corpus import stopwords
stops = stopwords.words('english')

lexicalDensity = pd.Series(np.zeros(len(aFileSeries)))  
for i in range(len(aFileSeries)):
    aText = ''
    with open(aFileSeries[i],'rb') as f:
        for line in f:
            lineb = line.decode(errors='replace')
            linec = lineb.strip('\n')
            aText += linec
    aBOW = makeConventionalBoW(aText)
    totalWords = sum(aBOW.values())
    nonStopWords = 0
    dictKeys = list(aBOW.keys())
    for entry in dictKeys:
        if entry not in stops:
            nonStopWords += aBOW[entry]
    lexicalDensity[i] = (nonStopWords/totalWords)*100

dfCorpus6 = addColumnData(dfCorpus5,lexicalDensity,'Lexical Density')
dfLexDens = dfCorpus6.ix[:,['Year','Lexical Density']]
lexdenscount = {}
avglexdens = {}

for i in range(len(dfLexDens)):
    lexdenscount[str(dfLexDens.ix[i,'Year'])] = lexdenscount.get(dfLexDens.ix[i,'Year'],0) + dfLexDens.ix[i,'Lexical Density']
for i in range(13):
    avglexdens[str(i+2004)] = lexdenscount[str(i+2004)]/songcount[str(i+2004)]
    
plt.bar(avglexdens.keys(), avglexdens.values(),width=1)
plt.show()
