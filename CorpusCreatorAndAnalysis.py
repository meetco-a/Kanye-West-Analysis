# -*- coding: utf-8 -*-
"""
Created on Apr 27, 2016
Updated on Nov 05, 2020

@author: Dimitar Atanasov
"""

import os
import matplotlib.pyplot as plt
from Functions import *
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords


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
    # Number of files in each folder and path of each file
    dfNumRows = len([f for f in os.listdir(os.path.join('Lyrics', folder))])
    fullPaths += [(os.path.join('Lyrics', folder, f)) for f in
                  os.listdir(os.path.join('Lyrics', folder))]

    # The year, file name, and song name for each file
    songYears += [folder]*dfNumRows
    fileNames += [f for f in os.listdir(os.path.join('Lyrics', folder))]
    songNames = [song.replace('.txt', '') for song in fileNames]

# Put the data from above into a dictionary and convert it to a DataFrame
dfDict = {dfColumns[0]: fullPaths, dfColumns[1]: songYears, dfColumns[2]: fileNames, dfColumns[3]: songNames}
dfCorpus = pd.DataFrame(dfDict)

# Get file lengths for each song/file and add it to corpus index
fileLengths = get_corpus_file_lengths(dfCorpus)
dfCorpus['File Lengths'] = fileLengths

# Export the corpus index to a csv file
if not os.path.exists('Outputs'):
    os.makedirs('Outputs')
dfCorpus.to_csv(os.path.join('Outputs', 'Kanye Corpus Index.csv'), encoding='utf-8')


# I've defined two lexicons: I-words (words referring to oneself); and grandeur words
# Here I load each lexicon and then count the occurrence of its words in each song
lexiconI = pd.read_table(os.path.join('Lexicons', 'iwords.txt'), index_col=0, sep='\t')
lexiconGrand = pd.read_table(os.path.join('Lexicons', 'grandeurwords.txt'), index_col=0, sep='\t')

# Extract the list of regex patterns in each lexicon
patternListI = make_pattern_list(lexiconI['Regex'])
patternListGrand = make_pattern_list(lexiconGrand['Regex'])

# Count the number of pattern matches in each file for both lexicons and put them in a DataFrame
fileSeries = dfCorpus['Full Relative Path']
matchCountI = match_patterns_with_files(patternListI, fileSeries)
matchCountGrand = match_patterns_with_files(patternListGrand, fileSeries)
countIDF = df_pattern_matches(dfCorpus, matchCountI, lexiconI)
countGrandDF = df_pattern_matches(dfCorpus, matchCountGrand, lexiconGrand)

# Finally, count total lexicon words per song and add new columns to the corpus
totalIWords = countIDF.sum(axis=1)
totalGrandWords = countGrandDF.sum(axis=1)
dfCorpus["I-words"] = totalIWords
dfCorpus["Grandeur words"] = totalGrandWords

# The next measure we will use is the vocabulary size, i.e. number of unique words per song
vocabSize = pd.Series(np.zeros(len(fileSeries)))

# Get the lyrics of each song
for i in range(len(fileSeries)):
    lyrics = ''
    with open(fileSeries[i], 'rb') as f:
        for line in f:
            lineB = line.decode(errors='replace')
            lineC = lineB.strip('\n')
            lyrics += ' ' + lineC

    # Make a bag of words from the lyrics and count number of unique words
    aBOW = make_conventional_bow(lyrics)
    vocabSize[i] = len(aBOW)

dfCorpus["Vocabulary size"] = vocabSize

# The final measure we will use is lexical density, i.e. the ratio of non-stopwords to total words in a song
stops = stopwords.words('english')

lexicalDensity = pd.Series(np.zeros(len(fileSeries)))
for i in range(len(fileSeries)):
    aText = ''
    with open(fileSeries[i], 'rb') as f:
        for line in f:
            lineb = line.decode(errors='replace')
            linec = lineb.strip('\n')
            aText += linec
    aBOW = make_conventional_bow(aText)
    totalWords = sum(aBOW.values())
    nonStopWords = 0
    dictKeys = list(aBOW.keys())
    for entry in dictKeys:
        if entry not in stops:
            nonStopWords += aBOW[entry]
    lexicalDensity[i] = (nonStopWords/totalWords)*100

dfCorpus['Lexical density'] = lexicalDensity

# Finally, we get average I-words/Grandeur words/Vocabulary size per year
dfCorpusYear = dfCorpus[["Year", "I-words", "Grandeur words", "Vocabulary size", "Lexical density"]]\
    .groupby(["Year"]).mean()
dfCorpusYear = dfCorpusYear.reset_index()

# Now we plot the data
plt.bar(dfCorpusYear["Year"], dfCorpusYear["I-words"], width=0.5)
plt.show()

plt.bar(dfCorpusYear["Year"], dfCorpusYear["Grandeur words"], width=0.5)
plt.show()

plt.bar(dfCorpusYear["Year"], dfCorpusYear["Vocabulary size"], width=0.5)
plt.show()

plt.bar(dfCorpusYear["Year"], dfCorpusYear["Vocabulary size"], width=0.5)
plt.show()
