# -*- coding: utf-8 -*-
"""
Created on Nov 05, 2020

@author: Dimitar Atanasov
"""

import os
import matplotlib.pyplot as plt
from Functions import *


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


# Next we get the vocabulary size, i.e. number of unique words per song
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

dfCorpus["Vocabulary Size"] = vocabSize

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
