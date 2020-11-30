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

# Initialize empty lists that we'll use later
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

"""
# Export the corpus index to a csv file
if not os.path.exists('Outputs'):
    os.makedirs('Outputs')
dfCorpus.to_csv(os.path.join('Outputs', 'Kanye Corpus Index.csv'), encoding='utf-8')
"""

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
dfCorpusYear.reset_index(inplace=True)

# Since there are a few years with no data (i.e. no songs released), we need to impute the missing data
# First I add each missing year to the DF with NaN values
for year in range(2004, 2021):
    if str(year) not in dfCorpusYear['Year'].values:
        dfTemp = pd.DataFrame([[str(year), np.NaN, np.NaN, np.NaN, np.NaN]], columns=list(dfCorpusYear.columns))
        dfCorpusYear = dfCorpusYear.append(dfTemp, ignore_index=True)

# Then we sort and re-index the data, and finally fill in the missing values using linear interpolation
dfCorpusYear.sort_values(by=['Year'], inplace=True)
dfCorpusYear.reset_index(drop=True, inplace=True)
dfCorpusYear.interpolate(method='linear', inplace=True)

# Now we plot the data
# First we plot I-words and Grandeur words
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('I-words', color=color)
ax1.plot(dfCorpusYear["Year"], dfCorpusYear["I-words"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Grandeur words', color=color)
ax2.plot(dfCorpusYear["Year"], dfCorpusYear["Grandeur words"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

# Then plot vocab. size and lexical density
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Vocabulary size', color=color)
ax1.plot(dfCorpusYear["Year"], dfCorpusYear["Vocabulary size"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Lexical density', color=color)
ax2.plot(dfCorpusYear["Year"], dfCorpusYear["Lexical density"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.yscale("log")
plt.show()

# Let's also check the correlations between the 4 measures
print(dfCorpusYear.corr())
