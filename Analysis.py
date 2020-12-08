# -*- coding: utf-8 -*-
"""
Created in December 2020

@author: Dimitar Atanasov
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Functions import *
import pickle
from textblob import TextBlob
from nltk.corpus import stopwords

# Load the scraped lyrics from the pickled file
infile = open("lyrics.txt", "rb")
dfLyrics = pickle.load(infile)
infile.close()

# Drop missing lyrics and missing years
dfLyrics = dfLyrics.dropna(subset=["Lyrics", "Date"])

# Drop duplicates and everything which is not a song (I've compiled this list after manual inspection)
searchFor = ["Freestyle", "Speech", "Reference", "Version", "Alternate", "Jools", "Sunday Service", "Demo", "Mix",
             "Sessions", "Mos Def", "Paparazzi", "monologue", "Taylor Swift", "Single Art", "^On ", "Lecture",
             "\[*\]", "Notepad", "Making of", "Still Standing", "OG", "Solo", "SNL", "2 Ryde"]
dfLyrics = dfLyrics[~dfLyrics["Song Title"].str.contains('|'.join(searchFor))]
dfLyrics.reset_index(inplace=True, drop=True)

# Add a new Year column and replace newlines with blank space in Lyrics
dfLyrics["Year"] = pd.DatetimeIndex(dfLyrics["Date"]).year
dfLyrics["Lyrics"] = dfLyrics["Lyrics"].str.replace("\n", " ")

# I've defined two lexicons: I-words (words referring to oneself); and grandeur words
# Here I load each lexicon and then count the occurrence of its words in each song
lexiconI = pd.read_table(os.path.join('Lexicons', 'i_words.txt'), index_col=0, sep='\t')
lexiconGrand = pd.read_table(os.path.join('Lexicons', 'grandeur_words.txt'), index_col=0, sep='\t')

# Extract the list of regex patterns in each lexicon
patternListI = list_patterns(lexiconI['Regex'])
patternListGrand = list_patterns(lexiconGrand['Regex'])

# Count the number of pattern matches in each file for both lexicons and put them in a DataFrame
lyricsSeries = dfLyrics["Lyrics"]
matchCountI = count_patterns_series(patternListI, lyricsSeries)
matchCountGrand = count_patterns_series(patternListGrand, lyricsSeries)
countIDF = df_pattern_matches(dfLyrics, matchCountI, lexiconI)
countGrandDF = df_pattern_matches(dfLyrics, matchCountGrand, lexiconGrand)

# Finally, count total lexicon words per song and add new columns to the corpus
totalIWords = countIDF.sum(axis=1)
totalGrandWords = countGrandDF.sum(axis=1)
dfLyrics["I-words"] = totalIWords
dfLyrics["Grandeur words"] = totalGrandWords

# The next measure we will use is the vocabulary size, i.e. number of unique words per song
# We'll count unique words per year to avoid duplicate counting of words
stops = stopwords.words('english')
vocabSize = pd.Series(np.zeros(len(dfLyrics["Year"].unique())))

for i, year in enumerate(dfLyrics["Year"].unique()):
    dfYear = dfLyrics[dfLyrics["Year"] == year]

    # Put all the lyrics for that year in one string, then use it to make a BoW
    lyricsYear = ''
    for lyrics in dfYear["Lyrics"]:
        lyricsYear += ' ' + lyrics

    # Make a bag of words from the lyrics and count number of unique words, removing stopwords
    aBOW = make_conventional_bow(lyricsYear)
    for k, v in list(aBOW.items()):
        if k in stops or k == "":
            del aBOW[k]
    vocabSize[i] = len(aBOW)
    # We'll attach this to the corpus later when we group by year


# The final measure we will use is lexical density, i.e. the ratio of non-stopwords to total words in a song
lexicalDensity = pd.Series(np.zeros(len(lyricsSeries)))
for i, lyrics in enumerate(lyricsSeries):
    aBOW = make_conventional_bow(lyrics)
    totalWords = sum(aBOW.values())
    nonStopWords = 0
    dictKeys = list(aBOW.keys())
    for entry in dictKeys:
        if entry not in stops:
            nonStopWords += aBOW[entry]

    if totalWords == 0:
        lexicalDensity[i] = 0
    else:
        lexicalDensity[i] = (nonStopWords/totalWords)*100

dfLyrics['Lexical density'] = lexicalDensity


# Using TextBlob, I measure each lyrics' sentiment
def get_lyrics_sentiment(song_lyrics):
    analysis = TextBlob(song_lyrics)
    return analysis.sentiment.polarity


sentiment = dfLyrics.apply(lambda row: get_lyrics_sentiment(row["Lyrics"]), axis=1)
dfLyrics["Sentiment"] = sentiment

# Finally, we get average I-words/Grandeur words/Vocabulary size per year
dfLyricsYear = dfLyrics[["Year", "I-words", "Grandeur words", "Lexical density", "Sentiment"]]\
    .groupby(["Year"]).mean()
dfLyricsYear.reset_index(inplace=True)

# Since vocabulary size is grouped by year, we attach it to the DF here
dfLyricsYear["Vocabulary size"] = vocabSize

# Add number of songs per year as well and reset index
dfNoSongs = dfLyrics[["Year", "Date"]].groupby(["Year"]).count().reset_index()
dfLyricsYear["Number of Songs"] = dfNoSongs["Date"]

# Since there are a few years with no data (i.e. no songs released), we need to impute the missing data
# First I add each missing year to the DF with NaN values
for year in range(2003, 2021):
    if year not in dfLyricsYear['Year'].values:
        dfTemp = pd.DataFrame([[year, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 0]], columns=list(dfLyricsYear.columns))
        dfLyricsYear = dfLyricsYear.append(dfTemp, ignore_index=True)

# Then we sort and re-index the data, and finally fill in the missing values using linear interpolation
dfLyricsYear.sort_values(by=['Year'], inplace=True)
dfLyricsYear.reset_index(drop=True, inplace=True)
dfLyricsYear.interpolate(method='linear', inplace=True)

# Now we plot the data
# First we plot I-words and Grandeur words
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average I-words', color=color)
ax1.plot(dfLyricsYear["Year"], dfLyricsYear["I-words"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Grandeur words', color=color)
ax2.plot(dfLyricsYear["Year"], dfLyricsYear["Grandeur words"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
print(dfLyricsYear[["I-words", "Grandeur words"]].corr())

# Then plot vocab. size and lexical density
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Vocabulary size', color=color)
ax1.plot(dfLyricsYear["Year"], dfLyricsYear["Vocabulary size"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Lexical density', color=color)
ax2.plot(dfLyricsYear["Year"], dfLyricsYear["Lexical density"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
print(dfLyricsYear[["Vocabulary size", "Lexical density"]].corr())

####################################################################

# Same analysis but using a 3-year rolling average for each variable

dfLyricsTest = dfLyricsYear.rolling(3).mean()

# Now we plot the data
# First we plot I-words and Grandeur words
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average I-words', color=color)
ax1.plot(dfLyricsTest["Year"], dfLyricsTest["I-words"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Grandeur words', color=color)
ax2.plot(dfLyricsTest["Year"], dfLyricsTest["Grandeur words"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
print(dfLyricsTest[["I-words", "Grandeur words"]].corr())

# Then plot vocab. size and lexical density
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Vocabulary size', color=color)
ax1.plot(dfLyricsTest["Year"], dfLyricsTest["Vocabulary size"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Lexical density', color=color)
ax2.plot(dfLyricsTest["Year"], dfLyricsTest["Lexical density"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
print(dfLyricsTest[["Vocabulary size", "Lexical density"]].corr())
