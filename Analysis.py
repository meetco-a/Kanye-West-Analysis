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
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


###################################
# 1. DATA CLEANING #
###################################

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

# Add a new Year column and replace newlines with blank space in Lyrics; also drop tags, e.g. [Outro]
dfLyrics["Year"] = pd.DatetimeIndex(dfLyrics["Date"]).year
dfLyrics["Lyrics"] = dfLyrics["Lyrics"].str.replace("\n", " ")
dfLyrics["Lyrics"] = dfLyrics['Lyrics'].str.replace(r"\[[^\]]*\]", "")

# Drop songs where lyrics are not released or from leaked demo, as well as very short lyrics
dfLyrics = dfLyrics[~dfLyrics['Lyrics'].str.contains("Lyrics for this|Lyrics from")]
dfLyrics = dfLyrics[~dfLyrics['Lyrics'].apply(lambda x: len(x) < 10)]


###################################
# 2. CALCULATING FEATURES #
###################################

# I've defined two lexicons: I-words (words referring to oneself); and greatness words
# Here I load each lexicon and then count the occurrence of its words in each song
lexiconI = pd.read_table(os.path.join('Lexicons', 'i_words.txt'), index_col=0, sep='\t')
lexiconGreat = pd.read_table(os.path.join('Lexicons', 'greatness_words.txt'), index_col=0, sep='\t')

# Extract the list of regex patterns in each lexicon and compile them
patternListI = [re.compile(pattern, re.IGNORECASE) for pattern in lexiconI['Regex']]
patternListGreat = [re.compile(pattern, re.IGNORECASE) for pattern in lexiconGreat['Regex']]

# Count the number of pattern matches in each file for both lexicons and put them in a DataFrame
lyricsSeries = dfLyrics["Lyrics"]
matchCountI = count_patterns_series(patternListI, lyricsSeries)
matchCountGreat = count_patterns_series(patternListGreat, lyricsSeries)
countIDF = df_pattern_matches(dfLyrics, matchCountI, lexiconI)
countGreatDF = df_pattern_matches(dfLyrics, matchCountGreat, lexiconGreat)

# Finally, count total lexicon words per song and add new columns to the corpus
totalIWords = countIDF.sum(axis=1)
totalGreatWords = countGreatDF.sum(axis=1)
dfLyrics["I-words"] = totalIWords
dfLyrics["Greatness words"] = totalGreatWords

# The next measure we will use is the vocabulary size, i.e. number of unique words per song
# We'll count unique words per year to avoid duplicate counting of words
stops = stopwords.words('english')
totalWordsYear = pd.Series(np.zeros(len(dfLyrics["Year"].unique())))
uniqueWords = pd.Series(np.zeros(len(dfLyrics["Year"].unique())))
lexicalDiversity = pd.Series(np.zeros(len(dfLyrics["Year"].unique())))

for i, year in enumerate(dfLyrics["Year"].unique()):
    dfYear = dfLyrics[dfLyrics["Year"] == year]

    # Put all the lyrics for that year in one string, then use it to make a BoW
    lyricsYear = ''
    for lyrics in dfYear["Lyrics"]:
        lyricsYear += ' ' + lyrics

    # Make a bag of words from the lyrics and count number of unique words, removing stopwords
    aBOW = make_conventional_bow(lyricsYear)
    totalWordsYear[i] = sum(aBOW.values())
    for k, v in list(aBOW.items()):
        if k in stops or k == "":
            del aBOW[k]
    uniqueWords[i] = len(aBOW)
    lexicalDiversity[i] = uniqueWords[i]/totalWordsYear[i]
    # We'll attach this to the corpus later when we group by year


# Another measure we will use is lexical density, i.e. the ratio of non-stopwords to total words in a song
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
sentiment = dfLyrics.apply(lambda row: get_lyrics_sentiment(row["Lyrics"]), axis=1)
dfLyrics["Sentiment"] = sentiment

# Finally, we get average I-words/Greatness words/Vocabulary size per year
dfLyricsYear = dfLyrics[["Year", "I-words", "Greatness words", "Lexical density", "Sentiment"]]\
    .groupby(["Year"]).mean()
dfLyricsYear.reset_index(inplace=True)

# Since vocabulary size is grouped by year, we attach it to the DF here
dfLyricsYear["Vocabulary size"] = uniqueWords
dfLyricsYear["Lexical diversity"] = lexicalDiversity
dfLyricsYear["Total words"] = totalWordsYear

# Add number of songs per year as well and reset index
dfNoSongs = dfLyrics[["Year", "Date"]].groupby(["Year"]).count().reset_index()
dfLyricsYear["Number of Songs"] = dfNoSongs["Date"]

# Since there are a few years with no data (i.e. no songs released), we need to impute the missing data
# First I add each missing year to the DF with NaN values
for year in range(2003, 2021):
    if year not in dfLyricsYear['Year'].values:
        dfTemp = pd.DataFrame([[year, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 0]],
                              columns=list(dfLyricsYear.columns))
        dfLyricsYear = dfLyricsYear.append(dfTemp, ignore_index=True)

# Then we sort and re-index the data, and finally fill in the missing values using linear interpolation
dfLyricsYear.sort_values(by=['Year'], inplace=True)
dfLyricsYear.reset_index(drop=True, inplace=True)
dfLyricsYear.interpolate(method='linear', inplace=True)


###################################
# 3. PLOTTING #
###################################

# First we plot I-words and Greatness words
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average I-words', color=color)
ax1.plot(dfLyricsYear["Year"], dfLyricsYear["I-words"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Greatness words', color=color)
ax2.plot(dfLyricsYear["Year"], dfLyricsYear["Greatness words"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
print(dfLyricsYear[["I-words", "Greatness words"]].corr())

# Then plot vocab. size and lexical density
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Vocabulary size', color=color)
ax1.plot(dfLyricsYear["Year"], dfLyricsYear["Total words"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Lexical density', color=color)
ax2.plot(dfLyricsYear["Year"], dfLyricsYear["Lexical diversity"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(dfLyricsYear[["Total words", "Vocabulary size", "Lexical diversity", "Lexical density"]].corr())

####################################################################

# Same analysis but using a 3-year rolling average for each variable

dfLyricsRolling = dfLyricsYear.rolling(3).mean()

# Now we plot the data
# First we plot I-words and Greatness words
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average I-words', color=color)
ax1.plot(dfLyricsRolling["Year"], dfLyricsRolling["I-words"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Greatness words', color=color)
ax2.plot(dfLyricsRolling["Year"], dfLyricsRolling["Greatness words"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
print(dfLyricsRolling[["I-words", "Greatness words"]].corr())

# Then plot vocab. size and lexical density
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Vocabulary size', color=color)
ax1.plot(dfLyricsRolling["Year"], dfLyricsRolling["Total words"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Lexical density', color=color)
ax2.plot(dfLyricsRolling["Year"], dfLyricsRolling["Lexical diversity"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(12, 8)
plt.show()

# Let's also check the correlation between the two
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(dfLyricsRolling[["Total words", "Vocabulary size", "Lexical diversity", "Lexical density"]].corr())

plt.plot(dfLyricsRolling["Year"], dfLyricsRolling["Sentiment"])


# Let's also create a word cloud of all of Kanye's lyrics
# Put all lyrics in a string
allLyrics = ""
for lyric in dfLyrics['Lyrics']:
    allLyrics += lyric + " "

# Create a mask to display the cloud in the shape of a person
mask = np.array(Image.open('user.png'))

# Create a word cloud and display it
wordCloud = WordCloud(width=3000, height=2000, random_state=3, background_color='white', colormap='Set2',
                      collocations=False, stopwords=STOPWORDS, mask=mask).generate(allLyrics)

plot_cloud(wordCloud)
