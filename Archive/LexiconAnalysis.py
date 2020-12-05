# -*- coding: utf-8 -*-
"""
Created on Apr 27, 2016
Updated on ...

@author: Dimitar Atanasov
"""
#%%

import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from Functions import *

dfColumns = ['Full Relative Path', 'Year', 'File Name','Song Name']
dfNumCols = len(dfColumns)

dfNumRows = 0
fullPaths = []
songYears = []
fileNames = []
songNames = []
yearFoldersRaw = [year for year in os.listdir('Lyrics')]
yearFolders = [fol for fol in yearFoldersRaw if fol!='desktop.ini']
for folder in yearFolders:
    dfNumRows += len([f for f in os.listdir(os.path.join('Lyrics',folder)) if f!='desktop.ini'])
    fullPaths += [(os.path.join('Lyrics',folder,f)) for f in os.listdir(os.path.join('Lyrics',folder)) if f!='desktop.ini']
    songYears += [folder for f in os.listdir(os.path.join('Lyrics',folder)) if f!='desktop.ini']    
    fileNames += [f for f in os.listdir(os.path.join('Lyrics',folder)) if f!='desktop.ini']
    songNames = [song.replace('.txt', '') for song in fileNames]

dfDic = {}
#dictionary for index
#dictionary for full relative path
dfDic[dfColumns[0]]=fullPaths
#dictionary for years
dfDic[dfColumns[1]]=songYears
#dictionary for File Name
dfDic[dfColumns[2]]=fileNames
#dictionary for Song Name
dfDic[dfColumns[3]]=songNames

dfCorpus = pd.DataFrame(dfDic)
dfCorpus = dfCorpus[dfColumns]

if not os.path.exists('Outputs'):
    os.makedirs('Outputs')
dfCorpus.to_csv(os.path.join('Outputs','Kanye Corpus Index.csv'),encoding='utf-8')


#Get file lengths for each song and add it to corpus index
fileLengths = getCorpusFileLengths(dfCorpus)
dfCorpus2 = addColumnData(dfCorpus,fileLengths,'File Lengths')
dfCorpus2.to_csv(os.path.join('Outputs','Kanye Corpus Index.csv'),encoding='utf-8')

#Get lexicons from txt file; count occurance of each word in all files; add iwords column to corpus index
dfLexicon1 = pd.read_table('Lexicons/iwords.txt',index_col=0,sep='\t')
dfLexicon2 = pd.read_table('Lexicons/grandeurwords.txt',index_col=0,sep='\t')
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
            aText += linec
    aBOW = makeConventionalBoW(aText)
    vocabSize[i] = len(aBOW)
dfCorpus5 = addColumnData(dfCorpus4,vocabSize,'Size of Vocabulary')  

#Get average I-words/Grandeur Words/Vocabulary size per year and plot
dfIwords = dfCorpus5.ix[:,['Year','I-words']]
dfGrandwords = dfCorpus5.ix[:,['Year','Grandeur words']]
dfVocabsize = dfCorpus5.ix[:,['Year','Size of Vocabulary']]

iwordscount = {}
avgiwords = {}

grandwordscount = {}
avggrandwords = {}

vocabsizecount = {}
avgvocabsize = {}

songcount = {}

for i in range(len(dfIwords)):
    iwordscount[str(dfIwords.ix[i,'Year'])] = iwordscount.get(dfIwords.ix[i,'Year'],0) + dfIwords.ix[i,'I-words']
    grandwordscount[str(dfGrandwords.ix[i,'Year'])] = grandwordscount.get(dfGrandwords.ix[i,'Year'],0) + dfGrandwords.ix[i,'Grandeur words']
    vocabsizecount[str(dfVocabsize.ix[i,'Year'])] = vocabsizecount.get(dfVocabsize.ix[i,'Year'],0) + dfVocabsize.ix[i,'Size of Vocabulary']
    songcount[str(dfIwords.ix[i,'Year'])] = songcount.get(dfIwords.ix[i,'Year'],0) + 1
for i in range(13):
    avgiwords[str(i+2004)] = iwordscount[str(i+2004)]//songcount[str(i+2004)]
    avggrandwords[str(i+2004)] = grandwordscount[str(i+2004)]/songcount[str(i+2004)]
    avgvocabsize[str(i+2004)] = vocabsizecount[str(i+2004)]//songcount[str(i+2004)]

plt.bar(avgiwords.keys(), avgiwords.values(),width=1)
plt.show()

plt.bar(avggrandwords.keys(), avggrandwords.values(),width=1)
plt.show()

plt.bar(avgvocabsize.keys(), avgvocabsize.values(),width=1)
plt.show()

#Lexical Density
#nltk.download()
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

    
