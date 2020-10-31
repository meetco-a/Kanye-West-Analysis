# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:48:18 2016

@author: Dimitar Atanasov
"""

import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt

#%%
####################################PART 0#####################################

def addColumnData(df,aSeries,cName):
    df1 = df.copy()    
    df1[cName] = aSeries
    return df1

def getFileLength(filePath):
    fileLength = 0
    with open(filePath,'rb') as f:
        for line in f:
            lineb = line.decode(errors='replace')
            linec = lineb.strip('\n')
            fileLength += len(linec)
    return fileLength

def getCorpusFileLengths(corpusIndex):
    (numentries,_) = corpusIndex.shape
    lengths = np.zeros(numentries,dtype='int64')
    fileSeries = pd.Series(corpusIndex['Full Relative Path'])
    for entry in range(numentries):
        lengths[entry] = getFileLength(fileSeries[entry])
    return pd.Series(lengths,index=corpusIndex.index)

def makePatternList(aRegexSeries):
    toReturn=[]
    for item in aRegexSeries.index:
        toAdd = re.compile(aRegexSeries[item])
        toReturn.append(toAdd)
    return toReturn

def matchCountWithPatternListWith(aPatternList,filePath):
    hitCounts = np.zeros(len(aPatternList),dtype='int64')
    with open(filePath,'rb') as f:
        for line in f:
            lineb = line.decode(errors='replace')
            for index in range(len(aPatternList)):
                hitList = \
                aPatternList[index].findall(lineb,re.I)
                hitCounts[index] += len(hitList)
    return hitCounts

def secMatchPatternsWithFiles(aPatternList,aFileSeries,corpusLocation):
    toReturn = np.zeros((len(aFileSeries),len(aPatternList)),dtype='int64')
    idxs = list(aFileSeries.index)
    for fileidx in idxs:
        filePath = aFileSeries[fileidx]
        hits = matchCountWithPatternListWith(aPatternList,filePath)
        toReturn[idxs.index(fileidx),:] = hits
    return toReturn

def secMakeDataFrameOfPatternMatches(corpusIndexDF,countMatchArray,aLexiconDF):
   p = list(aLexiconDF['Label'])
   df = pd.DataFrame(countMatchArray,index = corpusIndexDF.index, columns = p)
   return df

def makeListOfConventionalWords(aText):
    textlower = aText.lower()
    convwordpat = re.compile(r'\b[a-zA-Z\'\-&*]{0,}\b')
    return convwordpat.findall(textlower)
   
def makeConventionalBoW(aText):
    thetext = makeListOfConventionalWords(aText)
    returnDict = {}
    for word in thetext:
        returnDict[word] = returnDict.get(word,0) + 1
    return returnDict 

#%%
####################################PART 1#####################################

#define Column Names
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


#%%
####################################PART 2#####################################

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
