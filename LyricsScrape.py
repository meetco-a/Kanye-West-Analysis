# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:03:28 2016

@author: Dimitar Atanasov
"""

import requests
from bs4 import BeautifulSoup
import re
import os

URLs = []
yearsAll = []
yearsDict = {}


#Get list of links to each song
for i in range(7):
    anURL = 'http://www.metrolyrics.com/kanye-west-alpage-'+ str(i+1) +'.html'
    r = requests.get(anURL)
    rtxt = r.text
    soup = BeautifulSoup(rtxt, "html.parser")
    txt = soup.get_text()
    links = soup.find_all('a',{'onmousedown':re.compile(r".*Popular Songs.*")})
    yearsRaw = soup.find_all('td',{'content':re.compile(r'\d{0,}')})
    years = [x.get('content') for x in yearsRaw]
    yearsAll += years
    hRefs = [x.get('href') for x in links]
    urlsTemp = [x for x in hRefs if x is not None and x.lower().startswith('http:')]
    URLs += urlsTemp
 
forbiddenChars = ['?','*','>','<','|', ':','\\','/','\"',]

#Extract lyrics for each song and write them into txt file
for i in range(len(URLs)):
    dd = requests.get(URLs[i])
    ddtext = dd.text
    x = BeautifulSoup(ddtext, "html.parser")
    txt = x.get_text()
    songTitle = x.select('title')[0].get_text()
    songName1 = songTitle[13:-21]
    for char in forbiddenChars:
        if  char in songName1:
            songName = songName1.replace(char, '')
            songName1 = songName
    lyricdatatemp = x.find_all('p', {'class':'verse'})
    lyriclist = [lyricdatatemp[i].get_text() for i in range(len(lyricdatatemp))]
    lyricdata = '\n'.join(lyriclist)
    if not os.path.exists(os.path.join('Lyrics2', str(yearsAll[i]))):
        os.makedirs(os.path.join('Lyrics2', str(yearsAll[i])))
    text_file = open(os.path.join('Lyrics2', str(yearsAll[i]), songName1 + '.txt'), 'w')
    try:    
        text_file.write(lyricdata)
    except:
        pass
    text_file.close()
 
 
