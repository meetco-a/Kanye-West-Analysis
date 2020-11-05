# -*- coding: utf-8 -*-
"""
Created on Oct 31, 2020

@author: Dimitar Atanasov
"""

import requests
from bs4 import BeautifulSoup
import re
import os

URLs = []
yearsAll = []
yearsDict = {}

# Get list of links to each song

for i in range(1, 8):
    # This specific website has 7 pages of songs, so we open them 1 by 1
    URL = 'http://www.metrolyrics.com/kanye-west-alpage-{}.html'.format(str(i))
    r = requests.get(URL)

    # After we open the page, extract its html
    rtxt = r.text
    soup = BeautifulSoup(rtxt, "html.parser")

    # Get all the release years for each song
    yearsRaw = soup.find_all('td', {'content': re.compile(r'\d{0,}')})
    years = [x.get('content') for x in yearsRaw]
    yearsAll += years

    # Get all 'a' tags in the page containing a URL except for the "Popular Songs" section to avoid duplicates
    # Then add the URLs to a list
    links = soup.find_all('a', {'onmousedown': re.compile(r".*Popular Songs.*")})
    hRefs = [x.get('href') for x in links]
    urlsTemp = [x for x in hRefs if x is not None and x.lower().startswith('http:')]
    URLs += urlsTemp
 
# Extract lyrics for each song and write them into a txt file
forbiddenChars = ['?', '*', '>', '<', '|', ':', '\\', '/', '\"']

for i in range(len(URLs)):
    # Open each URL and get the page contents
    dd = requests.get(URLs[i])
    ddtext = dd.text
    x = BeautifulSoup(ddtext, "html.parser")

    # Find song name and remove forbidden characters from it
    songTitle = x.select('title')[0].get_text()
    songName = songTitle[13:-21]
    for char in forbiddenChars:
        if char in songName:
            songNameTemp = songName.replace(char, '')
            songName = songNameTemp

    # Extract all verses for each song
    lyricDataTemp = x.find_all('p', {'class':'verse'})
    lyricList = [lyricDataTemp[i].get_text() for i in range(len(lyricDataTemp))]
    lyricData = '\n'.join(lyricList)

    # Write each song to a txt file
    if not os.path.exists(os.path.join('Lyrics', str(yearsAll[i]))):
        os.makedirs(os.path.join('Lyrics', str(yearsAll[i])))
    text_file = open(os.path.join('Lyrics', str(yearsAll[i]), songName + '.txt'), 'w')
    try:    
        text_file.write(lyricData)
    except:
        continue
    finally:
        text_file.close()

# Finally, some files required additional clean up
yearsSet = set(yearsAll)

for year in yearsSet:
    path = os.path.join("Lyrics", str(year))
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # We open each file and read it, then delete any line that contains "[sample text]"
    # Most often these are tags such as "[Intro]", "[Chorus]", etc.
    for file in files:
        pathFile = os.path.join(path, file)
        print(pathFile)
        with open(pathFile, "r", encoding="utf-8") as f:
            try:
                lines = f.readlines()
            except:
                continue
        with open(pathFile, "w", encoding="utf-8") as f:
            for line in lines:
                if len(re.findall(r"\[.*\]", line)) == 0:
                    f.write(line)
                else:
                    pass
