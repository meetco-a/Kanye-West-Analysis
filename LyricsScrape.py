
import pandas as pd
import pickle
import lyricsgenius


# Using the Genius API, I download all of Kanye West's songs, excluding remixes, live versions, and interviews
# Additional clean up will be done later upon inspection of the scraped data
genius = lyricsgenius.Genius("akw_nUqg_bhW3em_eiKq7yBNrb_KNEfhPqDhY1tavRm8fbmB0RSWZjK7o8fKmbl9")
genius.verbose = False
genius.remove_section_headers = True
genius.excluded_terms = ["(Remix)", "(Live)", "Interview"]
artist = genius.search_artist('Kanye West', sort="title")

# I then put them in a DataFrame
dfColumns = ['Song Title', 'Date', 'Lyrics']
dfCorpus = pd.DataFrame(columns=dfColumns)
for song in artist.songs:
    dfTemp = pd.DataFrame([[song.title, song.year, song.lyrics]], columns=dfColumns)
    dfCorpus = dfCorpus.append(dfTemp, ignore_index=True)

# Pickle the scraped lyrics into a file
outfile = open("lyrics.txt", "wb")
pickle.dump(dfCorpus, outfile)
outfile.close()
