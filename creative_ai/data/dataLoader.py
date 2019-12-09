#!/usr/bin/env python
import string
from creative_ai.utils.print_helpers import ppListJson
import json
import spacy
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
from spacy.lang.en import English
import os
from tqdm import tqdm

nlp = English()
sentencizer = Sentencizer()
sentencizer = nlp.create_pipe('sentencizer')
nlp.add_pipe(sentencizer)
nlp = spacy.load('en_core_web_sm')


def prepData(text):
    """
    Returns a copy of text where each inner list starts with the special symbols
    '^::^' and '^:::^', and ends with the symbol '$:::$'.
    >>> prepData(['hello', 'goodbye'])
    ['^::^', '^:::^', 'hello', 'goodbye', '$:::$]
    """
    textCopy = []

    for line in tqdm(text, total=len(list(text)), desc="Prepping data", ncols=80):
        textCopy.append(['^::^', '^:::^'] + line + ['$:::$'])
    return textCopy


def prepTweetData(text):
    textCopy = []
    for i in text:
        doc = nlp(i)
        for token in doc.sents:
            sentenceList = token.text
            inputList = []
            hello = ''
            for j in range(len(sentenceList)):
                if (sentenceList[j] != ' '):
                    hello += sentenceList[j]
                    if (j == len(sentenceList) - 1):
                        if (hello == '&amp;'):
                            hello = '&'
                        inputList.append(hello)
                else:
                    if (hello == '&amp;'):
                        hello = '&'
                    inputList.append(hello)
                    hello = ' '
            textCopy.append(['^::^', '^:::^'] + inputList + [' '] + ['$:::$'])
    return textCopy

def saveData(data, dirName):

    saveDir = os.path.dirname(os.path.abspath(__file__)) + "/saved/"
    fileToSave = os.path.join(saveDir, dirName) + ".json"

    data = ppListJson(data)

    with open(fileToSave, 'w+') as f:
        f.write(data)

def saveTweetData(data, dirName):

    saveDir = os.path.dirname(os.path.abspath(__file__)) + "/saved/"
    fileToSave = os.path.join(saveDir, dirName) + ".json"

    data = ppListJson(data)

    with open(fileToSave, 'w+') as f:
        f.write(data)


def loadSavedLyrics(dirName):

    saveDir = os.path.dirname(os.path.abspath(__file__)) + "/saved/"
    fileToLoad = os.path.join(saveDir, dirName) + ".json"

    with open(fileToLoad, 'r') as f:
        data = json.load(f)

    return data

def loadSavedTweets(dirName):
    #puts it into a JSON file
    saveDir = os.path.dirname(os.path.abspath(__file__)) + "/saved/"
    fileToLoad = os.path.join(saveDir, dirName) + ".json"

    with open(fileToLoad, 'r') as f:
        data = json.load(f)

    return data


def loadSavedMusic(dirName):

    saveDir = os.path.dirname(os.path.abspath(__file__)) + "/saved/"
    fileToLoad = os.path.join(saveDir, dirName) + ".json"

    with open(fileToLoad, 'r') as f:
        data = json.load(f)

    musicData = []

    for line in tqdm(data, total=len(data), desc="Loading saved music", ncols=80):
        tokenSentence = []
        for token in line:
            tokenSentence.append(tuple(token))
        musicData.append(tokenSentence)

    return musicData


def loadLyrics(dirName):
    """
    Loads the lyrics files from the directory specified by dirName,
    if that directory exists. For each line in each file,
    cleans that line by removing punctuation and extraneous
    whitespaces, and lowercasing all words in the line.
    """
    '''
    try:
        return loadSavedLyrics(dirName)
    except:
        pass
    '''
    lyricsDir = os.path.dirname(os.path.abspath(__file__)) + "/lyrics/"
    artistDir = os.path.join(lyricsDir, dirName) + "/"

    if not os.path.isdir(artistDir):
        print("No artist named", artistDir, "in directory", lyricsDir)
        return None

    lyrics = []

    songs = os.listdir(artistDir)
    for song in tqdm(songs, total=len(songs), desc="Loading lyric files", ncols=80):
        with open(artistDir + song, 'r') as songFile:
            songLines = songFile.readlines()

        # clean each line in each song and add if not empty
        for line in songLines:
            line = line.translate(str.maketrans('','',string.punctuation))
            line = line.lower().strip()
            if line:
                lyrics.append(line.split())


    saveTweetData(lyrics, dirName)

    return lyrics

def loadTweets(dirName):
    """
    Loads the lyrics files from the directory specified by dirName,
    if that directory exists. For each line in each file,
    cleans that line by removing punctuation and extraneous
    whitespaces, and lowercasing all words in the line.
    """
    '''
    try:
        return loadSavedTweets(dirName)
    except:
        pass
    '''
    tweetsDir = os.path.dirname(os.path.abspath(__file__)) + "/elonMusk/"
    elonDir = os.path.join(tweetsDir, dirName) + "/"

    if not os.path.isdir(elonDir):
        print("No artist named", elonDir, "in directory", tweetsDir)
        return None

    lyrics = []

    songs = os.listdir(elonDir)
    for song in tqdm(songs, total=len(songs), desc="Loading tweet files", ncols=280):
        with open(elonDir + song, 'r') as songFile:
            songLines = songFile.readlines()

        # clean each line in each song and add if not empty

        for line in songLines:
            #line = line.translate(str.maketrans('','',string.punctuation))
            #line = line.lower().strip()
            #if line:
            lyrics.append(line)


    saveData(lyrics, dirName)

    return lyrics

def loadMusic(dirName):
    """
    Loads the midi files to the specified dirName directory by
    extracting data out of those midi .txt files and converting that
    data into PySynth tuple format.
    """

    try:
        return loadSavedMusic(dirName)
    except:
        pass

    midiDir = os.path.dirname(os.path.abspath(__file__)) + "/midi/"
    platformDir = os.path.join(midiDir, dirName) + "/"

    if not os.path.isdir(platformDir):
        print("No platform named", platformDir, "in directory", midiDir)
        return None

    midiFiles = os.listdir(platformDir)
    midiFiles = [platformDir + "/" + midiFile for midiFile in midiFiles]

    songs = []

    for midiFile in tqdm(midiFiles, total=len(midiFiles), desc="Loading music files", ncols=80):
        with open(midiFile, "r", errors='replace') as f:
            lines = f.readlines()

        song = []
        for line in lines:
            line = line.split()

            # extract pitch and duration from .txt song data, convert
            # those values to pysynth format, and add the
            # (pitch, duration) tuple to the song list
            if "TR" in line and line[line.index("TR") + 1] == "1" \
                    and "NT" in line:
                noteIndex = line.index("NT")
                pitch = line[noteIndex + 1]
                pitch = formatPitch(pitch)

                duration = line[noteIndex + 2]
                duration = formatDuration(duration)

                pysynthTuple = (pitch, duration)
                song.append(pysynthTuple)

        if song:
            songs.append(song)

    saveData(songs, dirName)

    return songs

def formatPitch(asciiPitch):
    """
    Converts from the ASCII representation of a note's pitch to the
    PySynth representation of a note's pitch, returning the
    converted string.
    """
    pitch = asciiPitch.lower()

    # get octave value relative to 4
    octave = 4
    if "'" in pitch:
        numApostrophes = pitch.count("'")
        octave += numApostrophes
        if octave >= 8:
            octave = 7
        pitch = pitch.replace("'", "")
    elif "-" in pitch:
        numDashes = pitch.count("-")
        octave -= numDashes
        if octave <= 0:
            octave = 1
        pitch = pitch.replace("-", "")

    # i don't think pysynth likes e# or b#
    if pitch == "e#":
        pitch = "f"
    elif pitch == "b#":
        pitch = "c"

    # these shouldn't be a problem - test these
    if pitch.count("#") > 1:
        pitch = pitch[0] + "#"
    elif pitch.count("b") > 2:
        pitch = pitch[0] + "b"

    pitch += str(octave)
    return pitch

def formatDuration(asciiDuration):
    """
    Converts from the ASCII representation of a note's duration to the
    PySynth representation of a note's duration, as described in
    the spec. Returns the integer representing the duration.
    """
    duration = re.split("[+ /]", asciiDuration)

    if len(duration) == 1:
        duration = float(duration[0])
    elif len(duration) == 2:
        nominator = float(duration[0])
        denominator = float(duration[1])
        try:
            duration = nominator / denominator
        except ZeroDivisionError:
            duration = nominator
    elif len(duration) == 3:
        wholeNumber = float(duration[0])
        nominator = float(duration[1])
        denominator = float(duration[2])
        try:
            duration = wholeNumber + nominator / denominator
        except ZeroDivisionError:
            duration = wholeNumber
    else: # should never get here
        duration = 1

    if duration < 0.5:
        duration = 16
    elif duration < .75:
        duration = -8
    elif duration < 1:
        duration = 8
    elif duration < 1.5:
        duration = 4
    elif duration < 2:
        duration = -4
    elif duration < 3:
        duration = 2
    elif duration < 4:
        duration = -2
    else: # default
        duration = 1

    return duration

if __name__ == "__main__":
    lyrics = loadLyrics('the_beatles')
    for line in lyrics:
        print(line)
