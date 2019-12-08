#!/usr/bin/env python
import sys
sys.dont_write_bytecode = True # Suppress .pyc files

import random
import os
import tweepy
import pysynth
from creative_ai.utils.menu import Menu
from creative_ai.data.dataLoader import *
from creative_ai.models.musicInfo import *
from creative_ai.models.languageModel import LanguageModel

TEAM = 'Turing Machine'
LYRICSDIRS = ['the_beatles']
TWEETSDIRS = ['elon']
TESTLYRICSDIRS = ['the_beatles_test']
MUSICDIRS = ['gamecube']
WAVDIR = 'wav/'

def output_models(val, output_fn = None):
    """
    Requires: nothing
    Modifies: nothing
    Effects:  outputs the dictionary val to the given filename. Used
              in Test mode.

    This function has been done for you.
    """
    from pprint import pprint
    if output_fn == None:
        print("No Filename Given")
        return
    with open('TEST_OUTPUT/' + output_fn, 'wt') as out:
        pprint(val, stream=out)

def sentenceTooLong(desiredLength, currentLength):
    """
    Requires: nothing
    Modifies: nothing
    Effects:  returns a bool indicating whether or not this sentence should
              be ended based on its length.

    This function has been done for you.
    """
    STDEV = 1
    val = random.gauss(currentLength, STDEV)
    return val > desiredLength

def printSongLyrics(verseOne, verseTwo, chorus):
    """
    Requires: verseOne, verseTwo, and chorus are lists of lists of strings
    Modifies: nothing
    Effects:  prints the song.

    This function is done for you.
    """
    verses = [verseOne, chorus, verseTwo, chorus]

    print()
    for verse in verses:
        for line in verse:
            print((' '.join(line)).capitalize())
        print()

def trainLyricModels(lyricDirs, test=False):
    """
    Requires: lyricDirs is a list of directories in data/lyrics/
    Modifies: nothing
    Effects:  loads data from the folders in the lyricDirs list,
              using the pre-written DataLoader class, then creates an
              instance of each of the NGramModel child classes and trains
              them using the text loaded from the data loader. The list
              should be in tri-, then bi-, then unigramModel order.
              Returns the list of trained models.

    This function is done for you.
    """
    model = LanguageModel()

    for ldir in lyricDirs:
        lyrics = prepData(loadLyrics(ldir))
        model.updateTrainedData(lyrics)

    return model

def trainTweetModels(tweetDirs, test=False):
    """
    Requires: lyricDirs is a list of directories in data/lyrics/
    Modifies: nothing
    Effects:  loads data from the folders in the lyricDirs list,
              using the pre-written DataLoader class, then creates an
              instance of each of the NGramModel child classes and trains
              them using the text loaded from the data loader. The list
              should be in tri-, then bi-, then unigramModel order.
              Returns the list of trained models.

    This function is done for you.
    """
    model = LanguageModel()

    for ldir in tweetDirs:
        tweets = prepTweetData(loadTweets(ldir))
        model.updateTrainedTweetData(tweets)

    return model

def trainMusicModels(musicDirs):
    """
    Requires: musicDirs is a list of directories in data/midi/
    Modifies: nothing
    Effects:  works exactly as trainLyricsModels, except that
              now the dataLoader calls the DataLoader's loadMusic() function
              and takes a music directory name instead of an artist name.
              Returns a list of trained models in order of tri-, then bi-, then
              unigramModel objects.

    This function is done for you.
    """
    model = LanguageModel()

    for mdir in musicDirs:
        music = prepData(loadMusic(mdir))
        model.updateTrainedData(music)

    return model

def runLyricsGenerator(models):
    """
    Requires: models is a list of a trained nGramModel child class objects
    Modifies: nothing
    Effects:  generates a verse one, a verse two, and a chorus, then
              calls printSongLyrics to print the song out.
    """
    verseOne = []
    verseTwo = []
    chorus = []

    for _ in range(4):
        verseOne.append(generateTokenSentence(models, 7))
        verseTwo.append(generateTokenSentence(models, 7))
        chorus.append(generateTokenSentence(models, 9))

    printSongLyrics(verseOne, verseTwo, chorus)






def runMusicGenerator(models, songName):
    """
    Requires: models is a list of trained models
    Modifies: nothing
    Effects:  uses models to generate a song and write it to the file
              named songName.wav
    """

    verseOne = []
    verseTwo = []
    chorus = []

    for i in range(4):
        verseOne.extend(generateTokenSentence(models, 7))
        verseTwo.extend(generateTokenSentence(models, 7))
        chorus.extend(generateTokenSentence(models, 9))

    song = []
    song.extend(verseOne)
    song.extend(verseTwo)
    song.extend(chorus)
    song.extend(verseOne)
    song.extend(chorus)

    pysynth.make_wav(song, fn=songName)

###############################################################################
# Begin Core >> FOR CORE IMPLEMENTION, DO NOT EDIT OUTSIDE OF THIS SECTION <<
###############################################################################


def generateTokenSentence(model, desiredLength):
    """
    Requires: model is a single trained languageModel object.
              desiredLength is the desired length of the sentence.
    Modifies: nothing
    Effects:  returns a list of strings where each string is a word in the
              generated sentence. The returned list should NOT include
              any of the special starting or ending symbols.

              For more details about generating a sentence using the
              NGramModels, see the spec.
    """
    L = ["^::^", "^:::^"]
    x = model.getNextToken(L)
    while sentenceTooLong(desiredLength,len(L) - 2) == False and x != "$:::$":
        L.append(x)
        x = model.getNextToken(L)

    return L[2:]


###############################################################################
# End Core
###############################################################################

###############################################################################
# Main
###############################################################################

PROMPT = [
    'Generate song lyrics by The Beatles',
    'Generate a song using data from Nintendo Gamecube',
    'Quit the music generator'
]

PROMPT2 = [
    'Generate a tweet',
    'End the program'
]

def main():
    """
    Requires: Nothing
    Modifies: Nothing
    Effects:  This is your main function, which is done for you. It runs the
              entire generator program for both the reach and the core.

              It prompts the user to choose to generate either lyrics or music.
    """

    mainMenu = Menu(PROMPT)

    lyricsTrained = False
    musicTrained = False

    print('Welcome to the {} music generator!'.format(TEAM))
    while True:
        userInput = mainMenu.getChoice()

        if userInput == 1:
            if not lyricsTrained:
                print('Starting lyrics generator...')
                lyricsModel = trainLyricModels(LYRICSDIRS)
                lyricsTrained = True

            runLyricsGenerator(lyricsModel)

        elif userInput == 2:
            if not musicTrained:
                print('Starting music generator...')
                musicModel = trainMusicModels(MUSICDIRS)
                musicTrained = True

            songName = input('What would you like to name your song? ')

            runMusicGenerator(musicModel, WAVDIR + songName + '.wav')

        elif userInput == 3:
            print('Thank you for using the {} music generator!'.format(TEAM))
            sys.exit()


def getTweet():
    consumer_key = "jwRrpCUD5nicMU8bkd31Eh9yV"
    consumer_secret = "ojFtjsecfskKeXso2IM8Jbekj5bZZCPECubfgmaOOaI9mWnNVg"
    access_token = "1203360136691011585-oEHF6waSe3DHWyKuWb4zbdLR2x0K5I"
    access_token_secret = "ShTIm1K0p9aEQKFaa711l97hoymIFUkwtthyV9zsouMVb"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    print('Welcome to the Elon Musk tweet generator!'.format(TEAM))

    #ignore links, retweets and replies
    '''
    f = open('elonTweets.txt', 'w')
    for item in tweepy.Cursor(api.user_timeline, id="elonmusk", tweet_mode='extended').items():
        if item.full_text[0:2] != "RT":
            f.write(item.full_text)
            f.write('\n')
    f.close()
    '''

    mainMenu = Menu(PROMPT2)

    tweetsTrained = False

    while True:
        userInput = mainMenu.getChoice()

        if userInput == 1:
            if not tweetsTrained:
                print('Starting tweet generator...')
                #change two tweetdirs
                tweetModel = trainTweetModels(TWEETSDIRS)
                tweetsTrained = True

            runTweetGenerator(tweetModel)
        elif userInput == 2:
            print('Thank you for using the {} tweet generator!'.format(TEAM))
            sys.exit()

def runTweetGenerator(models):
    """
    Requires: models is a list of a trained nGramModel child class objects
    Modifies: nothing
    Effects:  generates a verse one, a verse two, and a chorus, then
              calls printSongLyrics to print the song out.
    """
    Tweet = []
    consumer_key = "jwRrpCUD5nicMU8bkd31Eh9yV"
    consumer_secret = "ojFtjsecfskKeXso2IM8Jbekj5bZZCPECubfgmaOOaI9mWnNVg"
    access_token = "1203360136691011585-oEHF6waSe3DHWyKuWb4zbdLR2x0K5I"
    access_token_secret = "ShTIm1K0p9aEQKFaa711l97hoymIFUkwtthyV9zsouMVb"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    for _ in range(1):
        Tweet.append(generateTokenSentence(models, 10))
    tweetPost = " "
    for index in range(len(Tweet)):
        for index2 in range(len(Tweet[index])):
            store = str(Tweet[index][index2])
            tweetPost += store
            tweetPost += " "
    print(tweetPost)

    #api.update_status(tweetPost)

# This is how python tells if the file is being run as main
if __name__ == '__main__':
    #main()
    getTweet()

    # note that if you want to individually test functions from this file,
    # you can comment out main() and call those functions here. Just make
    # sure to call main() in your final submission of the project!
    #first set of tests
    # text = [["^::^", "^:::^", 'the', 'quick', 'brown', 'fox', "$:::$"], ["^::^", "^:::^", 'the', 'lazy', 'quick', 'dog', 'jumped', 'over', "$:::$"], ["^::^", "^:::^", 'the', 'quick', 'brown', 'dog', 'barked', "$:::$"],
    #           ["^::^", "^:::^", 'dog', 'jumped', 'over', 'the', 'fox', "$:::$"], ["^::^", "^:::^", 'brown', 'cat', "$:::$"], ["^::^", "^:::^",'the','quick','brown','fox','jumped','over','the','lazy','dog', "$:::$"], ["^::^", "^:::^",'the','brown','dog','fox','quick','brown','the','dog','jumped','the','jumped',"$:::$"]]
    '''
    x = LanguageModel()
    x.updateTrainedData(text)
    print(generateTokenSentence(x,1))
    print(generateTokenSentence(x,2))
    print(generateTokenSentence(x,3))
    print(generateTokenSentence(x,4))
    print(generateTokenSentence(x,5))
    print(generateTokenSentence(x,10))

    #next set of tests
    text2 = [["^::^", "^:::^", 'rocket', 'to', 'the', 'moon', "$:::$"], ["^::^", "^:::^", 'dad', 'is', 'from', 'nyc', "$:::$"], ["^::^", "^:::^", 'the', 'quick', 'brown', 'dog', 'barked', "$:::$"],
           ["^::^", "^:::^", 'hello', 'world', 'I', 'am', 'Macintosh', "$:::$"], ["^::^", "^:::^", 'I', 'am', 'from', 'mars', "$:::$"], ["^::^", "^:::^",'A','friend','went','to','the','moon',"$:::$"], ["^::^", "^:::^",'The','new','Macbook','Pro','has','a','better','keyboard','and','it','sells', "well","$:::$"],
           ["^::^", "^:::^", 'My', 'hopes', 'are', 'high', "$:::$"]]
    y = LanguageModel()
    y.updateTrainedData(text2)
    print(generateTokenSentence(y, 1))
    print(generateTokenSentence(y, 2))
    print(generateTokenSentence(y, 3))
    print(generateTokenSentence(y, 4))
    print(generateTokenSentence(y, 5))
    print(generateTokenSentence(y, 10))

    # next set of tests
    text3 = [["^::^", "^:::^", 'hello', 'my', 'name', 'is', 'nikhil', "$:::$"],
             ["^::^", "^:::^", 'dad', 'is', 'from', 'nyc', "$:::$"],
             ["^::^", "^:::^", 'the', 'quick', 'brown', 'dog', 'barked', "$:::$"],
             ["^::^", "^:::^", 'hello', 'world', 'I', 'am', 'iMac', "$:::$"],
             ["^::^", "^:::^", 'I', 'am', 'from', 'venus' , "$:::$"],
             ["^::^", "^:::^", 'A', 'friend', 'went', 'to', 'the', 'Soviet', 'Union', "$:::$"],
             ["^::^", "^:::^", 'The', 'new', 'iPhone', '11', 'has', 'a', 'better', 'camera', 'and', 'it', 'sells',
              "well", "$:::$"],
             ["^::^", "^:::^", 'My', 'hopes', 'are', 'high', "$:::$"]]
    z = LanguageModel()
    z.updateTrainedData(text3)
    print(generateTokenSentence(z, 1))
    print(generateTokenSentence(z, 2))
    print(generateTokenSentence(z, 3))
    print(generateTokenSentence(z, 4))
    print(generateTokenSentence(z, 5))
    print(generateTokenSentence(z, 7))
    print(generateTokenSentence(z, 30))
    '''''
