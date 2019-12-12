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
#pattern implementation
#from pattern.en import sentiment

TEAM = 'Turing Machine'
LYRICSDIRS = ['the_beatles']
TWEETSDIRS = ['elon']
LINKDIRS = ['elon2']
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
    #import pdb;pdb.set_trace()
    for ldir in tweetDirs:
        tweets = prepTweetData(loadTweets(ldir))
        model.updateTrainedTweetData(tweets)

    return model

def trainLinkModels(linkDirs, test=False):
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

    for ldir in linkDirs:
        links = prepLinkData(loadTweets(ldir))
        model.updateTrainedLinkData(links)

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
    sentence = ["^::^", "^:::^"]
    x = model.getNextToken(sentence)

    while sentenceTooLong(desiredLength,len(sentence) - 2) == False and x != "$:::$":
        sentence.append(x)
        sentence = grammarRules(sentence, desiredLength)
        x = model.getNextToken(sentence)

    """ this adds a ./.../!/? to the end of a the sentence """
    i = random.randint(0, 7)
    if i in range(0, 5):
        sentence[-1] += '.'
    elif i is 5:
        sentence[-1] += '...'
    elif i is 6:
        sentence[-1] += '!'
    else:
        sentence[-1] += '?'

    return sentence[2:]



def grammarRules(tweet, desiredLength):
    """ this capitalizes the first word in a tweet """
    if len(tweet) is 3:
        firstWord = tweet[2]
        firstWord = firstWord.capitalize()
        tweet[2] = firstWord

    """ this removes amp and replaces it with & """
    for i in range(len(tweet)):
        if tweet[i] is '&amp;' or tweet[i] is 'amp' or tweet[i] is '&amp' or tweet[i] is 'amp;':
            tweet[i] = '&'

    return tweet


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

   print('Welcome to the Elon Musk tweet generator! Now loading... Please be patient :]'.format(TEAM))

   #copy path to your text file
   f = open('elonTweets.txt', 'w')
   l = open('elonLinks.txt', 'w')

   for item in tweepy.Cursor(api.user_timeline, id="elonmusk", tweet_mode='extended').items(100):
       if item.full_text[0:2] != "RT":
           for x in item.full_text.split():
               if x.startswith('https') and x[len(x)-1] != "." and x[len(x)-1] != "!":
                   l.write(x)
                   l.write('\n')
           editedString = ' '.join(x for x in item.full_text.split() if not (x.startswith('@') or x.startswith('https')))
           editedString.replace('&amp;','&')
           if (editedString == '&amp;'):
               editedString = '&'
           editedString.translate({ord(i): None for i in '&amp;'})
           #print(editedString)
           f.write(editedString)
           if item.full_text[len(item.full_text)-1] != '.' and item.full_text[len(item.full_text)-1] != '?' and item.full_text[len(item.full_text)-1] != '!':
               f.write('.')
           f.write('\n')
   f.close()
   l.close()

   mainMenu = Menu(PROMPT2)

   tweetsTrained = False

   while True:
       userInput = mainMenu.getChoice()

       if userInput == 1:
           if not tweetsTrained:
               print('Starting tweet generator...')
               tweetModel = trainTweetModels(TWEETSDIRS)
               linkModel = trainLinkModels(LINKDIRS)
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
    for _ in range(2):
        Tweet.append(generateTokenSentence(models, 10))

    tweetPost = " "
    for index in range(len(Tweet)):
        for index2 in range(len(Tweet[index])):
            store = str(Tweet[index][index2])
            if (store == 'amp' or store == '&amp;' or store == 'amp;'):
                store = '&'
            tweetPost += store
            tweetPost += ' '

    l = open('elonLinks.txt')
    links = l.readlines()
    chance = random.randint(0,4)
    if chance == 1:
        x = random.randint(0,len(links)-1)
        tweetPost = tweetPost + " " + links[x]
    l.close()


    #Future sentiment and subjectvity analysis
    '''
    store = sentiment(tweetPost)
    sentiment = 'postive or negative'
    subjectivity = 'opinion or fact'
    if (store[0] >= 0 and store[0] < 0.25 ):
        sentiment = 'highly negative'
    elif (store[0] >= 0.25 and store[0] < 0.4):
        sentiment = 'negative'
    elif (store[0] >= 0.4 and store[0] < 0.6):
        sentiment = 'neutral'
    elif (store[0] >= 0.6 and store[0] < 0.75):
        sentiment = 'postive'
    elif (store[0] >= 0.75 and store[0] <= 1):
        sentiment = 'highly postive'
    tweetPost += '\n' + 'Sentiment: ' + sentiment
    if (store[0] >= 0 and store[0] < 0.25 ):
        subjectivity = 'highly factual'
    elif (store[0] >= 0.25 and store[0] < 0.4):
        subjectivity = 'mostly fact'
    elif (store[0] >= 0.4 and store[0] < 0.6):
        subjectivity = 'not fact or opinion'
    elif (store[0] >= 0.6 and store[0] < 0.75):
        subjectivity = 'mostly opinion'
    elif (store[0] >= 0.75 and store[0] <= 1):
        subjectivity = 'personal opinion'
    tweetPost += '\n' + 'Subjectivity: ' + subjectivity
    '''

    print(tweetPost)

    api.update_status(tweetPost)

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
