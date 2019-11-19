################################################################################
# The following functions are stubbed for you. You must pass all of our test
# cases for these functions as part of the core. To run your functions with our
# test cases, run this file as main. This file uses python's doctest module,
# which will run your implementations against the test cases contained in each
# function. Do NOT change any of the comments in this file or we will NOT be
# able to grade your project.
#Nikhil
# Turn off bytecode .pyc files
import sys
sys.dont_write_bytecode = True
#
def returnDictionary(D):
    """
    Requires: Nothing
    Modifies: Nothing
    Effects:  Returns the input dictionary D unchanged.
    >>> returnDictionary({})
    {}
    >>> lyrics = {'a': 'hard', 'days': 'night'}
    >>> returnDictionary(lyrics) == {'a': 'hard', 'days': 'night'}
    True
    """
    pass

def keyInDict(D, K):
    """
    Requires: D is a dictionary
    Modifies: Nothing
    Effects:  Returns True if and only if the key K is already in D.
              Hint: use the 'in' operator
    >>> keyInDict({}, 'night')
    False
    >>> lyrics = {'a': 'hard', 'days': 'night'}
    >>> keyInDict(lyrics, 'days')
    True
    >>> keyInDict(lyrics, 'postman')
    False
    """
    pass

def returnKeyVal(D, K):
    """
    Requires: D is a dictionary and K is a key in D
    Modifies: Nothing
    Effects:  Returns the value associated with K in the dictionary D.
    >>> lyrics = {'wait': 'a', 'minute': 'mister', 'postman': {}}
    >>> returnKeyVal(lyrics, 'wait')
    'a'
    >>> returnKeyVal(lyrics, 'postman')
    {}
    """
    pass

def setKeyVal(D, K, V):
    """
    Requires: D is a dictionary
    Modifies: D
    Effects:  Sets the value associated with the key K in the dictionary D
              to be the value V. Returns the dictionary D.
    >>> setKeyVal({}, 'all the', 'lonely people') == {'all the': 'lonely people'}
    True
    >>> setKeyVal({'where do': 'they all'}, 'come', 'from') == {'where do': 'they all', 'come': 'from'}
    True
    """
    pass

def setKeyValList(D, K, V1, V2, V3, V4):
    """
    Requires: D is a dictionary
    Modifies: D
    Effects:  Sets the value associated with the key K, which is a key in
              the input dictionary D, to be a list composed of V1 through
              V4, in that order. Returns the dictionary D.
    >>> setKeyValList({}, 'taxman', 'cause', 'im', 'the', 'taxman') == {'taxman': ['cause', 'im', 'the', 'taxman']}
    True
    """
    pass

def asciiAssociate():
    """
    Requires: Nothing
    Modifies: Nothing
    Effects:  Makes a new dictionary, called asciiDict, whose keys are
              the lowercase characters from a to z, and whose values are
              the associated ascii values from 97 to 122. Returns the
              dictionary asciiDict.
    >>> asciiAssociate() == {'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101, 'f': 102, 'g': 103, 'h': 104, 'i': 105, 'j': 106, 'k': 107, 'l': 108, 'm': 109, 'n': 110, 'o': 111, 'p': 112, 'q': 113, 'r': 114, 's': 115, 't': 116, 'u': 117, 'v': 118, 'w': 119, 'x': 120, 'y': 121, 'z': 122}
    True
    """
    # You may find this useful
    from string import ascii_lowercase as alphabet
    pass

def nestedAscii():
    """
    Requires: Nothing
    Modifies: Nothing
    Effects:  Creates a new dictionary, D, where its keys are the
              lowercase characters from a to z, and each key has a value
              of an empty dictionary. Returns the new dictionary D.
    >>> nestedAscii() == {'a': {}, 'b': {}, 'c': {}, 'd': {}, 'e': {}, 'f': {}, 'g': {}, 'h': {}, 'i': {}, 'j': {}, 'k': {}, 'l': {}, 'm': {}, 'n': {}, 'o': {}, 'p': {}, 'q': {}, 'r': {}, 's': {}, 't': {}, 'u': {}, 'v': {}, 'w': {}, 'x': {}, 'y': {}, 'z': {}}
    True
    """
    # You may find this useful
    from string import ascii_lowercase as alphabet
    pass

def getNote(song, note):
    """
    Requires: song is a dictionary, note is a key in
              song, and the value associated with note is a non-empty
              list
    Modifies: Nothing
    Effects:  Returns the first element in the list associated with the
              key "name" in the input dictionary favoriteColors.
    >>> getNote({'start': ['c4']}, 'start')
    'c4'
    >>> getNote({'backbeat': ['e1', 'g1']}, 'backbeat')
    'e1'
    """
    pass

def translate(vocab, word, language):
    """
    Requires: vocab is a 2-dimensional dictionary, word is a key in vocab,
              and language is a key in the dictionary that word maps to
    Modifies: Nothing
    Effects:  The input dictionary, vocab, could look something like this:
              {"hello": {"Spanish" : "hola", "French": "bonjour"}}
              Given the input dictionary, this function returns the
              value associated with the input word and language.
    >>> translate({'river': {'Spanish': 'rio', 'French': 'riviere'}}, 'river', 'Spanish')
    'rio'
    >>> translate({'river': {'Spanish': 'rio', 'French': 'riviere'}}, 'river', 'French')
    'riviere'
    """
    pass

def nestedDictionary3D(L1, L2):
    """
    Requires: L1 and L2 are lists
    Modifies: Nothing
    Effects:  Creates a 3D dictionary, D, with keys of each item of list L1.
              The value for each key in D is a dictionary, which
              has keys of each item of list L2 and corresponding
              values of empty dictionaries. Returns the new dictionary D.
    >>> nestedDictionary3D(['abbey road'], ['come together', 'because'])
    {'abbey road': {'come together': {}, 'because': {}}}
    >>> albums = ['help', 'revolver']
    >>> attributes = ['sales', 'songs']
    >>> nestedDictionary3D(albums, attributes)
    {'help': {'sales': {}, 'songs': {}}, 'revolver': {'sales': {}, 'songs': {}}}
    """
    pass

def valueFrom3D(D, K1, K2, K3):
    """
    Requires: D is a 3D dictionary, K1 is a key in D, K2 is a key in the
              dictionary that K1 maps to, and K3 is a key in the dictionary
              that K2 maps to
    Modifies: Nothing
    Effects:  Given the 3D input dictionary D, returns the value associated
              with the innermost dictionary accessed using keys K1, K2, and K3,
              in that order.
    >>> valueFrom3D({'any': {'time': {'at': 'all'}}}, 'any', 'time', 'at')
    'all'
    >>> valueFrom3D({'twist': {'and': {'shout': 5}}}, 'twist', 'and', 'shout')
    5
    """
    pass

def keysIn2D(D, L1, L2):
    """
    Requires: D is a 2D dictionary, L1 is a list, and L2 is a list
    Modifies: Nothing
    Effects:  Given a 2D input dictionary D, returns True if and only
              if the last item of list L1 is a key in D, and that key
              is associated with a dictionary that contains the last
              item of list L2 as a key.
    >>> D = {'d#2': {'e3': 'ab3'}}
    >>> keysIn2D(D, ['c4', 'f#6'], ['ab3', 'd5', 'e3'])
    False
    >>> D = {'f#6': {'e3': 5}}
    >>> keysIn2D(D, ['c4', 'f#6'], ['ab3', 'd5', 'e3'])
    True
    """
    pass

class warmup(object):
    """A simple class with methods to get you used to how python classes work."""

    def makeBand(self, band):
        """
        Requires: nothing
        Modifies: self
        Effects: creates a self.bandName attribute and sets it to be the band that
                 gets passed in.
        >>> w = warmup()
        >>> w.makeBand('The Beatles')
        >>> w.bandName
        'The Beatles'
        """
        pass

    def setAlbum(self, album):
        """
        Requires: album is a string value.
        Modifies: self
        Effects: sets the warmup's self.album attribute to be the album that gets
                 passed in.
        >>> w = warmup()
        >>> w.setAlbum('Abbey Road')
        >>> w.album
        'Abbey Road'
        """
        pass

    def printAlbum(self):
        """
        Requires: nothing
        Modifies: nothing
        Effects: returns <the band's album> + " by " + <the band's name>
        >>> w = warmup()
        >>> w.makeBand('The Beatles')
        >>> w.setAlbum('Twist and Shout')
        >>> w.printAlbum()
        'Twist and Shout by The Beatles'
        """
        pass

    #hint: use the functions you defined above!
    def __init__(self, name_in = " ", album_in = " "):
        """
        Requires: nothing
        Modifies: self
        Effects: Constructor
                 Default (no arguments given): Sets self.bandName to "No name" and self.album to "No album"
                 Non-Default (arguments given): Sets self.bandName to name_in and self.album to album_in
        >>> w = warmup()
        >>> w.bandName
        'No name'
        >>> w.album
        'No album'
        >>> a = warmup('Smash Mouth', 'Astro Lounge')
        >>> a.bandName
        'Smash Mouth'
        >>> a.album
        'Astro Lounge'
        """
        pass


###############################################################################
# Main
###############################################################################

if __name__ == '__main__':
    import doctest
    doctest.testmod()
