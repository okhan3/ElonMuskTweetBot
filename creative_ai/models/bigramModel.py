from creative_ai.utils.print_helpers import ppGramJson

class BigramModel():

    def __init__(self):
        """
        Requires: nothing
        Modifies: self (this instance of the NGramModel object)
        Effects:  This is the NGramModel constructor. It sets up an empty
                  dictionary as a member variable.

        This function is done for you.
        """
        self.nGramCounts = {}

    def __str__(self):
        """
        Requires: nothing
        Modifies: nothing
        Effects:  Returns the string to print when you call print on an
                  NGramModel object. This string will be formatted in JSON
                  and display the currently trained dataset.

        This function is done for you.
        """

        return ppGramJson(self.nGramCounts)


###############################################################################
# Begin Core >> FOR CORE IMPLEMENTION, DO NOT EDIT ABOVE OF THIS SECTION <<
###############################################################################

    def trainModel(self, text):
        """
        Requires: text is a list of lists of strings
        Modifies: self.nGramCounts, a two-dimensional dictionary. For examples
                  and pictures of the BigramModel's version of
                  self.nGramCounts, see the spec.
        Effects:  this function populates the self.nGramCounts dictionary,
                  which has strings as keys and dictionaries of
                  {string: integer} pairs as values.
                  Returns self.nGramCounts

        >>> x = BigramModel()
        >>> y = [['strawberry', 'fields', 'nothing', 'is', 'real'], ['strawberry', 'fields', 'forever']]
        >>> x.trainModel(y)
        {'strawberry': {'fields': 2}, 'fields': {'nothing': 1, 'forever': 1}, 'nothing': {'is': 1}, 'is': {'real': 1}}
        >>> z = [['^::^', '^:::^', 'strawberry', 'fields', 'nothing', 'is', 'real', '$:::$'], ['^::^', '^:::^', 'strawberry', 'fields', 'forever', '$:::$']]
        >>> x.__init__()
        >>> x.trainModel(z)
        {'^::^': {'^:::^': 2}, '^:::^': {'strawberry': 2}, 'strawberry': {'fields': 2}, 'fields': {'nothing': 1, 'forever': 1}, 'nothing': {'is': 1}, 'is': {'real': 1}, 'real': {'$:::$': 1}, 'forever': {'$:::$': 1}}
        """

        # Iterates through 2D list
        for i in range (0,len(text)):
            for j in range (1,len(text[i])):
                # Sets first term as seed
                seed = text[i][j-1]
                if seed not in self.nGramCounts:
                    # Adds a new dictionary within first dictionary
                    self.nGramCounts[seed] = {text[i][j]: 1}
                elif text[i][j] not in self.nGramCounts[seed]:
                    # Adds a new item within second dictionary
                    self.nGramCounts[seed][text[i][j]] = 1
                else:
                    # Increments word frequency
                    self.nGramCounts[seed][text[i][j]] += 1

        return self.nGramCounts

    def trainingDataHasNGram(self, sentence):
        """
        Requires: sentence is a list of strings
        Modifies: nothing
        Effects:  returns True if this n-gram model can be used to choose
                  the next token for the sentence. For explanations of how this
                  is determined for the BigramModel, see the spec.

        >>> x = BigramModel()
        >>> x.nGramCounts = {'strawberry': {'fields': 2}, 'fields': {'nothing': 1, 'forever': 1}, 'nothing': {'is': 1}, 'is': {'real': 1}}
        >>> x.trainingDataHasNGram(['I', 'have', 'a', 'strawberry'])
        True
        >>> x.trainingDataHasNGram(['I', 'am', 'not', 'real'])
        False
        """

        # Checks if the last item in list 'sentence' is a key in nGramCounts dictionary
        if sentence[len(sentence)-1] in self.nGramCounts:
            return True
        else:
            return False

    def getCandidateDictionary(self, sentence):
        """
        Requires: sentence is a list of strings, and trainingDataHasNGram
                  has returned True for this particular language model
        Modifies: nothing
        Effects:  returns the dictionary of candidate next words to be added
                  to the current sentence. For details on which words the
                  BigramModel sees as candidates, see the spec.

        >>> x = BigramModel()
        >>> x.nGramCounts = {'strawberry': {'fields': 2}, 'fields': {'nothing': 1, 'forever': 1}, 'nothing': {'is': 1}, 'is': {'real': 1}}
        >>> x.getCandidateDictionary(['I', 'have', 'a', 'strawberry'])
        {'fields': 2}
        >>> x.getCandidateDictionary(['I', 'have', 'many', 'fields'])
        {'nothing': 1, 'forever': 1}
        """

        # Returns dictionary of all candidate words
        x = self.nGramCounts[sentence[len(sentence)-1]]
        return x

###############################################################################
# End Core
###############################################################################

###############################################################################
# Main
###############################################################################

if __name__ == '__main__':
    import doctest
    doctest.testmod()
