from creative_ai.utils.print_helpers import ppGramJson

class TrigramModel():

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
        Modifies: self.nGramCounts, a three-dimensional dictionary. For
                  examples and pictures of the TrigramModel's version of
                  self.nGramCounts, see the spec.
        Effects:  this function populates the self.nGramCounts dictionary,
                  which has strings as keys and dictionaries as values,
                  where those inner dictionaries have strings as keys
                  and dictionaries of {string: integer} pairs as values.
                  Returns self.nGramCounts
        """
        # Iterates through 2D list
        for i in range (0,len(text)):
            for j in range (2,len(text[i])):
                # Sets first term as seed
                seed = text[i][j-2]
                if seed not in self.nGramCounts:
                    # Adds a new dictionary within first dictionary
                    self.nGramCounts[seed] = {text[i][j-1]: {text[i][j]: 1}}
                elif text[i][j-1] not in self.nGramCounts[seed]:
                    # Adds a new dictionary within second dictionary
                    self.nGramCounts[seed][text[i][j-1]] = {text[i][j]: 1}
                elif text[i][j] not in self.nGramCounts[seed][text[i][j-1]]:
                    # Adds a new item within third dictionary
                    self.nGramCounts[seed][text[i][j-1]][text[i][j]] = 1
                else:
                    # Increments word frequency
                    self.nGramCounts[seed][text[i][j-1]][text[i][j]] += 1

        return self.nGramCounts

    def trainingDataHasNGram(self, sentence):
        """
        Requires: sentence is a list of strings
        Modifies: nothing
        Effects:  returns True if this n-gram model can be used to choose
                  the next token for the sentence. For explanations of how this
                  is determined for the TrigramModel, see the spec.
        """
        # Checks if the second to last item in list 'sentence' is a key in nGramCounts dictionary
        str1 = str(sentence[len(sentence)-1])
        str2 = str(sentence[len(sentence)-2])
        if str2 in self.nGramCounts:
            # Checks if the last item in list 'sentence' is a key in second layer of nGramCounts dictionary
            if str1 in self.nGramCounts[str2]:
                return True
        return False

    def getCandidateDictionary(self, sentence):
        """
        Requires: sentence is a list of strings, and trainingDataHasNGram
                  has returned True for this particular language model
        Modifies: nothing
        Effects:  returns the dictionary of candidate next words to be added
                  to the current sentence. For details on which words the
                  TrigramModel sees as candidates, see the spec.
        """
        # Returns dictionary of all candidate words
        x = self.nGramCounts[sentence[len(sentence)-2]][sentence[len(sentence)-1]]
        return x

###############################################################################
# End Core
###############################################################################

###############################################################################
# Main
###############################################################################

if __name__ == '__main__':
    # An example trainModel test case
    uni = TrigramModel()
    x = TrigramModel()

    text = [ ['the', 'brown', 'fox'], ['the', 'lazy', 'dog'] ]
    uni.trainModel(text)
    text = [['strawberry', 'fields', 'nothing', 'is', 'real'], ['strawberry', 'fields', 'forever']]
    x.trainModel(text)

    print(uni)
    print(x)

    print(x.trainingDataHasNGram(['strawberry', 'fields']))
    print(x.trainingDataHasNGram(['strawberry', 'forever']))

    print(x.getCandidateDictionary(['strawberry', 'fields']))
    print(x.getCandidateDictionary(['I', 'love', 'fields', 'nothing']))
