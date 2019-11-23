import random
from creative_ai.data.dataLoader import prepData
from creative_ai.models.unigramModel import UnigramModel
from creative_ai.models.bigramModel import BigramModel
from creative_ai.models.trigramModel import TrigramModel
from creative_ai.utils.print_helpers import key_value_pairs

class LanguageModel():

    def __init__(self, models=None):
        """
        Requires: nothing
        Modifies: self (this instance of the LanguageModel object)
        Effects:  This is the LanguageModel constructor. It sets up an empty
                  dictionary as a member variable.
        
        This function is done for you.
        """

        if models != None:
            self.models = models
        else:
            self.models = [TrigramModel(), BigramModel(), UnigramModel()]

    def __str__(self):
        """
        Requires: nothing
        Modifies: nothing
        Effects:  This is a string overloaded. This function is
                  called when languageModel is printed.
                  It will show the number of trained paths
                  for each model it contains. It may be
                  useful for testing.
        
        This function is done for you.
        """

        output_list = [
            '{} contains {} trained paths.'.format(
                model.__class__.__name__, key_value_pairs(model.nGramCounts)
                ) for model in self.models
            ]

        output = '\n'.join(output_list)

        return output

    def updateTrainedData(self, text, prepped=True):
        """
        Requires: text is a 2D list of strings
        Modifies: self (this instance of the LanguageModel object)
        Effects:  adds new trained data to each of the languageModel models.
        If this data is not prepped (prepped==False) then it is prepepd first
        before being passed to the models.

        This function is done for you.
        """

        if (not prepped):
            text = prepData(text)

        for model in self.models:
            model.trainModel(text)


###############################################################################
# Begin Core >> FOR CORE IMPLEMENTION, DO NOT EDIT ABOVE OF THIS SECTION <<
###############################################################################

    def selectNGramModel(self, sentence):
        """
        Requires: self.models is a list of NGramModel objects sorted by descending
                  priority: tri-, then bi-, then unigrams.

                  sentence is a list of strings.
        Modifies: nothing
        Effects:  returns the best possible model that can be used for the
                  current sentence based on the n-grams that the models know.
                  (Remember that you wrote a function that checks if a model can
                  be used to pick a word for a sentence!)
        """
        pass

    def weightedChoice(self, candidates):
        """
        Requires: candidates is a dictionary; the keys of candidates are items
                  you want to choose from and the values are integers
        Modifies: nothing
        Effects:  returns a candidate item (a key in the candidates dictionary)
                  based on the algorithm described in the spec.
        """
        
        keysList = []
        valuesList = []
        cumulativeList = []
        
        for i in candidates:
            keysList.append(i)
            valuesList.append(candidates[i])
    
        
        cumulativeList[0] = valuesList[0]
        for i in range(1, lens(valuesList) - 1):
            cumulativeList.append(valuesList[i] + cumulativeList[i - 1])
            
        x = random.randrange(0, valuesList[len(valuesList) - 1])
        
        j = 0
        while x >= cumulativeList[j]:
            j += 1
        
        return keysList[j]


    def getNextToken(self, sentence, filter=None):
        """
        Requires: sentence is a list of strings, and this model can be used to
                  choose the next token for the current sentence
        Modifies: nothing
        Effects:  returns the next token to be added to sentence by calling
                  the getCandidateDictionary and weightedChoice functions.
                  For more information on how to put all these functions
                  together, see the spec.

                  If a filter is being used, and none of the models
                  can produce a next token using the filter, then a random
                  token from the filter is returned instead.
        """
        pass

###############################################################################
# End Core
###############################################################################

###############################################################################
# Main
###############################################################################

if __name__ == '__main__':
  
  test1 = LanguageModel()
  dict1 = {"north" : 4, "south" : 1, "east" : 3, "west" : 2}
  
  print("Test1 output should be east: ", test1.weightedChoice(dict1))
  
