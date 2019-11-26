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
# Begin Core >> FOR CORE IMPLEMENTATION, DO NOT EDIT ABOVE OF THIS SECTION <<
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

        """ iterates through list from models starting with trigram -> bigram -> unigram """
        for i in range(len(self.models)):

            """ checks which list model can be used for the sentence """
            if (self.models[i].trainingDataHasNGram(sentence)):
                return self.models[i]


    def weightedChoice(self, candidates):
        """
        Requires: candidates is a dictionary; the keys of candidates are items
                  you want to choose from and the values are integers
        Modifies: nothing
        Effects:  returns a candidate item (a key in the candidates dictionary)
                  based on the algorithm described in the spec.
        """

        keysList = list(candidates.keys())
        valuesList = list(candidates.values())

        """ for loop calls cumulativeList[i-1], so the first element is populated outside of the loop """
        cumulativeList = []
        cumulativeList.append(valuesList[0])

        for i in range(1, len(valuesList)):
            """ creates a cumulative list containing the values in valuesList """
            cumulativeList.append(valuesList[i] + cumulativeList[i - 1])

        """ random number generator. comment-in print command while testing """
        x = random.randrange(0, cumulativeList[len(cumulativeList) - 1])

        """ finds the first element in cumulativeList that is greater than x """
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
        B = self.selectNGramModel(sentence)
        D = B.getCandidateDictionary(sentence)
        if (filter == None):
            S = self.weightedChoice(D)
            return S
        else:
            filteredCandidates = {}
            store = D.keys()
            #check index out of bounds
            for index in range(len(D)):
                if store(index) == filter[index]:
                  filteredCandidates[store(index)] = D[index]
            if filteredCandidates[len(filteredCandidates) - 1] == None:
                x = random.choice(filter)
                return filter[x]
            else:
                r = self.weightedChoice(filteredCandidates)
                return r



###############################################################################
# End Core
###############################################################################

###############################################################################
# Main
###############################################################################

if __name__ == '__main__':

    print("Now Testing selectNGramModel()")
    print()

    trigramText1 = [['this', 'is', 'a', 'test', 'case'], ['this', 'is', 'very', 'fun'],
                  ['this', 'test', 'case', 'should', 'work']]

    trigramText2 = [['this', 'is', 'a', 'test', 'case'], ['this', 'is', 'a', 'test', 'case'],
                   ['this', 'test', 'case', 'should', 'work']]

    bigramText = [['eecs', 'is'], ['love', 'eecs'], ['is', 'life'], ['eecs', 'love'], ['eecs', 'life'],
                  ['love', 'life'], ['love', 'eecs']]

    unigramText = [['test'], ['cases'], ['test'], ['cases'], ['fun'], ['fun']]


    trigramTest1 = LanguageModel()
    trigramTest1.updateTrainedData(trigramText1)
    print("Should Print: {"'a'": {"'test'": {"'case'": 1}}, "'case'": {"'should'": {"'work'": 1}},"
          " "'is'": {"'a'": {"'test'": 1},"'very'": {"'fun'": 1}}, "'test'": {"'case'": {"'should'": 1}},"
          ""'this'": {"'is'": {"'a'": 1,"'very'": 1}, "'test'": {"'case'": 1}}}")
    print(trigramTest1.selectNGramModel(trigramText1[0]))
    print(trigramTest1)
    print()

    trigramTest2 = LanguageModel()
    trigramTest2.updateTrainedData(trigramText2)
    print("Should Print: {"'a'": {"'test'": {"'case'": 2}}, "'case'": {"'should'": {"'work'": 1}},"
          " "'is'": {"'a'": {"'test'": 2}}, "'test'": {"'case'": {"'should'": 1}},"
          ""'this'": {"'is'": {"'a'": 2, {"'test'": {"'case'": 1}}}")
    print(trigramTest2.selectNGramModel(trigramText2[0]))
    print(trigramTest2)
    print()

    bigramTest = LanguageModel()
    bigramTest.updateTrainedData(bigramText)
    print("Should Print: {'eecs': {'is': 1, 'life': 1, 'love': 1}, 'is', {'life': 1},"
          "'love': {'eecs': 2, 'life': 1}}")
    print(bigramTest.selectNGramModel(bigramText[0]))
    print(bigramTest)
    print()

    unigramTest = LanguageModel()
    unigramTest.updateTrainedData(unigramText)
    print("Should Print: {'cases': 2, 'fun': 2, 'test': 2}")
    print(unigramTest.selectNGramModel(unigramText[0]))
    print(unigramTest)
    print()

    print("Finished Testing selectNGramModel")
    print()
    print()





    print("Now Testing weightedChoice()")
    print()

    print("Test 1 Expected Output -- comment in print(x)")
    print("when number is 0 - 3: north")
    print("when number is 4: south")
    print("when number is 5 - 7: east")
    print("when number is 8 - 9: west")

    test1 = LanguageModel()
    dict1 = {"north" : 4, "south" : 1, "east" : 3, "west" : 2}

    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()
    print("Output: ", test1.weightedChoice(dict1))
    print()

    print("Test 2 Expected Output -- comment in print(x)")
    print("when number is 0 - 2: Alex")
    print("when number is 3 - 4: Is")
    print("when number is 5 - 8: Very")
    print("when number is 9 - 11: Crazy")

    test2 = LanguageModel()
    dict2 = {"Alex" : 3, "Is" : 2, "Very" : 4, "Crazy" : 3}

    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()
    print("Output: ", test2.weightedChoice(dict2))
    print()

    print("Finished Testing weightedChoice")
    print()
    print()




    """start at test 6"""
    test10 = LanguageModel()
    sentence = [['Eagles', 'fly', 'in', 'the', 'sky']]
    test10.updateTrainedData(sentence)
    testVal = test10.getNextToken(sentence)
    print (testVal)