import random

import spacy
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from spacy.tokens import Doc
from creative_ai.data.dataLoader import prepData, prepTweetData, prepLinkData
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

    def updateTrainedTweetData(self, text, prepped=True):
        """
        Requires: text is a 2D list of strings
        Modifies: self (this instance of the LanguageModel object)
        Effects:  adds new trained data to each of the languageModel models.
        If this data is not prepped (prepped==False) then it is prepepd first
        before being passed to the models.
        This function is done for you.
        """

        if (not prepped):
            text = prepTweetData(text)

        for model in self.models:
            model.trainModel(text)

    def updateTrainedLinkData(self, text, prepped=True):
        """
        Requires: text is a 2D list of strings
        Modifies: self (this instance of the LanguageModel object)
        Effects:  adds new trained data to each of the languageModel models.
        If this data is not prepped (prepped==False) then it is prepepd first
        before being passed to the models.
        This function is done for you.
        """

        if (not prepped):
            text = prepLinkData(text)

        for model in self.models:
            model.trainModel(text)



###############################################################################
# Begin Core >> FOR CORE IMPLEMENTATION, DO NOT EDIT ABOVE OF THIS SECTION <<
###############################################################################

    def refine(self):
        nlp = English()
        #change to .lg if needed
        nlp = spacy.load('en')  # load model with shortcut link "en"

        # Created by processing a string of text with the nlp object
        doc = nlp("Hello my name is Nikhil.")
        print(nlp.pipe_names)

        print('parts of speech')
        for token in doc:
            # Print the text and the predicted part-of-speech tag
            #use to indentify parts of speech in a sentance
            print(token.text, token.pos_, token.dep_)
        print()

        print('entities')
        #iterates over entities in doc
        for ent in doc.ents:
            #prints name of enity and their label
            #use to identify Tesla, SpaceX and NASA
            print(ent.text, ent.label_)
        print()

        print('exploring matcher')
        #could be used to match organization names SpaceX, Tesla
        matcher = Matcher(nlp.vocab)
        #also works for {"IS_DIGIT": True}
        pattern = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]
        matcher.add('IPHONE_PATTERN', None, pattern)
        # Process some text
        doc = nlp("New iPhone X release date leaked")
        # Call the matcher on the doc
        matches = matcher(doc)
        # Iterate over the matches
        for match_id, start, end in matches:
            # Get the matched span
            matched_span = doc[start:end]
            print(matched_span.text)
        print()

        print('type of text')
        for token in doc:
            # Index into the Doc to get a single Token
            print(token.text)
            print(token.is_punct)
            print(token.like_num)
        print('is_alpha:', [token.is_alpha for token in doc])
        print('is_punct:', [token.is_punct for token in doc])
        print('like_num:', [token.like_num for token in doc])
        #gets the hash value
        print(nlp.vocab.strings['iPhone'])

        print()
        matcher = PhraseMatcher(nlp.vocab)

        pattern = nlp("Golden Retriever")
        matcher.add('DOG', None, pattern)
        doc = nlp("I have a Golden Retriever")

        # Iterate over the matches
        for match_id, start, end in matcher(doc):
            # Get the matched span
            span = doc[start:end]
            print('Matched span:', span.text)

        nlp.begin_training


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

        nlp = English()
        # change to .lg if needed
        nlp = spacy.load('en_core_web_sm')  # load model with shortcut link "en"

        # Created by processing a string of text with the nlp object
        doc = nlp(str(sentence))

        for token in doc:
            # Print the text and the predicted part-of-speech tag
            # use to indentify parts of speech in a sentance
            if(token.text != '^:::^' or token.text != "^::^" or token.text != "$:::$" or token.text != '[' or token.text != ']' or token.text != ','):
                store = (token.text, token.pos_, token.dep_)

        if (filter == None):
            S = self.weightedChoice(D)
            return S
        else:
            filteredCandidates = {}
            store = list(D.keys())
            #check index out of bounds
            smaller = filter
            larger = store
            if (len(filter) >= len(store)):
                larger = filter
                smaller = store
            else:
                smaller = filter
                larger = store
                #replace filter with larger
                #replace store with smaller
            for index in range(len(larger)):
                for index2 in range(len(smaller)):
                    if smaller[index2] == larger[index]:
                        filteredCandidates[smaller[index2]] = D[smaller[index2]]

            #filteredCandidates = index for index in store if index in filter
            if not bool(filteredCandidates):
                x = random.choice(filter)
                return x
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
    #hello = LanguageModel()
    #hello.refine()


    print("Now Testing selectNGramModel()")
    print()

    trigramText1 = [['this', 'is', 'a', 'test', 'case'], ['this', 'is', 'very', 'fun'],
                  ['this', 'test', 'case', 'should', 'work']]

    trigramText2 = [['this', 'is', 'a', 'test', 'case'], ['this', 'is', 'a', 'test', 'case'],
                   ['this', 'test', 'case', 'should', 'work']]

    trigramText3 = [['a', 'a', 'a', 'a', 'a'], ['a', 'a', 'a', 'a', 'a'],
                   ['a', 'a', 'a', 'a', 'a']]

    bigramText = [['eecs', 'is'], ['love', 'eecs'], ['is', 'life'], ['eecs', 'love'], ['eecs', 'life'],
                  ['love', 'life'], ['love', 'eecs']]

    bigramText2 = [['a', 'a'], ['a', 'a'], ['a', 'a'], ['a', 'a'], ['a', 'a'],
                  ['a', 'a'], ['a', 'a']]

    unigramText = [["^::^", "^:::^", 'test', "$:::$"], ["^::^", "^:::^", 'cases', "$:::$"], ["^::^", "^:::^", 'test', "$:::$"],
                   ["^::^", "^:::^", 'cases', "$:::$"], ["^::^", "^:::^", 'fun', "$:::$"], ["^::^", "^:::^", 'fun', "$:::$"]]

    unigramText2 = [["^::^", "^:::^", 'a', "$:::$"], ["^::^", "^:::^", 'a', "$:::$"], ["^::^", "^:::^", 'a', "$:::$"],
                   ["^::^", "^:::^", 'a', "$:::$"], ["^::^", "^:::^", 'a', "$:::$"], ["^::^", "^:::^", 'a', "$:::$"]]


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

    trigramTest3 = LanguageModel()
    trigramTest3.updateTrainedData(trigramText3)
    print("Should Print: {"'a'": {"'a'": {"'a'": 9}}}}")
    print(trigramTest3.selectNGramModel(trigramText3[0]))
    print(trigramTest3)
    print()

    bigramTest = LanguageModel()
    bigramTest.updateTrainedData(bigramText)
    print("Should Print: {'eecs': {'is': 1, 'life': 1, 'love': 1}, 'is', {'life': 1},"
          "'love': {'eecs': 2, 'life': 1}}")
    print(bigramTest.selectNGramModel(bigramText[0]))
    print(bigramTest)
    print()

    bigramTest2 = LanguageModel()
    bigramTest2.updateTrainedData(bigramText2)
    print("Should Print: {'a': {'a': 7}}")
    print(bigramTest2.selectNGramModel(bigramText2[0]))
    print(bigramTest2)
    print()

    unigramTest = LanguageModel()
    unigramTest.updateTrainedData(unigramText)
    print("Should Print: {'cases': 2, 'fun': 2, 'test': 2}")
    print(unigramTest.selectNGramModel(unigramText[0]))
    print(unigramTest)
    print()

    unigramTest2 = LanguageModel()
    unigramTest2.updateTrainedData(unigramText2)
    print("Should Print: {'cases': 2, 'fun': 2, 'test': 2}")
    print(unigramTest2.selectNGramModel(unigramText2[0]))
    print(unigramTest2)
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

    i = 0
    n = 0
    s = 0
    e = 0
    w = 0
    while i < 500:
        x = test1.weightedChoice(dict1)
        if x == "north":
            n += 1
        if x == "south":
            s += 1
        if x == "east":
            e += 1
        if x == "west":
            w += 1
        i += 1
    print("We expect descending order to be north, east, west, south")
    print("North count: ", n)
    print("South count: ", s)
    print("East count: ", e)
    print("West count: ", w)


    print("Test 2 Expected Output -- comment in print(x)")
    print("when number is 0 - 2: Alex")
    print("when number is 3 - 4: Is")
    print("when number is 5 - 8: Very")
    print("when number is 9 - 11: Crazy")

    test2 = LanguageModel()
    dict2 = {"Alex" : 3, "Is" : 2, "Very" : 4, "Crazy" : 3}
    dict3 = {"Alex": 4, "Hello": 4, "World": 4, "moon": 2}

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
    print("Output: ", test2.weightedChoice(dict3))
    print()

    print("Finished Testing weightedChoice")
    print()
    print()



    print("Now Testing getNextToken()")
    print()

    testNonFilter1 = LanguageModel()
    sentence1 = [['Eagles', 'fly', 'in', 'the', 'sky']]
    testNonFilter1.updateTrainedData(sentence1)
    testVal1 = testNonFilter1.getNextToken(sentence1)
    print(testVal1)

    testNonFilter2 = LanguageModel()
    sentence2 = [["^::^", "^:::^", 'rocket', 'to', 'the', 'moon', "$:::$"],
                ["^::^", "^:::^", 'dad', 'is', 'from', 'nyc', "$:::$"],
                ["^::^", "^:::^", 'the', 'quick', 'brown', 'dog', 'barked', "$:::$"],
                ["^::^", "^:::^", 'hello', 'world', 'I', 'am', 'Macintosh', "$:::$"],
                ["^::^", "^:::^", 'I', 'am', 'from', 'mars', "$:::$"],
                ["^::^", "^:::^",'A','friend','went','to','the','moon',"$:::$"],
                ["^::^", "^:::^",'The','new','Macbook','Pro','has','a','better','keyboard','and','it','sells', "well","$:::$"],
                ["^::^", "^:::^", 'My', 'hopes', 'are', 'high', "$:::$"]]
    testNonFilter2.updateTrainedData(sentence2)
    testVal2 = testNonFilter2.getNextToken(sentence2)
    print(testVal2)

    testNonFilter3 = LanguageModel()
    sentence3 = [['Eagles', 'Eagles', 'Eagles'], ['Eagles', 'Eagles', 'Eagles'], ['Eagles', 'Eagles', 'Eagles']]
    testNonFilter3.updateTrainedData(sentence3)
    testVal3 = testNonFilter3.getNextToken(sentence3)
    print(testVal3)

    testNonFilter4 = LanguageModel()
    sentence4 = [['Eagles', 'Eagles', 'Eagles']]
    testNonFilter4.updateTrainedData(sentence4)
    testVal4 = testNonFilter4.getNextToken(sentence4)
    print(testVal4)

    sentence5 = ["^::^", "^:::^", 'rocket', 'to', 'the', 'moon', "$:::$"]
    testNonFilter5 = LanguageModel()
    testNonFilter5.updateTrainedData(sentence2)
    testVal5 = testNonFilter5.getNextToken(sentence5)
    print(testVal5)

    testFilter1 = LanguageModel()
    sentence6 = [['Eagles', 'Eagles', 'Eagles', 'fly', 'in', 'the', 'sky']]
    filter1 = ['Eagles', 'Eagles', 'Eagles', 'die']
    testFilter1.updateTrainedData(sentence6)
    testVal6 = testFilter1.getNextToken(sentence6, filter1)
    print(testVal6)

    testFilter2 = LanguageModel()
    sentence7 = [['Eagles', 'Eagles', 'Eagles', 'fly', 'in', 'the', 'sky']]
    filter2 = ['Eagles', 'Eagles', 'Eagles', 'fly', 'die']
    testFilter2.updateTrainedData(sentence7)
    testVal7 = testFilter2.getNextToken(sentence7, filter2)
    print(testVal7)

    testFilter3 = LanguageModel()
    sentence8 = [['Eagles', 'Eagles', 'Eagles', 'fly', 'in', 'the', 'sky']]
    filter3 = ['Aidan', 'Nikhil', 'Joe']
    testFilter3.updateTrainedData(sentence8)
    testVal8 = testFilter3.getNextToken(sentence8, filter3)
    print(testVal8)
    
    testFilter4 = LanguageModel()
    sentence9 = [['rocket', 'to', 'the', 'moon']]
    filter4 = ['moon', 'mars', 'venus']
    testFilter4.updateTrainedData(sentence9)
    testVal9 = testFilter4.getNextToken(sentence9, filter4)
    print(testVal9)

    testFilter5 = LanguageModel()
    sentence10 = [['rocket', 'to', 'moon']]
    filter5 = ['moon', 'mars', 'venus']
    testFilter5.updateTrainedData(sentence10)
    testVal10 = testFilter5.getNextToken(sentence10, filter5)
    print(testVal10)

    testFilter6 = LanguageModel()
    sentence11 = [['rocket', 'to', 'moon']]
    filter6 = ['not moon', 'mars', 'venus']
    testFilter6.updateTrainedData(sentence11)
    testVal11 = testFilter5.getNextToken(sentence11)
    print(testVal11)

    testFilter7 = LanguageModel()
    sentence12 = [['rocket', 'to', 'moon', 'goal', 'of', 'NASA', 'by', '2023']]
    filter7 = ['not moon', 'mars', 'venus']
    testFilter7.updateTrainedData(sentence12)
    testVal12 = testFilter5.getNextToken(sentence12, filter7)
    print(testVal12)
    dict10 = {'Hello':1, 'World':2, "this": 3, "is": 4, 'death':5}
    print(testFilter7.weightedChoice(dict10))

    print("Finished Testing getNextToken")
