from creative_ai.utils.print_helpers import ppGramJson

class UnigramModel():

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
        Modifies: self.nGramCounts
        Effects:  this function populates the self.nGramCounts dictionary,
                  which is a dictionary of {string: integer} pairs.
                  For further explanation of UnigramModel's version of
                  self.nGramCounts, see the spec.
                  Returns self.nGramCounts
        """
        for index in range(len(text)):
            for index2 in range(2,len(text[index])):
                store = text[index][index2]
                if (store in self.nGramCounts):
                    self.nGramCounts[store] = self.nGramCounts[store] + 1
                else:
                    self.nGramCounts[store] = 1
        return self.nGramCounts

    def trainingDataHasNGram(self, sentence):
        """
        Requires: sentence is a list of strings
        Modifies: nothing
        Effects:  returns True if this n-gram model can be used to choose
                  the next token for the sentence. For explanations of how this
                  is determined for the UnigramModel, see the spec.
        """
        store = len(self.nGramCounts)
        if (store > 0):
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
                  UnigramModel sees as candidates, see the spec.
        """

        return self.nGramCounts

###############################################################################
# End Core
###############################################################################

###############################################################################
# Main
###############################################################################

# This is the code python runs when unigramModel.py is run as main
if __name__ == '__main__':

    # An example trainModel test case
    uni1 = UnigramModel()
    text = [["^::^", "^:::^", 'brown', "$:::$"]]
    uni1.trainModel(text)
    # Should print: { 'brown' : 1 }
    print(uni1)

    text = [["^::^", "^:::^", 'the', 'brown', 'fox', "$:::$"], ["^::^", "^:::^", 'the', 'lazy', 'dog', "$:::$"]]
    uni1.trainModel(text)
    # Should print: { 'brown': 2, 'dog': 1, 'fox': 1, 'lazy': 1, 'the': 2 }
    print(uni1)

    # Second example trainModel test case
    uni3 = UnigramModel()
    text2 = [['Two']]
    uni3.trainModel(text2)
    # Should print: { 'Two' : 1 }
    print(uni3)

    text2 = [['two', 'driven', 'jocks', 'help'], ['fax', 'my', 'big', 'quiz']]
    uni3.trainModel(text2)
    # Should print: { 'Two': 1, 'big': 1, 'driven': 1, 'fax': 1, 'help': 1, 'jocks': 1, 'my': 1, 'quiz': 1, 'two': 1 }
    print(uni3)

    # Third example trainModel test case
    uni4 = UnigramModel()
    text3 = [['AAAAA']]
    uni4.trainModel(text3)
    # Should print: { 'AAAAA' : 1 }
    print(uni4)

    text3 = [['AAAAA', 'AZAZA', 'ABBBB', 'AABBB'], ['kmnjbh', 'iyg', 'a', 'b']]
    uni4.trainModel(text3)
    # Should print: { 'AAAAA': 2, 'AABBB': 1, 'ABBBB': 1, 'AZAZA': 1, 'a': 1, 'b': 1, 'iyg': 1, 'kmnjbh': 1 }
    print(uni4)

    # Fourth example trainModel test case
    uni5 = UnigramModel()
    text4 = [['a']]
    uni5.trainModel(text4)
    # Should print: { 'a' : 1 }
    print(uni5)

    text4 = [['a', 'a', 'a', 'a'], ['a', 'a', 'a', 'a']]
    uni5.trainModel(text4)
    # Should print: { 'a': 9}
    print(uni5)

    # Fifth example trainModel test case
    uni6 = UnigramModel()
    text5 = [['2']]
    uni6.trainModel(text5)
    # Should print: { '2' : 1 }
    print(uni6)

    text5 = [['2', '30', '40', '500!'], ['1000', '600000', '-7', '.9']]
    uni6.trainModel(text5)
    # Should print: { '-7': 1, '.9': 1, '1000': 1, '2': 2, '30': 1, '40': 1, '500!': 1, '600000': 1, }
    print(uni6)

    # An example trainingDataHasNGram test case
    uni7 = UnigramModel()
    sentence1 = [['Eagles', 'fly', 'in', 'the', 'sky']]
    print(uni7.trainingDataHasNGram(sentence1))  # should be False
    uni7.trainModel(text)
    print(uni7.trainingDataHasNGram(sentence1))  # should be True

    # A second example trainingDataHasNGram test case
    uni8 = UnigramModel()
    sentence = [['a']]
    print(uni8.trainingDataHasNGram(sentence))  # should be False
    uni8.trainModel(text3)
    print(uni8.trainingDataHasNGram(sentence))  # should be True

    # A third example trainingDataHasNGram test case
    uni9 = UnigramModel()
    sentence = [['00000']]
    print(uni9.trainingDataHasNGram(sentence))  # should be False
    uni9.trainModel(text4)
    print(uni9.trainingDataHasNGram(sentence))  # should be True

    # A fourth example trainingDataHasNGram test case
    uni10 = UnigramModel()
    sentence = [['AAAAAAAA', '!!!!!!???']]
    print(uni10.trainingDataHasNGram(sentence))  # should be False
    uni10.trainModel(text5)
    print(uni10.trainingDataHasNGram(sentence))  # should be True

    # A getCandidateDictionary test case
    uni11 = UnigramModel()
    text = [['the', 'brown', 'fox'], ['the', 'lazy', 'dog']]
    uni11.trainModel(text)
    uni12 = uni11.getCandidateDictionary(text)
    print(uni12)

    # A second getCandidateDictionary test case
    uni13 = UnigramModel()
    uni13.trainModel(text2)
    uni14 = uni13.getCandidateDictionary(text2)
    print(uni14)

    # A third getCandidateDictionary test case
    uni15 = UnigramModel()
    uni15.trainModel(text3)
    uni16 = uni15.getCandidateDictionary(text3)
    print(uni16)

    # A fourth getCandidateDictionary test case
    uni17 = UnigramModel()
    uni17.trainModel(text4)
    uni18 = uni17.getCandidateDictionary(text4)
    print(uni18)

    # A fifth getCandidateDictionary test case
    uni19 = UnigramModel()
    uni19.trainModel(text5)
    uni20 = uni19.getCandidateDictionary(text5)
    print(uni20)

    # A sixth getCandidateDictionary test case
    uni21 = UnigramModel()
    text6 = [['?????', '777777', 'AAAAA'], ['0A0A0A0', '1', 'z']]
    uni21.trainModel(text6)
    uni22 = uni21.getCandidateDictionary(text6)
    print(uni22)
