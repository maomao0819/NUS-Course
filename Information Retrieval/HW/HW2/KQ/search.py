#!/usr/bin/python3
from heapq import heappop, heappush
import re
import sys
import getopt
import pickle
from xml.etree.ElementTree import iselement
from nltk.stem import PorterStemmer
import math


# This class functions as a linear data structure that stores items in a First-In-Last-Out (FILO) manner.
# It is implemented with the use of Python's built-in data structure, list.
class Stack:
    def __init__(self):
        """
        Initializes a new Stack.
        """
        self.data = []

    def push(self, item):
        """
        Adds the new item to the top of the Stack.

        Parameters
        ----------
        item : any type
            The new item to be added to the Stack.
        """
        self.data.append(item)

    def pop(self):
        """
        Returns and removes the item at the top of the Stack.
        If there is no item, None will be returned instead.

        Parameters
        ----------
        item : any type
            The new item to be added to the Stack.
        """
        if len(self.data) > 0:
            return self.data.pop()
        else:
            return None

    def peek(self):
        """
        Returns the item at the top of the Stack.
        If there is no item, None will be returned instead.
        """        
        if len(self.data) > 0:
            return self.data[len(self.data) - 1]
        else:
            return None

    def isEmpty(self):
        return len(self.data) == 0

    def size(self):
        return len(self.data)


# global variables for the boolean operators
OPERATOR_NOT = "NOT"
OPERATOR_AND = "AND"
OPERATOR_OR = "OR"

ps = PorterStemmer()


def usage():
    print("usage: " +
          sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below

    # Reads and stores all the terms' dictionary in the dictionary file as a single dictionary (dictionaryTerms) in memory 
    dictFile = open(dict_file, "rb")
    dictUnpickler = pickle.Unpickler(dictFile)
    dictionaryTerms = {}
    try:
        while True:
            termDict = dictUnpickler.load()
            dictionaryTerms[termDict["term"]] = termDict
            # as a term's dictionary consists of the key-value pair ("term", the actual term)
            # need to remove the duplicate value of "term"
            # since the actual term is already used as a key in the single dictionary (dictionaryTerms)
            del dictionaryTerms[termDict["term"]]["term"]
    except EOFError:
        dictFile.close()

    # to get all the docIDs of the training data set, load from the first record of the postings file
    postingFile = open(postings_file, "rb")
    postingsUnpickler = pickle.Unpickler(postingFile)
    docIDs = postingsUnpickler.load()

    # read each query in the queries file, process and evaluate the query
    # then store the result of the query into the results file
    resultsFile = open(results_file, "w")
    isFirstQuery = True
    with open(queries_file) as f:
        for line in f:
            try:
                # substituting multiple spaces with single whitespace
                line = re.sub(' +', ' ', line).strip()
                postFixQuery = shuntingYard(line)
                # if the postfix of the query is empty, then it is an invalid query, so just output empty result
                if len(postFixQuery) == 0:
                    resultsFile.write("\n")
                else:
                    result = evaluateQuery(
                        postFixQuery, docIDs, dictionaryTerms, postingFile)
                    if isFirstQuery:
                        # as the result of the evaluation of the query is a list of docIDs,
                        # need to output it as a string of docIDs separated by a whitespace
                        resultsFile.write(' '.join(str(docID)
                                          for docID in result))
                        isFirstQuery = False
                    else:
                        resultsFile.write("\n" + ' '.join(str(docID)
                                          for docID in result))
            except:
                resultsFile.write("\n")
    postingFile.close()
    resultsFile.close()
    
def shuntingYard(query):
    """
    Returns the postfix expression of the Boolean query.

    Parameters
    ----------
    query : string
        The Boolean query to be processed and evaluated.     
    """
    tokens = getTokenList(query)
    operatorStack = Stack()
    outputQueue = []
    for token in tokens:
        if isTerm(token):
            outputQueue.append(token)
        elif isOperator(token):
            # while there is an operator, o2, other than the left parenthesis at the top of the operator stack,
            # and o2 has higher precedence than operator, o1, remove o2 from the stack and add to the output queue
            while not operatorStack.isEmpty() and operatorStack.peek() != "(" \
                    and isHigherPrecedence(token, operatorStack.peek()):
                outputQueue.append(operatorStack.pop())
            operatorStack.push(token)
        elif token == '(':
            operatorStack.push(token)
        elif token == ')':
            # while the operator at the top of the operator stack is not a left parenthesis,
            # remove the operator from the stack and add to the output queue
            while not operatorStack.isEmpty() and operatorStack.peek() != "(":
                outputQueue.append(operatorStack.pop())

            # If the operator at the top of the stack is a left parenthesis,
            # it is a valid query, discard the left parenthesis.
            # Else, there are mismatched parentheses i.e. invalid query.
            if operatorStack.peek() == "(":
                operatorStack.pop()
            else:
                return []
    # pop the remaining items from the operator stack into the output queue
    while not operatorStack.isEmpty():
        # If the operator at the top of the stack is a left parenthesis,
        # it means that there are mismatched parentheses i.e. invalid query.
        if operatorStack.peek() == "(":
            return []
        outputQueue.append(operatorStack.pop())
    return outputQueue

def getTokenList(query):
    """
    Returns the token list of the Boolean query.

    Parameters
    ----------
    query : string
        The Boolean query to be processed and evaluated.     
    """
    tokens = []
    index = 0
    # iterate through the entire query and convert it into tokens
    while index < len(query):
        # if the extracted portion of the query is a NOT or AND operator,
        # add it as a token to the token list
        # and advance the index by the length of the operator.
        if query[index:index + len(OPERATOR_NOT)] == OPERATOR_NOT or \
            query[index:index + len(OPERATOR_AND)] == OPERATOR_AND:
            tokens.append(query[index:index + len(OPERATOR_NOT)])
            index += len(OPERATOR_NOT)
        # if the extracted portion of the query is a OR operator,
        # add it as a token to the token list
        # and advance the index by the length of the operator.
        elif query[index:index + len(OPERATOR_OR)] == OPERATOR_OR:
            index += len(OPERATOR_OR)
            tokens.append(OPERATOR_OR)
        elif query[index] == '(' or query[index] == ')':
            tokens.append(query[index])
            index += 1
        # skip whitespace
        elif query[index] == ' ':
            index += 1
        # since the current character is neither an operator nor a parenthesis,
        # extract the term, normalize it (stem and case fold) and add to the token list.
        # finally, advance the index by the length of the term.
        else:
            endIndex = index
            while endIndex < len(query) and query[endIndex] != ' ' and query[endIndex] != '(' and query[endIndex] != ')':
                endIndex += 1
            token = query[index:endIndex]
            stemmedToken = ps.stem(token.strip()).lower()
            index += len(token)
            tokens.append(stemmedToken)
    return tokens

def isOperator(token):
    if token == OPERATOR_AND or token == OPERATOR_NOT or token == OPERATOR_OR:
        return True
    return False

def getOperatorPrecendence(op):
    if op == OPERATOR_NOT:
        return 3
    elif op == OPERATOR_AND:
        return 2
    elif op == OPERATOR_OR:
        return 1

def isHigherPrecedence(op1, op2):
    return getOperatorPrecendence(op2) > getOperatorPrecendence(op1)

def isTerm(token):
    """
    Returns True if the token is a term i.e. it is neither an operator nor a parenthesis
    Else returns False

    Parameters
    ----------
    token : string
        The token to check whether it is a term or not.
    """
    if not isOperator(token) and token != '(' and token != ')':
        return True
    return False

def evaluateQuery(postFixQuery, docIDs, dictionaryTerms, postingFile):
    """
    Returns the result of the evaluation of the Boolean query if it is a valid query.
    Else return "" for invalid query.

    Parameters
    ----------
    postFixQuery : list of string
        The postfix form of the Boolean query.
    docIDs : list of int
        The list of document ids in the dataset.
    dictionaryTerms: dictionary
        A dictionary that contains all the indexed terms in the dataset.
    postingFile: file
        A file that consists the indexed postings in the dataset.
    """
    operands = Stack()
    noOfSamePrecendenceOperators = 0
    currentOperator = ""
    for i in range(len(postFixQuery)):
        elem = postFixQuery[i]
        if isTerm(elem):
            operands.push(getPostingListOfTerm(elem, dictionaryTerms, postingFile))
            continue
        
        if elem == OPERATOR_NOT:
            operands.push(applyNot(docIDs, operands.pop()[1]))
        # else it is an operator other than NOT i.e. AND or OR
        else:
            # if there is no set of operator being keep tracked yet,
            # update the current operator
            # and set the no. of same precedence operators as 1
            if currentOperator == "":
                currentOperator = elem
                noOfSamePrecendenceOperators = 1
            # else if the current operator is the same as the operator being keep tracked,
            # increment the no. of same precedence operators
            elif currentOperator == elem:
                noOfSamePrecendenceOperators += 1     

            # if the current element is the last element,
            # or if the next element is a term or a different operator,
            # set isEvaluate to True
            isEvaluate = False
            if i == len(postFixQuery) - 1 \
                or (isTerm(postFixQuery[i + 1]) or (not isTerm(postFixQuery[i + 1]) and postFixQuery[i + 1] != currentOperator)):
                isEvaluate = True
            
            # if isEvaluate is True, evaluate the subquery that is stored so far
            # with the operator that has been kept tracked of.
            if isEvaluate:
                # get all operands involved with this operator
                operandsInvolved = []
                for j in range(noOfSamePrecendenceOperators + 1):
                    # if there are not enough operands to be operated on, then this is an invalid query
                    if operands.isEmpty():
                        return ""
                    operand = operands.pop()
                    heappush(operandsInvolved, operand)
                # continue evaluating the operands with the operator
                # until there is only 1 result left
                while len(operandsInvolved) > 1:
                    if currentOperator == OPERATOR_AND:
                        heappush(operandsInvolved, applyAnd(
                            heappop(operandsInvolved)[1], heappop(operandsInvolved)[1]))
                    elif currentOperator == OPERATOR_OR:
                        heappush(operandsInvolved, applyOr(
                            heappop(operandsInvolved)[1], heappop(operandsInvolved)[1]))
                # add the result to the top of the operands stack
                operands.push(operandsInvolved[0])
                # reset the operator being kept tracked of.
                currentOperator = ""
                noOfSamePrecendenceOperators = 0
    # the query is valid, if there is only 1 result left
    if operands.size() == 1:
        return operands.pop()[1][1]
    # else it is an invalid query
    return ""

def getPostingListOfTerm(term, dictionaryTerms, postingFile):
    """
    Returns the detailed posting list of the specified term in a form of a nested array,
    i.e., [the document frequencies of the specified term, the posting list of the term 
    (an array where the 1st element is the skip pointer distance and the 2nd element is an array of the docIDs that contains the specified term)]

    Parameters
    ----------
    term : string
        The term that is to be retrieved from the dictionary.
    dictionaryTerms: dictionary
        A dictionary that contains all the indexed terms in the dataset.
    postingFile: file
        A file that consists the indexed postings in the dataset.
    """
    termDict = dictionaryTerms.get(term)
    # if the specified term does not exist,
    # return an empty posting list
    if termDict == None:
        return [0, [0, []]]
    postingFile.seek(termDict["pointer"])
    postingList = pickle.load(postingFile)
    return [termDict["documentFrequencies"], postingList]

def applyNot(docIDs, pl):
    """
    Returns the result (in the form of detailed posting list) of applying NOT operator on the given posting list.

    Parameters
    ----------
    docIDs : list of int
        The list of document ids in the dataset.
    pl: a nested list
        The posting list of the operand (a term/result of previous Boolean operation).
    """
    postingsSet = set(pl[1])
    result = []
    # finds all document id that does not exist in the given posting list.
    for docID in docIDs:
        if docID not in postingsSet:
            result.append(docID)
    return [len(result), [calculateSkipPointerValue(result), result]]

def applyAnd(pl1, pl2):
    """
    Returns the result (in the form of detailed posting list) of applying AND operator on the given posting list.

    Parameters
    ----------
    pl1 : a nested list
        The posting list of the 1st operand (a term/result of previous Boolean operation).
    pl: a nested list
        The posting list of the 2nd operand (a term/result of previous Boolean operation).
    """
    intersection = []
    pl1SkipPointerPeriod = pl1[0]
    pl1Postings = pl1[1]
    pl2SkipPointerPeriod = pl2[0]
    pl2Postings = pl2[1]
    pl1PointerIndex = 0
    pl2PointerIndex = 0
    # gets the intersection set of the 2 posting list 
    # for optimization, skip pointers are utilized if the current pointer in the posting
    # has a skip pointer and the value to skipped to is lesser than or equal to the value of the other posting
    while pl1PointerIndex != len(pl1Postings) and pl2PointerIndex != len(pl2Postings):
        if pl1Postings[pl1PointerIndex] == pl2Postings[pl2PointerIndex]:
            intersection.append(pl1Postings[pl1PointerIndex])
            pl1PointerIndex += 1
            pl2PointerIndex += 1
        elif pl1Postings[pl1PointerIndex] < pl2Postings[pl2PointerIndex]:
            if hasSkip(pl1PointerIndex, pl1SkipPointerPeriod, len(pl1Postings)) and \
                    pl1Postings[skip(pl1PointerIndex, pl1SkipPointerPeriod)] <= pl2Postings[pl2PointerIndex]:
                while hasSkip(pl1PointerIndex, pl1SkipPointerPeriod, len(pl1Postings)) and \
                        pl1Postings[skip(pl1PointerIndex, pl1SkipPointerPeriod)] <= pl2Postings[pl2PointerIndex]:
                    pl1PointerIndex = skip(
                        pl1PointerIndex, pl1SkipPointerPeriod)
            else:
                pl1PointerIndex += 1
        elif hasSkip(pl2PointerIndex, pl2SkipPointerPeriod, len(pl2Postings)) and \
                pl2Postings[skip(pl2PointerIndex, pl2SkipPointerPeriod)] <= pl1Postings[pl1PointerIndex]:
            while hasSkip(pl2PointerIndex, pl2SkipPointerPeriod, len(pl2Postings)) and \
                    pl2Postings[skip(pl2PointerIndex, pl2SkipPointerPeriod)] <= pl1Postings[pl1PointerIndex]:
                pl2PointerIndex = skip(pl2PointerIndex, pl2SkipPointerPeriod)
        else:
            pl2PointerIndex += 1
    return [len(intersection), [calculateSkipPointerValue(intersection), intersection]]

def applyOr(pl1, pl2):
    """
    Returns the result (in the form of detailed posting list) of applying OR operator on the given posting list.

    Parameters
    ----------
    pl1 : a nested list
        The posting list of the 1st operand (a term/result of previous Boolean operation).
    pl: a nested list
        The posting list of the 2nd operand (a term/result of previous Boolean operation).
    """
    pl1Postings = pl1[1]
    pl2Postings = pl2[1]
    combinedPosting = []
    pl1PointerIndex = 0
    pl2PointerIndex = 0
    # accumulates a list of document id that appears in either the 1st or 2nd posting list until the end of either posting list.
    # removes duplicated document id as well
    while pl1PointerIndex != len(pl1Postings) and pl2PointerIndex != len(pl2Postings):
        if pl1Postings[pl1PointerIndex] < pl2Postings[pl2PointerIndex]:
            combinedPosting.append(pl1Postings[pl1PointerIndex])
            pl1PointerIndex += 1
        elif pl2Postings[pl2PointerIndex] < pl1Postings[pl1PointerIndex]:
            combinedPosting.append(pl2Postings[pl2PointerIndex])
            pl2PointerIndex += 1
        else:
            combinedPosting.append(pl1Postings[pl1PointerIndex])
            pl1PointerIndex += 1
            pl2PointerIndex += 1

    # add all the remaining docIDs
    while pl1PointerIndex != len(pl1Postings):
        combinedPosting.append(pl1Postings[pl1PointerIndex])
        pl1PointerIndex += 1

    while pl2PointerIndex != len(pl2Postings):
        combinedPosting.append(pl2Postings[pl2PointerIndex])
        pl2PointerIndex += 1

    return [len(combinedPosting), [calculateSkipPointerValue(combinedPosting), combinedPosting]]

def hasSkip(pointerIndex, skipPointerValue, postingsSize):
    """
    Returns True if the current pointer has a skip pointer and the skip pointer points to a valid element in the postings.
    Else returns False.

    Parameters
    ----------
    pointerIndex : int
        The current index in the posting list that the pointer is pointing to.
    skipPointerValue: int
        The skip pointer distance in the posting list.
    postingsSize: int
        The size of the posting list.
    """
    return pointerIndex % skipPointerValue == 0 and (pointerIndex + skipPointerValue) <= postingsSize - 1

def skip(pointerIndex, skipPointerValue):
    return pointerIndex + skipPointerValue

def calculateSkipPointerValue(postings):
    return math.floor(math.sqrt(len(postings)))


dictionary_file = postings_file = file_of_queries = output_file_of_results = None


try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
