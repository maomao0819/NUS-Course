#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import pickle
import os
import math

operators_info  = {
    # operator: (precedence, associativity)
    'NOT': (2, 'L'), 
    'AND': (1, 'L'),
    'OR': (0, 'L')
}

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below

    # read the data
    dictFile = open(dict_file, "rb")
    postingFile = open(postings_file, "rb")
    dictUnpickler = pickle.Unpickler(dictFile)
    postingsUnpickler = pickle.Unpickler(postingFile)
    dictionaryTerms = {}
    try:
        while True:
            termDict = dictUnpickler.load()
            dictionaryTerms[termDict["term"]] = termDict
            del dictionaryTerms[termDict["term"]]["term"]
    except EOFError:
        dictFile.close()
    
    clearFile(results_file)
    # docIDs = os.listdir('../../nltk_data/corpora/reuters/training/')
    docIDs = postingsUnpickler.load()
    docIDs = list(map(int, docIDs))
    docIDs.sort()

    # print(dictionaryTerms)
    queries = loadFileToString(queries_file)
    # print('queries', queries)
    # process query
    for i in range(len(queries)):
        if '\n' in queries[i]:
            queries[i] = queries[i][:-1]
        queries[i] = Tokenize(queries[i])
        queries[i] = ShuntYard(queries[i])

    universe = docIDs

    for query in queries:
        query_result = []
        while len(query):
            current_token = query.pop(0)
            # print('current_token', current_token)
            if current_token in operators_info.keys():
                if len(query_result) < 2:
                    print('The query has an error!')
                else:
                    # NOT
                    if current_token == 'NOT':
                        query_result.append(NotqueriesSkip(query_result.pop(-1), universe))
                    # AND, OR
                    else:
                        list2 = query_result.pop(-1)
                        list1 = query_result.pop(-1)
                        if current_token == 'AND':
                            query_result.append(ANDqueriesSkip(list1, list2))
                        elif current_token == 'OR':
                            query_result.append(ORqueriesSkip(list1, list2))
            # Not a operator
            else:
                # data = FindTermData(dict_file, current_token)
                data = dictionaryTerms.get(current_token) 
                if data == None:
                    postings = [-99]
                else:
                    postingFile.seek(data["pointer"])
                    postings = pickle.load(postingFile)[-1]
                    # print(postings)
                query_result.append(postings)
            # print('query_result', query_result)
        query_result = query_result[0]
        if -99 in query_result:
            query_result.remove(-99)
        
        outputToFiles(query_result, results_file)

# def FindTermData(dict_file, term):
#     dictFile = open(dict_file, "rb")
#     postingFile = open(postings_file, "rb")
#     dictUnpickler = pickle.Unpickler(dictFile)
#     for i in range(34148):
#         data = dictUnpickler.load()
#         if data['term'] == term:
#             return data

def Tokenize(Query):
    # split with space
    QueryList = Query.split()
    NewQueryList = []
    # split with bracket
    for i in range(len(QueryList)):
        current_token = QueryList.pop(0)
        # left bracket
        if current_token[0] == '(':
            NewQueryList.append('(')
            NewQueryList.append(current_token[1:])
        # right bracket
        elif current_token[-1] == ')':
            NewQueryList.append(current_token[:-1])
            NewQueryList.append(')')
        else:
            NewQueryList.append(current_token)
    return NewQueryList

def ShuntYard(tokens):
    tokens += ['end']
    operators = []
    output = []

    while len(tokens) != 1:
        current_token = tokens.pop(0)
        # Is a operator
        if current_token in operators_info.keys():
            while True:
                if len(operators) == 0:
                    break
                satisfied = False
                if operators[-1] not in ["(", ")"]:
                    # operator at top has greater precedence
                    if operators_info[operators[-1]][0] > operators_info[current_token][0]:
                        satisfied = True
                    # equal precedence and has left associativity
                    elif operators_info[operators[-1]][0] == operators_info[current_token][0]:
                        if operators_info[operators[-1]][1] == 'L':
                            satisfied = True
                satisfied = satisfied and operators[-1] != "("
                if not satisfied:
                    break
                output.append(operators.pop())
            operators.append(current_token)
        # Is a left bracket
        elif current_token == "(":
            operators.append(current_token)
        # Is a right bracket
        elif current_token == ")":
            while True:
                if len(operators) == 0:
                    break
                if operators[-1] == "(":
                    break
                output.append(operators.pop())
            if len(operators) != 0 and operators[-1] == "(":
                operators.pop()
        # Not a operator
        else:
            output.append(current_token) 
    output.extend(operators[::-1])
    return output

def NotqueriesSkip(List, Universe, skipPointer=10):
    # return list(set(universe) - set(list))
    id_list = 0
    id_universe = 0
    skipPointer = min(int(math.floor(math.sqrt(len(Universe)))), 2)
    output = []
    while id_list < len(List):
        if List[id_list] == Universe[id_universe]:
            id_list += 1
            id_universe += 1
        elif List[id_list] < Universe[id_universe]:
            id_list += 1
        elif List[id_list] > Universe[id_universe]:
            if id_universe % skipPointer == 0:
                further_id_universe = id_universe + skipPointer
                while further_id_universe < len(Universe) and List[id_list] > Universe[further_id_universe]:
                    further_id_universe += skipPointer
                further_id_universe -= skipPointer
                output.extend(Universe[id_universe:further_id_universe+1])
                id_universe = further_id_universe
            else:
                output.append(Universe[id_universe])
            id_universe += 1
    if id_universe < len(Universe):
        output += Universe[id_universe:]
    return output

def ANDqueriesSkip(list1, list2, skipPointer1=2, skipPointer2=2):
    # return list(set(list1) & set(list2))
    id1 = 0
    id2 = 0
    skipPointer1 = min(int(math.floor(math.sqrt(len(list1)))), 2)
    skipPointer2 = min(int(math.floor(math.sqrt(len(list2)))), 2)
    output = []
    while id1 < len(list1) and id2 < len(list2):
        if list1[id1] == list2[id2]:
            output.append(list1[id1])
            id1 += 1
            id2 += 1
        elif list1[id1] < list2[id2]:
            if id1 % skipPointer1 == 0:
                further_id1 = id1 + skipPointer1
                while further_id1 < len(list1) and list1[further_id1] < list2[id2]:
                    id1 += skipPointer1
                    further_id1 += skipPointer1
            id1 += 1
        elif list1[id1] > list2[id2]:
            if id2 % skipPointer2 == 0:
                further_id2 = id2 + skipPointer2
                while further_id2 < len(list2) and list1[id1] > list2[further_id2]:
                    id2 += skipPointer2
                    further_id2 += skipPointer2
            id2 += 1
    return output

def ORqueriesSkip(list1, list2, skipPointer1=2, skipPointer2=2):
    # return list(set(list1) | set(list2))
    id1 = 0
    id2 = 0
    skipPointer1 = min(int(math.floor(math.sqrt(len(list1)))), 2)
    skipPointer2 = min(int(math.floor(math.sqrt(len(list2)))), 2)
    output = []
    while id1 < len(list1) and id2 < len(list2):
        if list1[id1] == list2[id2]:
            output.append(list1[id1])
            id1 += 1
            id2 += 1
        elif list1[id1] < list2[id2]:
            if id1 % skipPointer1 == 0:
                further_id1 = id1 + skipPointer1
                while further_id1 < len(list1) and list1[further_id1] < list2[id2]:
                    further_id1 += skipPointer1
                further_id1 -= skipPointer1
                output.extend(list1[id1:further_id1+1])
                id1 = further_id1
            else:
                output.append(list1[id1])
            id1 += 1
        elif list1[id1] > list2[id2]:
            if id2 % skipPointer2 == 0:
                further_id2 = id2 + skipPointer2
                while further_id2 < len(list2) and list1[id1] > list2[further_id2]:
                    further_id2 += skipPointer2
                further_id2 -= skipPointer2
                output.extend(list2[id2:further_id2+1])
                id2 = further_id2
            else:
                output.append(list2[id2])
            id2 += 1
    if id1 < len(list1):
        output += list1[id1:]
    if id2 < len(list2):
        output += list2[id2:]
    return output

def clearFile(filename):
    file = open(filename, 'w')
    file.close()

def loadFileToString(filename):
    f = open(filename,'r')
    content = f.readlines()
    f.close()
    return content

def outputToFiles(result, filefullname, writeWithString=True):
    filename, file_extension = os.path.splitext(filefullname)
    if file_extension == '':
        filefullname += '.txt'
    result.sort()
    outputFile = open(filefullname, 'a')
    if writeWithString:
        output = ''
        for i in result:
            output += str(i)
            output += ', '
        output = output[:-2] + '\n'
        outputFile.write(output)
    else:
        resultPickler = pickle.Pickler(outputFile)
        resultPickler.dump(result)
    return

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
