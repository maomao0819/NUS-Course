#!/usr/bin/python3
import nltk
import sys
import getopt
import os
import string
from nltk.stem import PorterStemmer
import math
import pickle

ps = PorterStemmer()
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')
    # This is an empty method
    # Pls implement your code in below

    # gets all the files' name in the input directory
    docIDs = os.listdir(in_dir)
    docIDs.sort()

    sizeOfTrainingDocuments = len(docIDs) # should be len(docIDs)
    memoryLimit = 10000 # limited based on no. of pairs 
    noOfDocumentsProcessedFully = 0
    blockNos = ["1"]
    # noOfPairs = 0
    # index 0: blockNo, index 1: isProcessedFinishDoc, index 2: isFirstTimeProcessingDoc
    # index 3: currentDocID, index 4: currentDocIDIndex, index 5: noOfPairs,
    # index 6: unprocessedWords, index 7: indexStopAtForUpw (index stop at for unprocessed words)
    # index 8: dictionary (stores the terms, their document frequencies and pointer (index in postings list) to postings list),
    # index 9: postingsList (a nested array where each array of postings consists of the document ids that a term appears in)
    # index 10: seen (an array of sets to contain the docIDs that appeared in each posting list)
    processingResult = [1, False, True, docIDs[0], 0, 0, 0, [], {}, [], []]
    while noOfDocumentsProcessedFully != sizeOfTrainingDocuments:
        if noOfDocumentsProcessedFully % 100 == 0:
            print("no. of documents processed: " + str(noOfDocumentsProcessedFully))
        if processingResult[1] and noOfDocumentsProcessedFully + 1 == sizeOfTrainingDocuments:
            noOfDocumentsProcessedFully += 1
            outputToFiles(processingResult)
            break
        elif processingResult[1]:
            noOfDocumentsProcessedFully += 1
            processingResult[4] += 1
            processingResult[3] = docIDs[processingResult[4]]
            processingResult[1] = False
            processingResult[2] = True
        
        if len(processingResult[8]) == memoryLimit:
            outputToFiles(processingResult)
            processingResult[0] += 1
            blockNos.append(str(processingResult[0]))
            processingResult[8] = {}
            processingResult[9] = []
            processingResult[10] = []
        
        processingResult = processFile(processingResult, in_dir, memoryLimit)
    mergeBlocks(blockNos, 1, out_dict, out_postings)


# index 0: blockNo, index 1: isProcessedFinishDoc, index 2: isFirstTimeProcessingDoc
# index 3: currentDocID, index 4: currentDocIDIndex, index 5: noOfPairs,
# index 6: unprocessedWords, index 7: indexStopAtForUpw (index stop at for unprocessed words)
# index 8: dictionary (stores the terms, their document frequencies and pointer (index in postings list) to postings list),
# index 9: postingsList (a nested array where each array of postings consists of the document ids that a term appears in)
# index 10: seen (an array of sets to contain the docIDs that appeared in each posting list)
def processFile(processingResult, directoryFilePath, memoryLimit):
    if processingResult[2]:
        filepath = os.path.join(directoryFilePath, processingResult[3])
        content = ""
        with open(filepath) as f:
            content = f.read()
        sentences = nltk.sent_tokenize(content)
        wordsUnprocessed = []
        for sentence in sentences:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            words = nltk.word_tokenize(sentence)
            for word in words:
                stemmedWord = ps.stem(word.strip()).lower()
                if len(processingResult[8]) == memoryLimit:
                    wordsUnprocessed.append(word)
                else:
                    processTerm(processingResult, stemmedWord)
        processingResult[1] = len(wordsUnprocessed) == 0
        if not processingResult[1]:
            processingResult[2] = False
        processingResult[6] = wordsUnprocessed
        processingResult[7] = 0
    else:
        # else process the unprocessed stemmed words
        for i in range(processingResult[7], len(processingResult[6])):
            if len(processingResult[8]) == memoryLimit:
                # processingResult[6] = processingResult[6][i:]
                processingResult[7] = i
                break
            else:
                processTerm(processingResult, processingResult[6][i])
                if i == len(processingResult[6]) - 1:
                    processingResult[1] = True
                    processingResult[6] = []
                    processingResult[7] = 0
    return processingResult


def processTerm(processingResult, stemmedWord):
    termDetails = processingResult[8].get(stemmedWord)
    if termDetails == None:
        postings = [int(processingResult[3])]
        seen = set(postings)
        processingResult[9].append(postings)
        processingResult[10].append(seen)
        processingResult[8][stemmedWord] = {"documentFrequencies" : len(postings), "pointer": len(processingResult[9]) - 1}
    else:
        if int(processingResult[3]) not in processingResult[10][termDetails["pointer"]]:
            processingResult[9][termDetails["pointer"]].append(int(processingResult[3]))
            processingResult[10][termDetails["pointer"]].add(int(processingResult[3]))
            processingResult[8][stemmedWord]["documentFrequencies"] += 1


def outputToFiles(processingResult):
    postingFile = open("postings" + "_block" + str(processingResult[0]) + ".txt", "wb")
    postingsPickler = pickle.Pickler(postingFile)
    for p in processingResult[9]:
        byteOffset = postingFile.tell()
        postingsPickler.dump(p)
        p.append(byteOffset)
    postingFile.close()

    dictFile = open("dictionary" + "_block" + str(processingResult[0]) + ".txt", "wb")
    dictPickler = pickle.Pickler(dictFile)
    for k in sorted(processingResult[8]):
        # get its posting list
        postings = processingResult[9][processingResult[8][k]['pointer']] 
        processingResult[8][k]['pointer'] = postings[len(postings) - 1]
        processingResult[8][k]['term'] = k
        dictPickler.dump(processingResult[8][k])
    dictFile.close()

def mergeBlocks(blocks, iterationNo, out_dict, out_postings):
    if len(blocks) > 1:
        mid = len(blocks) // 2
        left = blocks[:mid]
        right = blocks[mid:]
        leftMerged = mergeBlocks(left, iterationNo + 1, out_dict, out_postings)
        rightMerged = mergeBlocks(right, iterationNo + 1, out_dict, out_postings)
        return merge(leftMerged, rightMerged, iterationNo, out_dict, out_postings)
    else:
        return blocks[0]

def merge(left, right, iterationNo, out_dict, out_postings):
    lDictFile = open("dictionary" + "_block" + str(left) + ".txt", "rb")
    lPostingFile = open("postings" + "_block" + str(left) + ".txt", "rb")
    lDictUnpickler = pickle.Unpickler(lDictFile)

    rDictFile = open("dictionary" + "_block" + str(right) + ".txt", "rb")
    rPostingFile = open("postings" + "_block" + str(right) + ".txt", "rb")
    rDictUnpickler = pickle.Unpickler(rDictFile)

    isReadFinishLeftDictFile = False 
    isReadFinishRightDictFile = False
    mergedFileName = "_merged" + str(left) + str(right)
    mergedDictFilePath = "dictionary" + "_block" + mergedFileName + ".txt"
    mergedPostingFilePath = "postings" + "_block" + mergedFileName + ".txt"
    # last merging
    if iterationNo == 1:
        mergedDictFilePath = out_dict
        mergedPostingFilePath = out_postings
    mergedDictFile = open(mergedDictFilePath, "wb")
    mergedPostingFile = open(mergedPostingFilePath, "wb")

    mergedDictPickler = pickle.Pickler(mergedDictFile)
    mergedPostingsPickler = pickle.Pickler(mergedPostingFile)

    isLoadNextTermOfLeftDictFile = True
    isLoadNextTermOfRightDictFile = True
    leftTermDict = {}
    rightTermDict = {}
    while not isReadFinishLeftDictFile and not isReadFinishRightDictFile:
        if isLoadNextTermOfLeftDictFile:
            try:
                leftTermDict = lDictUnpickler.load()
            except EOFError:
                isReadFinishLeftDictFile = True
                lDictFile.close()
                lPostingFile.close()
                break
        
        if isLoadNextTermOfRightDictFile:
            try:
                rightTermDict = rDictUnpickler.load()
            except EOFError:
                isReadFinishRightDictFile = True
                rDictFile.close()
                rPostingFile.close()
                break
        
        if leftTermDict["term"] == rightTermDict["term"]:
            # merge their postings and write it into the merged file
            # then load the new terms from both left and right dictionary
            lPostingFile.seek(leftTermDict["pointer"])
            leftTermPosting = pickle.load(lPostingFile)
            rPostingFile.seek(rightTermDict["pointer"])
            rightTermPosting = pickle.load(rPostingFile)

            leftTermPostingSet = set(leftTermPosting)
            rightTermPostingSet = set(rightTermPosting)
            docIDsNotInLeftTermPosting = list(rightTermPostingSet - leftTermPostingSet)
            combinedPosting = leftTermPosting + docIDsNotInLeftTermPosting
            leftTermDict["pointer"] = mergedPostingFile.tell()
            leftTermDict["documentFrequencies"] = len(combinedPosting)
            if iterationNo == 1:
                mergedPostingsPickler.dump([math.floor(math.sqrt(len(combinedPosting))), combinedPosting])
            else:
                mergedPostingsPickler.dump(combinedPosting)
            mergedDictPickler.dump(leftTermDict)

            isLoadNextTermOfLeftDictFile = True
            isLoadNextTermOfRightDictFile = True
        elif rightTermDict["term"] < leftTermDict["term"]:
            # won't find current right block's dictionary term in the left block's dictionary
            # so can just write out the right term into the merged file
            # and load new term from right block's dictionary
            writeToMergedFile(rPostingFile, rightTermDict, mergedPostingsPickler, mergedDictPickler, mergedPostingFile, iterationNo)
            isLoadNextTermOfRightDictFile = True
            isLoadNextTermOfLeftDictFile = False
        else:
            # won't find current left block's dictionary term in the right block's dictionary
            # so can just write out the left term into the merged file
            # and load new term from left block's dictionary
            writeToMergedFile(lPostingFile, leftTermDict, mergedPostingsPickler, mergedDictPickler, mergedPostingFile, iterationNo)
            isLoadNextTermOfLeftDictFile = True
            isLoadNextTermOfRightDictFile = False

    # write all the remaining terms in the left dictionary file (i.e., not mergable) to the merged file
    if not isReadFinishLeftDictFile and not isLoadNextTermOfLeftDictFile:
        writeToMergedFile(lPostingFile, leftTermDict, mergedPostingsPickler, mergedDictPickler, mergedPostingFile, iterationNo)
        isLoadNextTermOfLeftDictFile = True
    while not isReadFinishLeftDictFile:
        try:
            leftTermDict = lDictUnpickler.load()
        except EOFError:
            isReadFinishLeftDictFile = True
            lDictFile.close()
            lPostingFile.close()
            break
        writeToMergedFile(lPostingFile, leftTermDict, mergedPostingsPickler, mergedDictPickler, mergedPostingFile, iterationNo)
        
    # write all the remaining terms in the right dictionary file (i.e., not mergable) to the merged file
    if not isReadFinishRightDictFile and not isLoadNextTermOfRightDictFile:
        writeToMergedFile(rPostingFile, rightTermDict, mergedPostingsPickler, mergedDictPickler, mergedPostingFile, iterationNo)
        isLoadNextTermOfRightDictFile = True
    while not isReadFinishRightDictFile:
        try:
            rightTermDict = rDictUnpickler.load()
        except EOFError:
            isReadFinishRightDictFile = True
            rDictFile.close()
            rPostingFile.close()
            break
        writeToMergedFile(rPostingFile, rightTermDict, mergedPostingsPickler, mergedDictPickler, mergedPostingFile, iterationNo)
        
    return mergedFileName


def writeToMergedFile(postingFile, termDict, postingsPickler, dictPickler, mergedPostingFile, iterationNo):
    postingFile.seek(termDict["pointer"])
    posting = pickle.load(postingFile)
    termDict["pointer"] = mergedPostingFile.tell()
    if iterationNo == 1:
        postingsPickler.dump([math.floor(math.sqrt(len(posting))), posting])
    else:
        postingsPickler.dump(posting)
    dictPickler.dump(termDict)


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
