#!/usr/bin/python3
import csv
import filecmp
from pydoc import doc
import re
import nltk
import os
import sys
import getopt
import math
import json
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
from nltk import stem
from string import punctuation
import math


"""
HOMEWORK 4 : normal dict
Index without number ?
special character error : UnicodeEncodeError: 'charmap' codec can't encode character '\u2033' in position 6: character maps to <undefined>
"""

csv.field_size_limit(sys.maxsize)

STEMMER = stem.PorterStemmer()
BIGRAMS = False
UNIGRAMS = True
TRIGRAMS = False


# === Useful functions ===
def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def sortDict(_dict):
    keys = list(_dict.keys())
    keys.sort()
    res = {}

    for key in keys:
        res[key] = _dict[key]

    return res

def sortPosting(post,_dict):
    res = {}
    for postKey in list(_dict.values()):
        res[str(postKey)] = post[postKey]

    return res

#temporary function for debug -> remove before submitting
def printDico(dico,_iter):
    print("..print")
    idx = 0
    iter = _iter + 15210
    while idx < iter:
        for key,val in dico.items():
            if idx > 15210:
                if idx > iter:
                    break
                #print("{} => {}\n=====\n".format(key,val))
            idx +=1


# === Writting functions ===
def writeDict(_dict,postL):
    # Write a dictionary onto harddsik during the "build index" part.
    # :param idx: index in the name of the file
    # :param _dict: dictionary
    # :param postL: posting list {postingListID1: {docID1: (termFrequency, weight), docID2: ...} ...}
    # :return: void
    # print(_dict)
    # print(postL)
    # offset = 0
    offset = 1
    with open("dictionary.txt", "w", encoding="utf-8") as f:
        for key,val in _dict.items():
            sorted_docIDS = [int(elt) for elt in list(postL[str(val)].keys())]
            sorted_docIDS.sort()

#            sorted_docIDS_5dig = ["{}{}".format( ("0000"+str(elt))[-5:], (str(postL[str(val)][(elt)][1]))[:5] ) for elt in sorted_docIDS]
            sorted_docIDS_5dig = ["{}_{}".format( ("0000"+str(elt)), str(postL[str(val)][str(elt)][1])[:5] ) for elt in sorted_docIDS] # temporary
            all_docIDS = " ".join( sorted_docIDS_5dig )

            post_line = "{}\n".format(all_docIDS)
            new_line = "{} {} {}\n".format(key,len( postL[str(val)] ),offset)
            f.write(new_line)

            # offset += len(post_line)+1
            offset += 1

def writePosting(post,encoding="utf-8"):
    # Write a posting list onto harddisk during the "build index" part.
    # :param idx: index in the name of the file
    # :param post: posting list {postingListID1: {docID1: (termFrequency, weight), docID2: ...} ...}
    # :return: void
    offset = 0
    with open("postings.txt", "w") as f:
        for postID,docIDS in post.items():
            sorted_postings_list = sorted(docIDS.keys())
            sorted_docIDS_5dig = ["{}_{}".format(("0000" + str(elt)), (str(docIDS[(elt)][1]))[:5]) for elt in
                                  sorted_postings_list]
            # sorted_posting = sorted(docIDS.items(), key=lambda x: x[1][1], reverse=True) # sort according to weights *Heur3*
            # sorted_docIDS = [int(elt[0]) for elt in sorted_posting]
            #sorted_docIDS = [int(elt) for elt in list(docIDS.keys())]
#            sorted_docIDS_5dig = ["{}{}".format( ("0000"+str(elt))[-5:], (str(docIDS[(elt)][1]))[:5] ) for elt in sorted_docIDS] #with term weights
#             sorted_docIDS_5dig = ["{}_{}".format( ("0000"+str(elt))[:], (str(docIDS[(elt)][1]))[:5] ) for elt in sorted_docIDS] # temporary
            all_docIDS = " ".join( sorted_docIDS_5dig )
           
            new_line = "{}\n".format(all_docIDS)
            f.write(new_line)
            offset += len(new_line)+1


# === Index building functions ===

def computeWeights(postingLists, N):
    # Compute idf.tf for each document of all the posting lists
    # :param postingLists: posting lists, format : {postingListID1: {docID1: termFrequency, docID2: ...} ...}
    # :param N: size of the collection
    # :return: posting lists, format : {postingListID1: {docID1: (termFrequency, weight), docID2: ...} ...}

    documents_length = {}
    print("..computing weights : N = {}".format(N))
    for pL_Id,docs in postingLists.items():
        for docID, termFreq in docs.items():
            weight = (1+math.log(int(termFreq), 10))
            postingLists[pL_Id][docID] = (termFreq,weight)

            # Update document length
            try : 
                documents_length[docID] += weight**2
            except :
                documents_length[docID] = weight**2

    # Export the length for normalization
    with open("documents_length.txt", "w") as f:
        for docID, length in documents_length.items():
            f.write("{} {}\n".format(docID,math.sqrt(length)))

    return postingLists

    

def build_index(in_dir, out_dict, out_postings,path_data):
    # Build index from documents stored in the input directory,
    # then output the dictionary file and postings file
    
    print('indexing...')

    columns_to_index = {1,2,3} # Columns : "document_id","title","content","date_posted","court"
    #data = pd.read_csv(path_data).head() # Get the data in a dataframe
    # print("Data length : {}".format(len(data)))

    data = []
    with open(in_dir) as csvfile:
        data_raw = csv.reader(csvfile)
        for idx, row in enumerate(data_raw):
            if idx == 0:
                continue
            data.append(row)

    #Init
    dictionary = {} # Format : {"token": {title : postingListID, content: postingListID} ..}
    postingList = {} # Format : {postingListID1: { docID1: termFrequency,  docID2: termFrequency }, postingListID2: ... } 
    index = -1
    months_correspondence = {"01":"Jan", "02":"Feb", "03":"Mar", "04":"Apr", "05":"May", "06":"Jun", "07":"Jul", "08":"Aug", "09":"Sep", "10":"Oct", "11":"Nov", "12":"Dec"}

    # We are going through all the documents
    for row in data:
        docID = row[0]
        print(docID)
        index +=1

        for col in columns_to_index :
            date_col = False
            line = row[col]
            # == PREPROCESS STUFF ==
            final_tokens = {}
            stemmed_tokens_without_punct = []
            if col == 3 : #"date_posted": # dates are not processed as the other columns
                date_col = True
                yy,mm,dd = line.split()[0].split("-")

                # Change the date format (which one ?)
                # date = "{} {} {}".format(int(dd),months_correspondence[mm],yy) # 21 May 2020
                date = "{}-{}-{}".format(int(dd),int(mm),yy) # 21-5-2020
                # date = "{}-{}-{}".format(int(mm),int(dd),yy) # 5-21-2020
                # date = "{}-{}-{}".format(yy,int(mm),int(dd)) # 2020-5-21

                final_tokens["unigrams"] = [ date ]
            else:
                #Tokenization
                for word in word_tokenize(line) : # "Is U.S. big?" --> ["Is", "U.S.", "big?"] 

                    # We only index number that are in the title
                    if col == 1 : # "title":
                        #Stemm
                        stemmed_token = (STEMMER.stem(word)) # are -> be
                        
                        #Remove punctuations
                        stemmed_tokens_without_punct += stemmed_token.strip(punctuation).split(" ")
                    else:
                        # Numbers are not indexed
                        try:
                            float(word)
                        except:
                            #Stemm
                            stemmed_token = (STEMMER.stem(word)) # are -> be
                            
                            #Remove punctuations
                            stemmed_tokens_without_punct += stemmed_token.strip(punctuation).split(" ")
                        
                
                # finally  -> ["be", "u.s", "big"]

                if UNIGRAMS :
                    final_tokens["unigrams"] = stemmed_tokens_without_punct #get the bigrams if BIGRAMS
                if BIGRAMS:
                    final_tokens["bigrams"] = list(bigrams(stemmed_tokens_without_punct))
                if TRIGRAMS:
                    final_tokens["trigrams"] = list(trigrams(stemmed_tokens_without_punct))

            # == Build dictionary and postings ==
            for key, tokens in final_tokens.items() :
                for _token in tokens :
                    token = _token if (date_col or key == "unigrams") else " ".join(_token) # uncomment if using bigrams
                    

                    if token != "":
                        #Is the token in the dictionary ? 
                        try:
                            postingListID = dictionary[token]
                            
                            #We add the current docID to the posting list if it is not in yet
                            try:
                                postingList[postingListID][docID] += 1
                            except:
                                postingList[postingListID][docID] = 1
                        except:
                            dictionary[token] = list(dictionary.values())[-1]+1 if len(dictionary.values()) != 0 else 1
                            postingList[dictionary[token]] = {}
                            postingList[dictionary[token]][docID] = 1

    # Write the current dictionary
    if len(dictionary) != 0:
        print("sort dico")
        dictionary = sortDict(dictionary)
        postingList = sortPosting(postingList,dictionary)
        postingList = computeWeights(postingList, len( data ) )

        writePosting(postingList)
        writeDict(dictionary,postingList)
        #printDico(postingList,3)
        dictionary = {}
        postingList = {}

    print("end indexing...")
    return 1 #change




if __name__ == "__main__":

    # === INPUT PROCESS ===

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



    # === INDEX CONSTRUCTION ===
    try:
        os.remove(output_file_dictionary)
        os.remove(output_file_postings)
        print("Former dictionary and posting list deleted.")
    except FileNotFoundError:
        pass

    # Build index -> several dict and posting lists
    current_index = build_index(input_directory, output_file_dictionary, output_file_postings, input_directory)

    