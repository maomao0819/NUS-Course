#!/usr/bin/python3
from pydoc import doc
import re
import os
import nltk
import sys
import getopt
import math
from math import sqrt, log, pow
from nltk import stem
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from string import punctuation
from collections import Counter
import linecache


STEMMER = stem.PorterStemmer()
DIGITS = 5
N = sum(1 for line in open('documents_length.txt'))
# Size of the collection
### return max 10 documents
K = 10
WEIGHT_QUERY_THRESHOLD = 0.35 # Used to select the most important query terms
WEIGHT_DOCUMENT_THRESHOLD = 1 # Used in optimization heuristic 3
HEURISTIC3 = False # By default, we don't use optimization heuristic (cf. README)

POS_SET = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "R": wordnet.ADV,
    "V": wordnet.VERB,
}

# == Normalization functions ==

def get_euclidean_doc_len(docID):
    """get the euclidean doc len"""

    with open("documents_length.txt","r") as f:
        while(l:=f.readline()) : 
            if int(l.split(" ")[0]) == docID :
                return float(l.split(" ")[1])
    return 0

def normalize_docScore(docID, score):
    # Normalize a document's score with the document's length we compute at the indexing time
    # :param docID: (int) document ID
    # :param score: (float) current document's score
    # :return: (float) document's normalized score, -1 if we didn't find the document length
    with open("documents_length.txt","r") as f:
        while(l:=f.readline()) :
            if int(l.split(" ")[0]) == docID :
                return score/float(l.split(" ")[1])
    return -1

def normalize(tf, li):
    # :param tf: is the number of a particular term
    # :param li: is the list of terms that contributes to the distance (considering only query terms here)
    if tf == 0:
        return 0
    
    deno = 0
    for i in li:
        deno += pow(i, 2)

    deno = sqrt(deno)
    return tf / deno

# =============

def retrieve_dict(filepath):
    # Populate hashtable for easy retrieval
    # :param filepath: (str) path of the dictionary file
    # :return: dictionary object -> format : {term1: (termFrequency, offset)...}

    dictionary = {}
    with open(filepath, "r") as f:
        ### read till eof
        while (line := f.readline()):
            ### term, docFreq, Offset in postingslist
            word, freq, offset = line.split(" ")
            dictionary[word] = (int(freq), int(offset))

    return dictionary

def update_documentvector(document_vectors, documents, token):
    # Update the weights of the document vectors when we get documents for a certain term 
    # :param document_vectors: Vector representations of all the documents -> format: {docId: {tk1: q1_weight, tk2: q2_weight, ...}, ...} qX_score representing the weight for the X-th term of the query
    # :param documents: Document's score [(docId1, score1), ...]
    # :param token: (str) Token we are updating
    for el in documents:
        document_vectors[el[0]][token] = el[1]

def compute_cosscore(document_vectors, query_dict):
    # :param document_vectors: Vector representations of all the documents -> format: {docId: {tk1: q1_weight, tk2: q2_weight, ...}, ...}
    # :param query_dict: Vector representation of the query -> format : {tk1: q1_weight, tk2: q2_weight, ...}, qX_weight representing the weight for the X-th term of the query

    cosscore = []
    for docid, vector_dict in document_vectors.items():
        score = 0
        for token in query_dict:
            score += query_dict[token] * vector_dict[token]

        if score != 0:
            cosscore.append((docid, normalize_docScore(docid,score))) # normalization with document length
    return cosscore

def get_documents(cosscores):
    # Keep the K better documents : those with the best cosscore.
    # :param cosscore: Scores for each documents, format : [(docId, cosscore), ...]
    # :return: K documents with the highest score

    sorted_cos_scores = sorted(cosscores.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_cos_scores)

    return [doc for doc, score in sorted_cos_scores]

def get_doc_id_and_weight(str):
    doc, weight = str.split('_')

    return int(doc)  # , float(weight)

def get_postings_list(token, dictionary, postings_file):
    tf, offset = dictionary.get(token, (False, False))

    if not offset:
        return []

    return list(map(get_doc_id_and_weight, linecache.getline(postings_file, offset).rstrip().split()))

def search_documents(token, dictionary, postings_file):
    # Get the document that contain a precise token
    # :param token: (str) Token we are processing
    # :param dictionary: Dictionary of the collection, format : {term1: (termFrequency, offset)...}
    # :param postings_file: (str) Path of the postings
    # :return: Set of documents that contain the token, format = [(docID,tf.idf) , ...]

    documents = [] 
    if token not in dictionary: ### if key does not exist
        return []
    
    with open(postings_file, "r") as f:
        f.seek(dictionary[token][1])
        line = f.readline().rstrip()
        line = line.split()[1:] # todo check why the offset is wrong damn bro is damn off HAHAHA

        # Only consider the documents with high enough weight *Heur3*
        documents = []
        for elt in line:
            # docID_weight = ( int(elt.split("_")[0]),float(elt.split("_")[1]) ) # (docID,tf.idf)
            docID_weight = get_doc_id_and_weight(elt)
            # Optimization heuristic : we only return document with a high enough weight
            if HEURISTIC3 :
                if ( docID_weight[1] ) > WEIGHT_DOCUMENT_THRESHOLD :
                    documents.append(docID_weight)
                else:
                    break
            else:
                documents.append(docID_weight)
        
    return documents

def do_and(p1, p2):
    res = []

    ptr1, ptr2 = 0, 0

    while ptr1 < len(p1) and ptr2 < len(p2):
        d1, d2 = p1[ptr1], p2[ptr2]

        if d1 == d2:
            res.append(d1)
            ptr1 += 1
            ptr2 += 1

        elif d1 < d2:
            ptr1 += 1

        else:
            ptr2 += 1

    return res

def eval_AND(tokens,dictionary,posting_file):
    # Get documents that contains every token in tokens
    # :param tokens: Array of unigrams -> ["bank","of","newyork"]
    # :param dictionary: Dictionary of the collection, format : {term1: (termFrequency, offset)...}
    # :param postings_file: (str) Path of the postings
    # :return res: documents that match the AND query

    documents = [] # contains the list of documents that contains each token -> [ [(doc1 that contains tokens[0], score).. ], [(doc1 that contains tokens[1], score)..  ], ... ]

    # Get the documents lists for each token
    for token in tokens :
        postings_list = get_postings_list(token=token, dictionary=dictionary, postings_file=postings_file)
        documents.append(postings_list)

    while len(documents) > 1:
        p1 = documents.pop()
        p2 = documents.pop()
        new_combined_postings_list = do_and(p1=p1, p2=p2)
        documents.append(new_combined_postings_list)

    res = documents[0]

    return res

def get_log_weighted_term_freq(collection_size, df):
    """calculate the log weighted term frequency (in corpus)"""

    return (
        math.log10(collection_size / df)
        if collection_size > 0 and df > 0
        else 0
    )

def get_tf_idf_score(tf, idf):
    """calculate tf-idf"""
    return (1+math.log(tf, 10))*idf


def get_log_freq_weighting_scheme(tf):
    """calculate the log weighted term frequency"""
    return (1+math.log10(tf)) if tf > 0 else 0


def get_centroid_vector(tokens, dictionary, relevant_docs, collection_size):
    centroid_vector = {}
    doc_len = len(relevant_docs)

    for term, freq in tokens.items():
        tf = freq

        # get doc freq if exists, else return 0
        df = dictionary.get(term, [0])[0]

        idf = get_log_weighted_term_freq(collection_size=collection_size, df=df)

        ln = get_tf_idf_score(tf=tf, idf=idf)

        for doc in relevant_docs:
            euclidean_doc_len = get_euclidean_doc_len(doc)

            ltc = get_log_freq_weighting_scheme(tf=dictionary[term][0]) / euclidean_doc_len

            ln_ltc = ln*ltc
            centroid_vector[term] = centroid_vector.get(term, 0)+ln_ltc

    for term in centroid_vector:
        centroid_vector[term]/=doc_len
        centroid_vector[term]*=0.8

    return centroid_vector

def get_cosine_scores(dictionary, docs, tokens, relevant_docs):
    """(recursively) compute cosine scores of documents"""
    # edge case - if there are no documents matching the query at all
    if not docs:
        return

    collection_size = N
    scores = {}

    centroid_vector = get_centroid_vector(
        tokens=tokens, dictionary=dictionary, relevant_docs=relevant_docs, collection_size=collection_size
    ) if relevant_docs else {term: 1 for term in tokens}

    for term, freq in tokens.items():
        tf = freq

        # get doc freq if exists, else return 0
        df = dictionary.get(term, [0])[0]

        idf = get_log_weighted_term_freq(collection_size=collection_size, df=df)

        ln = get_tf_idf_score(tf=tf, idf=idf)

        for doc in set(docs):
            euclidean_doc_len = get_euclidean_doc_len(doc)
            ltc = get_log_freq_weighting_scheme(tf=dictionary[term][0])/ euclidean_doc_len
            ln_ltc = ln*ltc
            scores[doc] = scores.get(doc, 0)+(ln_ltc*centroid_vector[term])

    # if no relevant documents were passed in, perform pseudo relevance feedback
    if not relevant_docs:
        pseudo_relevant_docs = get_documents(scores)[:8]
        return get_cosine_scores(
            dictionary=dictionary,
            docs=docs,
            tokens=tokens,
            relevant_docs=pseudo_relevant_docs
        )

    return get_documents(scores)

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def print_dictionary(d):
    for key in sorted(d):
        df = d[key][0]
        print(f'{key} {df}')

def get_synonyms(query, dictionary):
    tokens = []
    for token, pos in nltk.pos_tag(query):
        if token.isnumeric():
            continue

        synset = wordnet.synsets(token, pos=POS_SET.get(pos[0] if pos else None))
        
        results = [el.name().split(".")[0] for el in synset]
        results = set(results)

        for el in results:
            if el in dictionary and el != token:
                tokens.append(el)
                break
            
    return set(tokens)

def get_tokens(query_str, dictionary):
    # remove AND and "
    # split by whitespace
    tokens = query_str.replace('"', '').replace('AND', '').split()

    # get synonyms
    # tokens.extend(get_synonyms(tokens, dictionary))

    # apply case fold
    tokens = map(lambda word: word.casefold(), tokens)

    # apply stemming
    tokens = map(lambda word: STEMMER.stem(word), tokens)

    # remove dispensable punctuation
    punctuations = set(punctuation) - set("$@%")
    tokens = filter(lambda word: word not in punctuations, tokens)

    return Counter(tokens)

def get_query_and_relevant_docs(src):
    with open(src, 'r') as f:
        data = [line.rstrip() for line in f.readlines()]

    if len(data) == 1:
        return data[0], None
    else:
        # there are relevant documents provided
        return data[0], list(map(int, data[1:]))

def run_search(dict_file, postings_file, queries_file, results_file):
    dictionary = retrieve_dict(dict_file) # Get the dictionary

    query_str, relevant_docs = get_query_and_relevant_docs(queries_file)

    tokens = get_tokens(query_str, dictionary)

    documents = eval_AND(tokens=tokens.keys(), dictionary=dictionary, posting_file=postings_file)
    if relevant_docs:
        documents.extend(relevant_docs)

    synonyms = get_synonyms(tokens.keys(), dictionary)
    for synonym in synonyms:
        documents.extend(get_postings_list(token=synonym, dictionary=dictionary, postings_file=postings_file))
        tokens[synonym] = tokens.get(synonym, 0)+1

    res = get_cosine_scores(dictionary=dictionary, docs=documents, tokens=tokens, relevant_docs=relevant_docs)
    res = list(map(str, res)) if res else []

    with open(results_file, 'w') as output:
        print(' '.join(res), file=output)


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
# dictionary = retrieve_dict(dictionary_file)
# print(eval_AND(["real","madrid","footbal"],dictionary,postings_file))