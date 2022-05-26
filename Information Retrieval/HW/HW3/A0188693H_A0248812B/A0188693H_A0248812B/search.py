#!/usr/bin/python3
import getopt
import heapq as hq
import sys

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from models import Dictionary
from utils import get_tokens, get_term_frequency_dict, CosineScore, clearFile, WriteResult


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

    dictionary = Dictionary()
    dictionary.load(dict_file)

    queries = GetQueryFromDoc(queries_file)
    clearFile(results_file)

    for query in queries:
        # get cosine score
        score_norm = CosineScore(postings_file=postings_file, Query=query, Dictionary=dictionary)
        # get top k documents
        TopK_result = TopK_doc(score_norm)
        # writing
        WriteResult(results_file, TopK_result)


def GetQueryFromDoc(file):
    """tokenize the queries in documents"""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    with open(file, 'r') as f:
        lines = f.readlines()
    queries = []
    for line in lines:
        corpus = line
        tokens = get_tokens(corpus=corpus, stemmer=stemmer, stop_words=stop_words,
                            tokenizer=lambda query: query.split())
        queries.append(get_term_frequency_dict(tokens=tokens))
    return queries


def TopK_doc(Scores):
    """ranking and retrieve top k result"""
    # sort with the docID first
    Scores = dict(sorted(Scores.items()))
    Top_k = min(10, len(Scores))
    Scores_new = Scores
    while len(Scores_new) > 100:
        score_temp = Scores_new
        terms = list(score_temp.keys())
        doc_top = []
        while len(terms):
            term_100 =  terms[:100]
            Scores_100 = {i:Scores_new[i] for i in term_100}
            doc_top.extend(hq.nlargest(Top_k, Scores_100, key=Scores_new.get))
            terms = terms[100:]
        Scores_new = {i:Scores_new[i] for i in doc_top}
        Scores_new = dict(sorted(Scores_new.items()))
    return hq.nlargest(Top_k, Scores_new, key=Scores_new.get)

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
