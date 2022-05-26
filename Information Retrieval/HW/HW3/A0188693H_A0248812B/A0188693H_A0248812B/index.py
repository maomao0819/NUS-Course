#!/usr/bin/python3
import getopt
import os
import sys

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from models import Dictionary, Postings
from utils import get_tokens, build_dictionary, pickle_all, get_log_freq_weighting_scheme


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    documents = sorted(map(int, os.listdir(in_dir)))
    collection_size = len(documents)

    dictionary = Dictionary(collection_size=collection_size)
    postings = Postings(corpus_postings_list=documents)

    for document in documents:
        src = os.path.join(in_dir, str(document))
        with open(src, 'r') as f:
            corpus = f.read()
            tokens = get_tokens(corpus=corpus, stemmer=stemmer, stop_words=stop_words)
            euclidean_doc_length = sum(
                map(
                    lambda term_freq: get_log_freq_weighting_scheme(term_freq=term_freq)**2, tokens.values()
                )
            )**0.5

            for term, freq in tokens.items():
                normalized_tf_idf_weight = get_log_freq_weighting_scheme(term_freq=freq)/euclidean_doc_length \
                    if euclidean_doc_length != 0 else 0
                dictionary.add_document_term_and_frequency(doc=document, term=term,
                                                           normalized_tf_idf_weight=normalized_tf_idf_weight)
                postings.add_document_to_postings_list(doc=document, term=term)

    build_dictionary(dictionary=dictionary)
    pickle_all(dictionary=dictionary, dic_dest=out_dict, postings=postings, post_dest=out_postings)


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
