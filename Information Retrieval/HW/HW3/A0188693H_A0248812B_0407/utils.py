import math
import pickle
import string
from collections import Counter
import itertools

from nltk.tokenize import sent_tokenize, word_tokenize


def get_tokens(corpus, stemmer, stop_words, tokenizer=word_tokenize):
    """convert a string input into dictionary where key=term and value=frequency"""

    # use nltk.tokenize to separate corpus into individual words
    sentences = sent_tokenize(corpus)
    words = itertools.chain(*map(tokenizer, sentences))

    # case fold words
    words = map(lambda word: word.casefold(), words)

    # remove stop words
    # words = filter(lambda word: word not in stop_words, words)

    # perform stemming on words
    words = map(lambda word: stemmer.stem(word), words)

    # remove dispensable punctuation
    punctuations = set(string.punctuation) - set("$@%")
    words = filter(lambda word: word not in punctuations, words)

    return Counter(words)


def get_term_frequency_dict(tokens):
    """convert an array of tokens into a dictionary storing the terms (key) and their frequencies (value)"""

    return Counter(tokens)


def get_idf_weighting_scheme(collection_size, num_of_documents_with_term):
    """calculate the inverse document frequency for a term"""

    return (
        math.log10(collection_size/num_of_documents_with_term)
        if collection_size > 0 and num_of_documents_with_term > 0
        else 0
    )


def get_log_freq_weighting_scheme(term_freq):
    """calculate the log weighted term frequency"""
    return (1+math.log10(term_freq)) if term_freq > 0 else 0


def build_dictionary(dictionary):
    """takes in Dictionary object with all corpus terms added and processes inverse doc freq of terms"""

    dictionary_metadata = dictionary.dictionary

    for term, term_metadata in dictionary_metadata.items():
        num_of_documents_with_term = term_metadata.document_frequency
        term_metadata.inverse_document_frequency = get_idf_weighting_scheme(
            collection_size=dictionary.collection_size,
            num_of_documents_with_term=num_of_documents_with_term
        )


def read(src):
    """load Object from pickle file"""

    with open(src, 'rb') as f:
        return pickle.load(f)


def readline(file_writer, ptr):
    """read line of data from a pickle file"""

    file_writer.seek(ptr)
    pickled_data = file_writer.readline()

    return pickle.loads(pickled_data)


def get_pickle(data):
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_all(dictionary, dic_dest, postings, post_dest):
    """pickle Dictionary and Postings object"""

    pickled_corpus_postings_list = get_pickle(postings.corpus_postings_list)

    pickled_postings_lists = [pickled_corpus_postings_list]
    disk_position_offset = len(pickled_corpus_postings_list)

    dictionary.corpus_disk_data_length = disk_position_offset

    for term, postings_list in postings.postings_list_dict.items():
        pickled_postings_list = get_pickle(postings_list)
        pickled_postings_lists.append(pickled_postings_list)
        disk_data_length = len(pickled_postings_list)

        dictionary.dictionary[term].disk_position_offset = disk_position_offset
        dictionary.dictionary[term].disk_data_length = disk_data_length

        disk_position_offset += disk_data_length

    # write postings
    with open(post_dest, mode="wb") as f:
        f.writelines(pickled_postings_lists)

    # save dictionary
    dictionary.save(dic_dest)


def print_dictionary(dictionary):
    """function to print out entire dictionary for debugging purposes"""

    collection_size = dictionary.collection_size

    # print('collection size: ', collection_size)

    dictionary_metadata = dictionary.dictionary
    terms = sorted(dictionary_metadata.items())
    for term, ptr in terms:
        idf = dictionary_metadata[term].inverse_document_frequency
        # print(len(dictionary.get_posting_list(term, 'postings.txt')))
    # for term, term_metadata in dictionary_metadata.items():
    #     term_freq_in_docs = term_metadata.term_frequency_in_doc
    #     num_of_documents_with_term = term_metadata.num_of_documents_with_term
    #     postings_list_ptr = term_metadata.postings_list_ptr
    #     idf = term_metadata.inverse_doc_frequency
    #
    #     print('term: ', term)
    #     print('tf: ', term_freq_in_docs)
    #     print('df: ', num_of_documents_with_term)
    #     print('idf: ', idf)
    #     print('postings list ptr: ', postings_list_ptr)


def tf_idf(tf: int, idf: float) -> float:
    """calculate tf-idf"""
    return (1 + math.log(tf, 10)) * idf


def CosineScore(postings_file, Query, Dictionary):
    """calculate the cosine score"""

    scores = dict()

    for term, frequency in Query.items():
        tf = frequency
        if tf <= 0:
            continue

        idf = Dictionary.get_inverse_doc_frequency(term)
        if idf < 0:
            continue

        W_tq = tf_idf(tf, idf)
        docs = Dictionary.get_posting_list(postings_file=postings_file, term=term)

        if len(docs) == 0:
            continue

        for doc in docs:
            normalized_tf_idf_weight = Dictionary.get_normalized_tf_idf_weight(doc=doc, term=term)
            scores[doc] = scores.get(doc, 0) + (W_tq * normalized_tf_idf_weight)

    return scores


def clearFile(filename):
    """clear the file"""
    file = open(filename, 'w')
    file.close()


def WriteResult(file, result):
    """write results into the text file"""
    outputFile = open(file, 'a')
    for text in result:
        outputFile.write(str(text))
        outputFile.write(' ')
    outputFile.write('\n')