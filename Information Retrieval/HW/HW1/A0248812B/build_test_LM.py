#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import nltk
import sys
import getopt
import math     # for log

# keep only letters or spaces and cast to lower case
def process_word(sentence):
    process_sentence = ''
    for text in sentence:
        if text.isalpha() or text.isspace():
            process_sentence += text
    process_sentence.replace('  ', ' ')
    return process_sentence.lower()

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print("building language models...")
    # This is an empty method
    # Pls implement your code below

    # 4-grams
    N = 4
    labels = ['malaysian', 'indonesian', 'tamil']
    Ngrams = {
        'malaysian': {},
        'indonesian': {},
        'tamil': {}
    }
    sentences = open(in_file).readlines()

    def get_label(first_char):
        if first_char == 'M' or first_char == 'm':
            return 'malaysian'
        elif first_char == 'I' or first_char == 'i':
            return 'indonesian'
        elif first_char == 'T' or first_char == 't':
            return 'tamil'

    for sentence in sentences:
        # process the sentence, remove numbers, duplicated spaces, and non-alphabet text, and cast to lower case.
        process_sentence = process_word(sentence)
        current_label = get_label(sentence[0])
        # labels are no need for the key of Ngram
        offset = len(current_label) + 1
        # get certain 4-character strings, split into chars, and store with tuples.
        char_4grams = [tuple(process_sentence[i : i + N]) for i in range(offset, len(process_sentence) - N + 1)]
        for label in labels:
            if label == current_label:
                for char_4gram in char_4grams:
                    # Ngram has recorded
                    if char_4gram in Ngrams[label]:
                        Ngrams[label][char_4gram] += 1
                    # new tuples for Ngram
                    else:
                        Ngrams[label][char_4gram] = 1
            # create zero entry for other labels
            else:
                for char_4gram in char_4grams:
                    if not(char_4gram in Ngrams[label]):
                        Ngrams[label][char_4gram] = 0
    # add 1 smoothing
    for label in labels:
        for key in Ngrams[label]:
            Ngrams[label][key] += 1
    # convert counts to probabilities
    for label in labels:
        # calculate total counts in the dictionary with the certain key 
        values = Ngrams[label].values()
        total = sum(values)
        for key in Ngrams[label]:
            Ngrams[label][key] /= total
    return Ngrams



def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")
    # This is an empty method
    # Pls implement your code below

    # 4-grams
    N = 4
    labels = ['malaysian', 'indonesian', 'tamil']
    # tell the 'other'
    other_threshold = 1.2
    sentences = open(in_file).readlines()
    
    # get the key with max value
    def get_max_label(dic):
        v = list(dic.values())
        k = list(dic.keys())
        return k[v.index(max(v))]
    
    output_sentences = ''
    for sentence in sentences:
        # process the sentence, remove numbers, duplicated spaces, and non-alphabet text, and cast to lower case. 
        process_sentence = process_word(sentence)
        # create with keys and corresponding values
        # store the probabilities for corresponding labels
        label_predict = dict.fromkeys(labels, 0)
        # get certain 4-character strings, split into chars, and store with tuples
        char_4grams = [tuple(process_sentence[i : i + N]) for i in range(len(process_sentence) - N + 1)]
        # record how many tuples don't show up in training data
        # the factor for telling the 'other'
        other_time = 0
        for label in labels:
            for char_4gram in char_4grams:
                if char_4gram in LM[label]:
                    # a * b * c == exp(log(a) + log(b) + log(c))
                    # a * b > c * d <=> log(a) + log(b) > log(c) + log(d)
                    label_predict[label] += math.log(LM[label][char_4gram])
                else:
                    other_time += 1
        # the bigger the proportion of tuples that don't show up in training data to the length of sentence is, the bigger the chance of being other is 
        if other_time / len(process_sentence) > other_threshold:
            predict_label = 'other'
        else:
            # get the key with max value
            predict_label = get_max_label(label_predict)
        output_sentences += (predict_label + ' ' + sentence)
    # write the file
    f = open(out_file, 'w').write(output_sentences)

def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file"
    )


input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], "b:t:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == "-b":
        input_file_b = a
    elif o == "-t":
        input_file_t = a
    elif o == "-o":
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)

# Reference:
#     create empty dictionary: https://pythonguides.com/how-to-create-an-empty-python-dictionary/
#     create ngrams: https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
#     string to seperate char tuple: https://stackoverflow.com/questions/16449184/converting-string-to-tuple-without-splitting-characters
#     check keys if exist in dictionary: https://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary
#     sum values in dictionary: https://www.kite.com/python/answers/how-to-sum-the-values-in-a-dictionary-in-python
#     argmax of dictionary: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
#     write files: https://www.w3schools.com/python/python_file_write.asp