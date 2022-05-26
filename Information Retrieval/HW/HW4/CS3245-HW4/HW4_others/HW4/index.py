#!/usr/bin/python3
import sys
import getopt
import os
import re
import csv
from invertedindex import InvertedIndex
from typing import Dict

DEVELOPMENT = False
ZONE_DELIMITER = f"\n{'-' * 100}\n"


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -i directory-of-documents -d dictionary-file -p postings-file"
    )


def increase_csv_field_size_limit(new_limit: int):
    ## reference: https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
    while True:
        try:
            csv.field_size_limit(new_limit)
            break
        except OverflowError:
            new_limit //= 10


def parse_csv_to_corpora(doc: Dict[str, str]):
    os.makedirs("corpora", exist_ok=True)

    with open(f"corpora/{doc.get('document_id')}", mode="w", encoding="utf-8") as f:
        f.write(ZONE_DELIMITER.join(doc.values()))


def build_index(in_dir: str, out_dict: str, out_postings: str):
    """
    Build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print("Initializing inverted index...")

    inverted_index = InvertedIndex()

    increase_csv_field_size_limit(sys.maxsize)

    print("Indexing documents...")

    with open(in_dir, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, restval="")
        i = 0
        for doc in reader:
            ## uncomment to generate corpora
            ## parse_csv_to_corpora(doc)
            if DEVELOPMENT and i == 200:
                break
            inverted_index.index(doc=doc)
            i += 1
            print("Num docs indexed:", i)

    print("Building inverted index...")
    inverted_index.build(postings_file=out_postings)

    print("Exporting inverted index...")
    inverted_index.export(dictionary_file=out_dict)

    print("Done!")


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:d:p:")
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == "-i":  # input directory
        input_directory = a
    elif o == "-d":  # dictionary file
        output_file_dictionary = a
    elif o == "-p":  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if (
    input_directory == None
    or output_file_postings == None
    or output_file_dictionary == None
):
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
