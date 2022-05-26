#!/usr/bin/python3
import sys
import getopt
from invertedindex import InvertedIndex

## (optional) training data to improve search
TRAINING_DATA = [
    ("quiet phone call", [6807771, 4001247, 3992148]),
    ("good grades exchange scandal", [2211154, 2748529]),
    ('"fertility treatment" AND damages', [4273155, 3243674, 2702938]),
]


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"
    )


def run_search(
    dict_file: str, postings_file: str, queries_file: str, results_file: str
):
    """
    Using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print("Loading inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.load(dict_file)
    inverted_index.train_search(TRAINING_DATA)

    print("Running search on the queries...")

    with open(queries_file, mode="r") as f, open(results_file, mode="w") as output:
        lines = f.readlines()

        query = lines[0].strip()
        relevant_doc_ids = set()

        for line in lines[1:]:
            try:
                relevant_doc_ids.add(int(line.strip()))
            except:
                continue

        result = inverted_index.search(
            query=query, postings_file=postings_file, relevant_doc_ids=relevant_doc_ids
        )

        result_string = " ".join(map(str, result))

        print(result_string, file=output)

    print("Done!")


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "d:p:q:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == "-d":
        dictionary_file = a
    elif o == "-p":
        postings_file = a
    elif o == "-q":
        file_of_queries = a
    elif o == "-o":
        file_of_output = a
    else:
        assert False, "unhandled option"

if (
    dictionary_file == None
    or postings_file == None
    or file_of_queries == None
    or file_of_output == None
):
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
