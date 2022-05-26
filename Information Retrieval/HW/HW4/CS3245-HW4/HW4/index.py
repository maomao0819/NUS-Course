from datetime import datetime
# from nltk.stem import snowball
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords, wordnet
from typing import Sequence, Optional, Dict, Callable, Tuple, Any, Iterable, Set
import sys

CASE_FOLD = True
STEMMING = True
REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATIONS = True

def increase_csv_field_size_limit(new_limit: int):
    ## reference: https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
    while True:
        try:
            csv.field_size_limit(new_limit)
            break
        except OverflowError:
            new_limit //= 10

def Tokenize(Query):
    # split with space
    QueryList = Query.split()
    return QueryList

def ConcatenateWords(WordsList):
	Words = ''
	for elem in WordsList:
		Words += elem
	return Words

def TokenizeToBooleanQuery(Query):
	QueryList = Query.split()
	phrase_start_id = 0
	phrase_end_id = 0
	phrase_words = list()
	NewQueryList = list()
	for query in QueryList:
		if query == 'AND':
			phrase = ConcatenateWords(phrase_word)
			NewQueryList.append(phrase)
			phrase_words = list()
			NewQueryList.append('AND')
		else:
			phrase_words.append(query)
	if phrase_words:
		phrase = ConcatenateWords(phrase_word)
		NewQueryList.append(phrase)
	return NewQueryList

def TokenizeToFreeText(Query):
	QueryList = Query.split()
	return [query for query in QueryList if query != 'AND']



class Preprocessor:
    def __init__(
        self,
        case_fold: bool,
        stemming: bool,
        remove_stop_words: bool,
        remove_punctuations: bool,
    ):
        self.case_fold = case_fold
        self.stemming = stemming
        self.remove_stop_words = remove_stop_words
        self.remove_punctuations = remove_punctuations

        ## newer porter stemmer that is widely considered to be better than the orginal
        self.stemmer = snowball.SnowballStemmer(language="english")

        self.stop_words = set(stopwords.words("english"))

        self.non_punctuations_regex = re.compile("[a-zA-Z0-9]+")
        self.phrases_regex = re.compile("('.*?'|\".*?\"|\S+)")
        self.and_regex = re.compile(f"[\W\s_]({BOOLEAN_AND})[\W\s_]")

    def tokenize(self, corpus: str, no_stemming: bool = False) -> Sequence[str]:
        """
        Extract tokens from a string with the necessary modificiations
        (stemming/case folding/stop words removal/punctuations removal)
        """
        words = (
            self.non_punctuations_regex.findall(corpus)
            if self.remove_punctuations
            else itertools.chain(*map(word_tokenize, sent_tokenize(corpus)))
        )

        if self.case_fold:
            words = map(lambda word: word.casefold(), words)

        if self.remove_stop_words:
            words = filter(lambda word: word not in self.stop_words, words)

        if self.stemming and not no_stemming:
            words = map(lambda word: self.stemmer.stem(word), words)

        return list(words)

    def parse_query(self, query: str) -> Sequence[str]:
        """
        Parse a query into a list of space-separated tokens with the necessary
        modifications (case folding, stemming etc), preserving phrases in
        quotation marks.
        """
        tokens = [token.strip("\"' ") for token in self.phrases_regex.findall(query)]

        return [
            " ".join(self.tokenize(token))
            for token in tokens
            if token != BOOLEAN_AND and self.tokenize(token)
        ]

    def is_boolean_query(self, query: str) -> bool:
        """
        Check if a query is a boolean query
        """
        return bool(self.and_regex.search(query))

    def tokenize_date_string(self, date_string: str) -> Sequence[str]:
        """
        Extract the date tokens from a date string
        """
        try:
            date = datetime.fromisoformat(date_string)
            ## only extract year, month and day
            return [str(date.year), str(date.month), str(date.day)]
        except:
            ## manually extract datetime tokens
            tokens = []

            for string in self.tokenize(date_string):
                for substring in self.non_punctuations_regex.findall(string):
                    if not substring:
                        continue

                    try:
                        tokens.append(str(int(substring)))
                    except:
                        tokens.append(substring)

            return tokens

# increase_csv_field_size_limit(sys.maxsize)
print(sys.maxsize)
# in_dir = 'dataset/dataset.csv'

# with open(in_dir, mode="r", encoding="utf-8") as f:
#         reader = csv.DictReader(f, restval="")
#         i = 0
#         for doc in reader:
#             ## uncomment to generate corpora
#             ## parse_csv_to_corpora(doc)
#             if DEVELOPMENT and i == 200:
#                 break
#             inverted_index.index(doc=doc)
#             i += 1
#             print("Num docs indexed:", i)

# corpus = " ".join([doc.get(TITLE), doc.get(CONTENT), doc.get(COURT)])

# ## extract court hierarchy information
# court = (
#     doc.get(COURT).casefold() if self.preprocessor.case_fold else doc.get(COURT)
# )
# for hierarchy_type, courts in COURTS.items():
#     if court in courts:
#         court_hierarchy_type = hierarchy_type
#         break
# else:
#     court_hierarchy_type = CourtHierarchyType.DEFAULT

# corpus_tokens = self.preprocessor.tokenize(corpus)
# ## extract date information
# date_tokens = self.preprocessor.tokenize_date_string(doc.get(DATE_POSTED))