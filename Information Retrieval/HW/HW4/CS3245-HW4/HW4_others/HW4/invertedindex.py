import pickle
import sys
import re
import itertools
import heapq
import nltk
import os
import json
from http.client import HTTPSConnection
from datetime import datetime
from nltk.stem import snowball
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from typing import Sequence, Optional, Dict, Callable, Tuple, Any, Iterable, Set
from collections import Counter, defaultdict, deque

from models import CourtHierarchyType, PostingList, PostingListMetadata, Dictionary

## required to pickle large posting list
sys.setrecursionlimit(int(1e8))

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

CASE_FOLD = True
STEMMING = True
REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATIONS = True

ROCCHIO_RELEVANCE = True
PSEUDO_RELEVANCE_FEEDBACK = True and ROCCHIO_RELEVANCE
MAX_ROUNDS_OF_RELEVANCE_FEEDBACK = 2
NUM_TOP_DOCUMENTS_FOR_RELEVANCE_FEEDBACK = 5

KEYWORDS_MATCH_THRESHOLD = 0.5

BLOCK_SIZE_LIMIT = int(1e6)  ## max number of postings in memory

DOCUMENT_ID = "document_id"
TITLE = "title"
CONTENT = "content"
DATE_POSTED = "date_posted"
COURT = "court"

DICTIONARY_FILE_PREFIX = "temp-dictionary"
POSTINGS_FILE_PREFIX = "temp-postings"

BOOLEAN_AND = "AND"

ROCCHIO_ALPHA = 1.0
ROCCHIO_BETA = 0.75 if ROCCHIO_RELEVANCE else 0

ONLINE_SYN_API_URL = "twinword-word-associations-v1.p.rapidapi.com"
ONLINE_SYN_API_REQUEST_HEADERS = {
    "x-rapidapi-key": "1e86b2f3c0msh460154e5b4033f2p1426f2jsnb81cf8bf6f9e",
    "x-rapidapi-host": "twinword-word-associations-v1.p.rapidapi.com",
}
SYN_THRESHOLD = 5

POS_SET = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "R": wordnet.ADV,
    "V": wordnet.VERB,
}

COURTS = {
    CourtHierarchyType.MOST_IMPORTANT: {
        "SG Court of Appeal",
        "SG Privy Council",
        "UK House of Lords",
        "UK Supreme Court",
        "High Court of Australia",
        "CA Supreme Court",
    },
    CourtHierarchyType.IMPORTANT: {
        "SG High Court",
        "Singapore International Commercial Court",
        "HK High Court",
        "HK Court of First Instance",
        "UK Crown Court",
        "UK Court of Appeal",
        "UK High Court",
        "Federal Court of Australia",
        "NSW Court of Appeal",
        "NSW Court of Criminal Appeal",
        "NSW Supreme Court",
    },
}

if CASE_FOLD:
    for hierarchy_type, courts in COURTS.items():
        COURTS[hierarchy_type] = {court.casefold() for court in courts}

COURT_HIERARCHY_SCORE_FACTORS = {
    CourtHierarchyType.MOST_IMPORTANT: 0.2,
    CourtHierarchyType.IMPORTANT: 0.1,
    CourtHierarchyType.DEFAULT: 0,
}

BOOLEAN_SCORE_FACTORS = {True: 0.15, False: 0}


class SmallestTermExtractor:
    """
    Util class to perform merging of k-sorted lists
    """

    def __init__(self, sorted_lists: Sequence[Sequence[Tuple[str, Any]]]):
        self.index_to_queue_mapping = {
            index: deque(sorted_list) for index, sorted_list in enumerate(sorted_lists)
        }

        self.smallest_term_candidate_to_index_value_pair_mapping = defaultdict(dict)
        self.smallest_term_candidate_heap = []

        self.populate_smallest_term_candidates(
            indexes=self.index_to_queue_mapping.keys()
        )

    def is_empty(self):
        return len(self.smallest_term_candidate_heap) == 0

    def peek_smallest_term(self) -> Optional[str]:
        if self.is_empty():
            return

        return self.smallest_term_candidate_heap[0]

    def extract_smallest_term(self) -> Tuple[Optional[str], Dict[int, Any]]:
        if self.is_empty():
            return None, {}

        smallest_term = heapq.heappop(self.smallest_term_candidate_heap)

        index_value_pairs = self.smallest_term_candidate_to_index_value_pair_mapping[
            smallest_term
        ]
        del self.smallest_term_candidate_to_index_value_pair_mapping[smallest_term]

        self.populate_smallest_term_candidates(indexes=index_value_pairs.keys())

        return smallest_term, index_value_pairs

    def populate_smallest_term_candidates(self, indexes: Iterable[int]):
        for index in indexes:
            queue = self.index_to_queue_mapping[index]

            if not queue:
                del self.index_to_queue_mapping[index]
                continue

            term, value = queue.popleft()

            if term not in self.smallest_term_candidate_to_index_value_pair_mapping:
                heapq.heappush(self.smallest_term_candidate_heap, term)

            self.smallest_term_candidate_to_index_value_pair_mapping[term][
                index
            ] = value


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


class Indexer:
    def __init__(
        self,
        block_size_limit: int,
        dictionary_file_prefix=DICTIONARY_FILE_PREFIX,
        postings_file_prefix=POSTINGS_FILE_PREFIX,
    ):
        ## configs
        self.dictionary_file_prefix = dictionary_file_prefix
        self.postings_file_prefix = postings_file_prefix
        self.block_size_limit = block_size_limit

        ## indexing attributes
        self.current_block_size = 0
        self.num_intermediate_blocks = 0
        ## for merging intermediate blocks
        self.merged_dictionary: Dictionary = None
        self.merged_disk_position_offset = 0

        ## represents the in-memory block
        ## stores temporary term -> posting list mapping
        self.term_to_posting_list_mapping: Dict[str, PostingList] = {}

        self.doc_id_to_court_hierarchy_type_mapping: Dict[int, CourtHierarchyType] = {}
        self.collection_size = 0

    def clean_up(self):
        """Clean up and reset the state of the indexer"""
        self.remove_intermediate_files()
        ## reset states
        self.current_block_size = 0
        self.num_intermediate_blocks = 0
        self.merged_dictionary = None
        self.merged_disk_position_offset = 0
        self.term_to_posting_list_mapping = {}
        self.collection_size = 0
        self.doc_id_to_court_hierarchy_type_mapping = {}

    def index(
        self,
        tokens: Sequence[str],
        doc_id: int,
        court_hierarchy_type: CourtHierarchyType,
    ):
        """Index the tokens associated with the given doc id"""

        ## skip this document if doc id has already been indexed
        if doc_id in self.doc_id_to_court_hierarchy_type_mapping:
            return

        ## store the document's court hierarchy type
        self.doc_id_to_court_hierarchy_type_mapping[doc_id] = court_hierarchy_type

        self.collection_size += 1

        ## For each token, store a list of its positions in the document
        token_to_positions_mapping = defaultdict(list)

        for index, token in enumerate(tokens):
            token_to_positions_mapping[token].append(index)

        ## calculate the length of document as the sqrt of
        ## sum of squared term frequency weights for normalizaton
        euclidean_doc_length = (
            sum(
                Dictionary.compute_term_freq_weight(len(positions)) ** 2
                for positions in token_to_positions_mapping.values()
            )
            ** 0.5
        )

        normalization_factor = Dictionary.compute_normalization_factor(
            euclidean_doc_length
        )

        ## add information for each term to postings lists
        for term, positions in token_to_positions_mapping.items():
            if self.should_free_memory():
                self.write_intermediate_block_to_disk()

            term_freq = len(positions)

            ## retrieve posting list for term if it already exists
            ## otherwise create a new posting list
            if term in self.term_to_posting_list_mapping:
                posting_list = self.term_to_posting_list_mapping[term]
            else:
                posting_list = self.index_posting_list(
                    term=term, posting_list=PostingList()
                )

            ## calculate normalized term tf-idf weight and add to posting list
            normalized_tf_idf_weight = (
                Dictionary.compute_term_freq_weight(term_freq) * normalization_factor
            )

            posting_list.add(
                doc_id=doc_id,
                term_freq=term_freq,
                normalized_tf_idf_weight=normalized_tf_idf_weight,
                positions=positions,
            )
            self.current_block_size += 1

    def should_free_memory(self) -> bool:
        """Check if current block size exceeds the block size limit"""
        return self.current_block_size >= self.block_size_limit

    def index_posting_list(self, term: str, posting_list: PostingList) -> PostingList:
        """Create a (temporary) dictionary entry of term -> posting list"""
        self.term_to_posting_list_mapping[term] = posting_list
        self.current_block_size += len(posting_list)
        return posting_list

    def prepare_for_disk_storage(
        self, disk_position_offset: int, dictionary_to_be_updated: Dictionary
    ):
        """Pickle the posting lists and update the dictionary to contain the posting list metadata"""
        pickled_posting_lists = []

        for term, posting_list in self.term_to_posting_list_mapping.items():
            pickled_posting_list = pickle.dumps(
                posting_list, protocol=pickle.HIGHEST_PROTOCOL
            )

            posting_list_metadata = PostingListMetadata(
                doc_freq=len(posting_list),
                disk_position_offset=disk_position_offset,
                disk_data_length=len(pickled_posting_list),
            )

            dictionary_to_be_updated.add(
                term=term, posting_list_metadata=posting_list_metadata
            )

            pickled_posting_lists.append(pickled_posting_list)
            disk_position_offset += posting_list_metadata.disk_data_length

        return pickled_posting_lists, disk_position_offset

    def write_intermediate_block_to_disk(self):
        """Export current in-memory intermediate block to disk"""
        ## do nothing if there is nothing to write
        if self.current_block_size == 0:
            return

        intermediate_dictionary = Dictionary()

        pickled_posting_lists, _ = self.prepare_for_disk_storage(
            disk_position_offset=0, dictionary_to_be_updated=intermediate_dictionary
        )

        sorted_terms_dictionary = intermediate_dictionary.to_sorted_list()

        ## write postings to intermediate file
        with open(
            f"{self.postings_file_prefix}{self.num_intermediate_blocks}", mode="wb"
        ) as f:
            f.writelines(pickled_posting_lists)

        ## write dictionary with sorted terms to intermediate file
        with open(
            f"{self.dictionary_file_prefix}{self.num_intermediate_blocks}", mode="wb"
        ) as f:
            pickle.dump(
                sorted_terms_dictionary,
                file=f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        ## increment intermediate block count
        self.num_intermediate_blocks += 1
        ## reset in-memory states
        self.current_block_size = 0
        self.term_to_posting_list_mapping = {}

    def write_merged_block_to_disk(self, postings_file: str):
        """Export current in-memory merged block to disk"""
        ## do nothing if there is nothing to write
        if self.current_block_size == 0 or not self.merged_dictionary:
            return

        (
            pickled_posting_lists,
            self.merged_disk_position_offset,
        ) = self.prepare_for_disk_storage(
            disk_position_offset=self.merged_disk_position_offset,
            dictionary_to_be_updated=self.merged_dictionary,
        )

        ## append postings to merged postings file
        with open(postings_file, mode="ab") as f:
            f.writelines(pickled_posting_lists)

        ## reset in-memory states
        self.current_block_size = 0
        self.term_to_posting_list_mapping = {}

    def merge(self, postings_file: str) -> Dictionary:
        """Merge all the intermediate blocks from disk to form the final dictionary and postings"""

        ## write any remaining partial intermediate block to disk
        self.write_intermediate_block_to_disk()

        ## setup states required for merging
        self.merged_dictionary = Dictionary(
            collection_size=self.collection_size,
            doc_id_to_court_hierarchy_type_mapping=self.doc_id_to_court_hierarchy_type_mapping,
        )
        self.merged_disk_position_offset = 0
        ## remove any existing postings file to prevent file corruption when
        ## new data is written to it
        if os.path.exists(postings_file):
            os.remove(postings_file)

        smallest_term_extractor = SmallestTermExtractor(
            self.load_intermediate_dictionaries()
        )

        while not smallest_term_extractor.is_empty():
            if self.should_free_memory():
                self.write_merged_block_to_disk(postings_file)

            (
                term,
                index_to_posting_list_metadata_mapping,
            ) = smallest_term_extractor.extract_smallest_term()

            same_term_posting_lists = [
                metadata.load_posting_list(f"{self.postings_file_prefix}{index}")
                for index, metadata in index_to_posting_list_metadata_mapping.items()
            ]

            ## merge all the posting lists of the same term
            merged_posting_list = same_term_posting_lists.pop().multiple_or_merge(
                *same_term_posting_lists
            )

            ## build skip pointers for posting list
            ## merged_posting_list.build_skip_pointers()

            self.index_posting_list(term=term, posting_list=merged_posting_list)

        ## write any remaining partial merged block to disk
        self.write_merged_block_to_disk(postings_file)

        return self.merged_dictionary

    def remove_intermediate_files(self):
        """Remove intermediate block files from disk"""
        for i in range(self.num_intermediate_blocks):
            intermediate_dictionary_file = f"{self.dictionary_file_prefix}{i}"

            if os.path.exists(intermediate_dictionary_file):
                os.remove(intermediate_dictionary_file)

            intermediate_postings_file = f"{self.postings_file_prefix}{i}"

            if os.path.exists(intermediate_postings_file):
                os.remove(intermediate_postings_file)

    def load_intermediate_dictionaries(
        self,
    ) -> Sequence[Sequence[Tuple[str, PostingListMetadata]]]:
        """Load intermediate dictionary files from disk"""
        intermediate_dictionaries = []

        for i in range(self.num_intermediate_blocks):
            with open(
                f"{self.dictionary_file_prefix}{i}",
                mode="rb",
            ) as f:
                intermediate_dictionaries.append(pickle.load(f))

        return intermediate_dictionaries


class Searcher:
    def __init__(
        self, dictionary: Dictionary, postings_file: str, preprocessor: Preprocessor
    ):
        self.dictionary: Dictionary = dictionary
        self.postings_file: str = postings_file
        self.preprocessor: Preprocessor = preprocessor

    ## (Not used)
    ## Ranking by date
    """
    def date_filter(self, query: str) -> Tuple[Optional[Dict], str]:
        ## This only for VSM QUERIES
        range_years = 4
        date_dict = {}
        has_dates = False
        words = []
        for word in query.split(" "):
            ## Check if its a year or not
            if not word.isdigit() or not len(word) == 4:
                words.append(word)
                continue

            query_year = int(word)
            has_dates = True

            for nearby in range(-range_years, range_years + 1):
                # Get year range
                nett_year = query_year + nearby
                year_weightage = self.date_weightage(nearby)
                year_posting_list = self.getDocId(
                    dictionary.get_posting_list(str(nett_year), postings_file)
                )
                for docId in year_posting_list:
                    # If docId present add the boosted value
                    date_dict[docId] = date_dict.get(docId, 0) + year_weightage

        return date_dict if has_dates else None, " ".join(words)

    def date_weightage(self, year_distance: int) -> int:
        ## Apply normalization based on mean 0 and sd 2
        # Use the below math formula to tune the values between 0 to 1
        final_score = math.exp(-0.5 * (year_distance / 2) ** 2)
        return final_score
    """

    ## (Not used)
    ## Co-occurence of synonyms is not used
    """
    def cooccurence(self, syn: str, original: str) -> int:
        # Geting original term and synonym posting list
        syn_posting_list = self.dictionary.get_posting_list(syn, self.postings_file)

        if not syn_posting_list:
            return 0

        original_posting_list = self.dictionary.get_posting_list(
            original, self.postings_file
        )

        relevancy_list = original_posting_list.and_merge(
            other_posting_list=syn_posting_list, is_multiword_merge=False
        )

        # Find difference in lengths of postings lists
        diff = len(original_posting_list.postings) - len(relevancy_list.postings)
        # if there is a common doc, +1 , else -1
        total_count = len(relevancy_list) - diff
        return abs(total_count)

    def get_synonyms(self, word: str) -> str:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                tokenized_syn = self.preprocessor.tokenize(l.name())
                if tokenized_syn:
                    # Push out one synonym at a time
                    yield self.preprocessor.tokenize(l.name())[0]

    def synonym_expansion(
        self,
        query: str,
        relev_docs: [int],
    ) -> {str: [str]}:
        words = {}
        seen = set({})
        nett_words = []
        for word in set(query.split(" ")):
            stemmed_word = self.preprocessor.tokenize(word)
            if stemmed_word:
                stemmed = stemmed_word[0]
            else:
                continue
            ## Store base word as key
            words[stemmed] = []
            ## Now we get the Synonyms, First filter out common words
            if self.dictionary.get_idf(stemmed) < 0.3:
                continue
            nett_words.append(stemmed)
            for syn in self.get_synonyms(word):
                if self.dictionary.get_idf(syn) < 0.5:
                    continue
                if syn != stemmed and syn not in seen:
                    ## Collect unique stemmed synonyms
                    seen.add(syn)
                    words[stemmed].append(syn)
        synonyms = {}
        rel_doc_presence = None
        ## Boolean checker to see if got relevant docs
        # reldocpresence = True if relevDocs else False
        ## Either one iteration or however many relevant docs are there
        for relev_doc in relev_docs if rel_doc_presence else [1]:
            for w in words:
                if not synonyms.get(w):
                    synonyms[w] = []
                for syn in words[w]:
                    if syn in synonyms[w]:
                        continue

                    # If the synonym can be found in the relevant docs provided by query, we add it in
                    if rel_doc_presence and self.is_syn_relevant(
                        [relev_doc], syn, self.dictionary, self.postings_file
                    ):
                        synonyms[w].append(syn)

                    elif not rel_doc_presence and self.dictionary.get_idf(syn) > 0.5:
                        if syn not in synonyms[w]:
                            ## Apply co occurence and get list of scores between syn and terms in query
                            vals = [
                                self.cooccurence(
                                    syn, original, self.dictionary, self.postings_file
                                )
                                for original in nett_words
                            ]
                            avg = sum(vals) / len(vals)
                            sd = sum([(v - avg) ** 2 for v in vals]) ** 0.5

                            if min(vals) < 500 and sd < 1500:
                                synonyms[w].append(syn)
        return synonyms

    def is_syn_relevant(
        self, doc: [int], syn: str, dictionary: Dictionary, postings_file: str
    ) -> bool:
        syn_posting_list = self.get_doc_id(
            dictionary.get_posting_list(syn, postings_file)
        )
        ## Add the and_merge here
        relevancy_list = self.and_merge(sorted(doc), sorted(syn_posting_list))
        return len(relevancy_list) > 0
    """

    def get_query_term_idf(self, term: str) -> float:
        """Get the sum of the idfs of the individual subterms"""
        return sum(self.dictionary.get_idf(subterm) for subterm in term.split())

    def get_query_subterms_and_posting_lists(
        self, term: str
    ) -> Iterable[Tuple[str, PostingList]]:
        """Retrieve all subterms of the query and their corresponding posting lists"""
        subterms_and_posting_lists = (
            (
                subterm,
                self.dictionary.get_posting_list(
                    term=subterm, postings_file=self.postings_file
                ),
            )
            for subterm in term.split()
        )

        result = [
            (subterm, posting_list)
            for subterm, posting_list in subterms_and_posting_lists
            if posting_list
        ]

        subterm_and_posting_list_queue = deque(result)

        i = 0

        while len(subterm_and_posting_list_queue) > 1:
            new_frontier = []

            while len(subterm_and_posting_list_queue) > 1:
                (
                    current_subterm,
                    current_posting_list,
                ) = subterm_and_posting_list_queue.popleft()
                next_subterm, next_posting_list = subterm_and_posting_list_queue[0]

                merged_subterm = " ".join(
                    current_subterm.split() + next_subterm.split()[i:]
                )

                merged_posting_list = current_posting_list.and_merge(
                    next_posting_list, is_multiword_merge=True
                )

                new_frontier.append((merged_subterm, merged_posting_list))

            result.extend(new_frontier)

            subterm_and_posting_list_queue = deque(new_frontier)
            i += 1

        return result

    def get_satisfy_boolean_query_doc_ids(self, tokens: Iterable[str]) -> Iterable[int]:
        """Get the set of document ids which satisfy the boolean query"""
        result: PostingList = None

        for token in tokens:
            subterms_and_posting_lists = self.get_query_subterms_and_posting_lists(
                term=token
            )

            if not subterms_and_posting_lists:
                return set()

            _, posting_list = subterms_and_posting_lists[-1]

            if result is None:
                result = posting_list
            else:
                result = result.and_merge(posting_list)

        if result is None:
            return set()

        return {posting.doc_id for posting in result.postings}

    def get_weighted_vector_for_relevant_docs(
        self, query_terms: [str], relevant_doc_ids: [int]
    ) -> Dict[str, float]:
        """Compute the normalized log weight tf vector for all relevant documents"""

        ## represents vector sum of document vectors
        term_to_weight_mapping = defaultdict(float)

        for term in query_terms:
            subterms_and_posting_lists = self.get_query_subterms_and_posting_lists(
                term=term
            )

            for subterm, posting_list in subterms_and_posting_lists:
                if not posting_list:
                    continue

                for posting in posting_list:
                    doc_id = posting.doc_id

                    if doc_id not in relevant_doc_ids:
                        continue

                    term_to_weight_mapping[subterm] += posting.normalized_tf_idf_weight

        num_of_relevant_docs = len(relevant_doc_ids)

        if num_of_relevant_docs > 0:
            for term in term_to_weight_mapping:
                term_to_weight_mapping[term] /= num_of_relevant_docs

        return dict(term_to_weight_mapping)

    def get_synonyms(self, query: str) -> Iterable[str]:
        tokens = self.preprocessor.tokenize(query, no_stemming=True)
        synonyms = set()

        ## iterate through each token and its corresponding part-of-speech (POS) tag
        for token, pos in nltk.pos_tag(tokens):
            if token.isnumeric():
                continue

            ## (Not used)
            ## Online API method doesn't yield as good of a result
            ## probably because it doesn't account for POS tagging
            """
            try:
                ## Online API method: retrieve synonyms from online API
                connection = HTTPSConnection(ONLINE_SYN_API_URL)
                connection.request(
                    "GET",
                    f"/associations/?entry={token}",
                    headers=ONLINE_SYN_API_REQUEST_HEADERS,
                )
                response = connection.getresponse()
                data = json.loads(response.read().decode("utf-8"))
                result = data.get("associations_array", [])

            except:
            """
            ## alternative: use wordnet
            synsets = wordnet.synsets(token, pos=POS_SET.get(pos[0] if pos else None))
            result = [synset.name().split(".")[0] for synset in synsets]

            synonyms.update(
                itertools.chain(
                    *(
                        self.preprocessor.tokenize(synonym)
                        for synonym in result[:SYN_THRESHOLD]
                    )
                )
            )

        ## exclude synonyms which already appeared in query
        return synonyms - set(
            itertools.chain(*(self.preprocessor.tokenize(token) for token in tokens))
        )

    def get_court_hierarchy_quality_score(self, doc_id: int, raw_score: float) -> float:
        """Compute the court hierarchy quality score"""
        coefficient = COURT_HIERARCHY_SCORE_FACTORS[
            self.dictionary.get_court_hierarchy_type(doc_id)
        ]
        return coefficient * raw_score

    def get_satisfy_boolean_query_quality_score(
        self,
        raw_score: float,
        did_satisfy_boolean_query: Iterable[int],
    ) -> float:
        """Compute the satisfy boolean query quality score"""
        coefficient = BOOLEAN_SCORE_FACTORS[did_satisfy_boolean_query]
        return coefficient * raw_score

    def find(
        self,
        query: str,
        relevant_doc_ids: Set[int],
        k: Optional[int],
        iteration_num: int = 1,
    ) -> Sequence[int]:
        """Return a list of the documents ranked according to relevance to the query"""
        ## tokenize the query
        parsed_query_tokens = self.preprocessor.parse_query(query)

        satisfy_boolean_query_doc_ids = (
            self.get_satisfy_boolean_query_doc_ids(set(parsed_query_tokens))
            if self.preprocessor.is_boolean_query(query)
            else set()
        )

        parsed_query_tokens.extend(self.get_synonyms(query))

        ## for keeping track of the accumulated score for each document
        doc_id_to_score_mapping = defaultdict(float)

        query_token_counts = Counter(parsed_query_tokens)

        relevant_doc_vector = (
            self.get_weighted_vector_for_relevant_docs(
                query_terms=parsed_query_tokens, relevant_doc_ids=relevant_doc_ids
            )
            if ROCCHIO_RELEVANCE
            else {}
        )

        for query_term, query_term_freq in query_token_counts.items():
            subterms_and_posting_lists = self.get_query_subterms_and_posting_lists(
                term=query_term
            )

            for subterm, posting_list in subterms_and_posting_lists:
                subterm_tf_idf_weight = Dictionary.compute_term_freq_weight(
                    query_term_freq
                ) * self.get_query_term_idf(subterm)

                if subterm_tf_idf_weight == 0:
                    continue

                modified_rocchio_weight = (
                    ROCCHIO_ALPHA * subterm_tf_idf_weight
                    + ROCCHIO_BETA * relevant_doc_vector.get(subterm, 0)
                )

                for posting in posting_list:
                    ## compute and store the score between query and document
                    doc_id_to_score_mapping[posting.doc_id] += (
                        posting.normalized_tf_idf_weight * modified_rocchio_weight
                    )

        ## sum cosine scores with various other quality scores
        ## scores are negated to sort doc ids by non-increasing scores
        score_doc_id_pairs = [
            (
                -(
                    score
                    + self.get_court_hierarchy_quality_score(
                        doc_id=doc_id, raw_score=score
                    )
                    + self.get_satisfy_boolean_query_quality_score(
                        raw_score=score,
                        did_satisfy_boolean_query=(
                            doc_id in satisfy_boolean_query_doc_ids
                        ),
                    )
                ),
                doc_id,
            )
            for doc_id, score in doc_id_to_score_mapping.items()
        ]

        n = len(score_doc_id_pairs)
        k = n if k is None else min(n, k)

        ## if relevant doc ids is provided on first iteration, do not perform pseudo
        ## relevance feedback
        if iteration_num == 1 and relevant_doc_ids:
            most_relevant_doc_ids = []
            remaining_doc_ids = []

            for _, doc_id in sorted(score_doc_id_pairs)[:k]:
                if doc_id in relevant_doc_ids:
                    most_relevant_doc_ids.append(doc_id)
                    relevant_doc_ids.discard(doc_id)
                else:
                    remaining_doc_ids.append(doc_id)

            ## any leftover relevant doc ids which were not in search
            ## result are also added to the front
            most_relevant_doc_ids.extend(sorted(relevant_doc_ids))

            return most_relevant_doc_ids + remaining_doc_ids

        ## for subsequent iterations or relevant doc ids is not provided at the start
        result = [doc_id for _, doc_id in sorted(score_doc_id_pairs)[:k]]

        if (
            PSEUDO_RELEVANCE_FEEDBACK
            and iteration_num < MAX_ROUNDS_OF_RELEVANCE_FEEDBACK
        ):
            ## recursively perform pseudo relevance feedback
            result = self.find(
                query=query,
                relevant_doc_ids=set(result[:NUM_TOP_DOCUMENTS_FOR_RELEVANCE_FEEDBACK]),
                k=k,
                iteration_num=iteration_num + 1,
            )

        return result


class InvertedIndex:
    def __init__(
        self,
        case_fold: bool = CASE_FOLD,
        stemming: bool = STEMMING,
        remove_stop_words: bool = REMOVE_STOP_WORDS,
        remove_punctuations: bool = REMOVE_PUNCTUATIONS,
        block_size_limit: int = BLOCK_SIZE_LIMIT,
    ):
        self.preprocessor = Preprocessor(
            case_fold=case_fold,
            stemming=stemming,
            remove_stop_words=remove_stop_words,
            remove_punctuations=remove_punctuations,
        )
        self.indexer = Indexer(block_size_limit=block_size_limit)

        self.dictionary: Dictionary = None

        self.keywords_to_relevant_doc_ids_mapping: Dict[frozenset, set] = defaultdict(
            set
        )

    def index(self, doc: Dict[str, str]):
        """Index the document"""
        corpus = " ".join([doc.get(TITLE), doc.get(CONTENT), doc.get(COURT)])

        ## extract court hierarchy information
        court = (
            doc.get(COURT).casefold() if self.preprocessor.case_fold else doc.get(COURT)
        )
        for hierarchy_type, courts in COURTS.items():
            if court in courts:
                court_hierarchy_type = hierarchy_type
                break
        else:
            court_hierarchy_type = CourtHierarchyType.DEFAULT

        corpus_tokens = self.preprocessor.tokenize(corpus)
        ## extract date information
        date_tokens = self.preprocessor.tokenize_date_string(doc.get(DATE_POSTED))

        self.indexer.index(
            tokens=corpus_tokens + date_tokens,
            doc_id=int(doc.get(DOCUMENT_ID)),
            court_hierarchy_type=court_hierarchy_type,
        )

    def build(self, postings_file: str):
        """Build the dictionary from the postings file"""
        self.dictionary = self.indexer.merge(postings_file)
        self.indexer.clean_up()

    def export(self, dictionary_file: str):
        """Export the dictionary to the dictionary file"""
        with open(dictionary_file, mode="wb") as f:
            pickle.dump(self.dictionary, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, dictionary_file: str) -> "InvertedIndex":
        """Load the dictionary from the dictionary file"""

        with open(dictionary_file, mode="rb") as f:
            self.dictionary = pickle.load(f)

    def train_search(self, training_data: Iterable[Tuple[str, Iterable[int]]]):
        """Save keywords and its relevant document ids to aid in future searching"""
        for query, relevant_doc_ids in training_data:
            keywords = frozenset(self.preprocessor.tokenize(query))

            if not keywords:
                continue

            self.keywords_to_relevant_doc_ids_mapping[keywords].update(relevant_doc_ids)

    def search(
        self,
        query: str,
        postings_file: str,
        relevant_doc_ids: Set[int],
        k: Optional[int] = None,
    ) -> Sequence[int]:
        """Return a ranked list of the (optional top K most) relevant documents"""
        ## retrieve pre-trained relevant doc ids if any
        tokens = frozenset(self.preprocessor.tokenize(query))
        for keywords, doc_ids in self.keywords_to_relevant_doc_ids_mapping.items():
            common_terms = keywords & tokens

            ## if the ratio of common terms to keywords exceed the threshold
            if len(common_terms) / len(keywords) >= KEYWORDS_MATCH_THRESHOLD:
                relevant_doc_ids.update(doc_ids)

        searcher = Searcher(
            dictionary=self.dictionary,
            postings_file=postings_file,
            preprocessor=self.preprocessor,
        )

        ## perform search and return result
        return searcher.find(
            query=query,
            relevant_doc_ids=relevant_doc_ids,
            k=k,
        )
