import pickle


class Term:

    def __init__(self):
        self.disk_position_offset = 0
        self.disk_data_length = 0
        self.normalized_tf_idf_weight = {}
        self.document_frequency = 0
        self.inverse_document_frequency = 0

    def load_posting_list(self, postings_file):
        """load postings list from disk"""

        with open(postings_file, mode="rb") as f:
            f.seek(self.disk_position_offset)
            pickled_posting_list = f.read(self.disk_data_length)

        return pickle.loads(pickled_posting_list)


class Dictionary:

    def __init__(self, collection_size=None, dictionary=None):
        self.dictionary = dictionary if dictionary else {}
        self.collection_size = collection_size
        self.corpus_disk_data_length = 0

    def load(self, src):
        """load Dictionary from pickle file"""

        with open(src, 'rb') as handle:
            new_dictionary = pickle.load(handle)
            self.__init__(
                collection_size=new_dictionary.collection_size,
                dictionary=new_dictionary.dictionary
            )

    def save(self, dest):
        """save Dictionary as pickle file"""

        with open(dest, 'wb') as dictionary_file_writer:
            pickle.dump(self, dictionary_file_writer, protocol=pickle.HIGHEST_PROTOCOL)

    def add_document_term_and_frequency(self, doc, term, normalized_tf_idf_weight):
        """add the term to the dictionary"""

        term_metadata = self.dictionary.get(term, Term())
        term_metadata.document_frequency += 1
        term_metadata.normalized_tf_idf_weight[doc] = normalized_tf_idf_weight

        self.dictionary[term] = term_metadata

    def get_normalized_tf_idf_weight(self, term, doc):
        """get the normalized tf-idf weight"""

        if term in self.dictionary:
            return self.dictionary[term].normalized_tf_idf_weight[doc]
        return 0

    def get_inverse_doc_frequency(self, term):
        """get inverse doc frequency for term"""

        if term in self.dictionary:
            return self.dictionary[term].inverse_document_frequency
        return -1

    def get_posting_list(self, term, postings_file):
        """return a postings list for a term if exists, else return empty list"""

        if term in self.dictionary:
            return self.dictionary[term].load_posting_list(postings_file=postings_file)

        return []


class Postings:

    def __init__(self, corpus_postings_list):
        self.postings_list_dict = {}
        self.corpus_postings_list = corpus_postings_list

    def add_document_to_postings_list(self, doc, term):
        """add document to postings list for given term"""

        postings_list = self.postings_list_dict.get(term, list())
        postings_list.append(doc)
        self.postings_list_dict[term] = postings_list
