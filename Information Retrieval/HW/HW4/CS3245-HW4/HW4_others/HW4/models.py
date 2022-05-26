import pickle
import math
import functools
from typing import Optional, Dict, Sequence, Tuple
from enum import Enum, auto


def log_freq_weighting_scheme(term_freq: int) -> float:
    """Calculate the logarithmic term frequency"""
    return (1 + math.log10(term_freq)) if term_freq > 0 else 0


def idf_weighting_scheme(collection_size: int, doc_freq: int) -> float:
    """Calculate a term's idf given the collection size and document frequency of the term"""
    return (
        math.log10(collection_size / doc_freq)
        if collection_size > 0 and doc_freq > 0
        else 0
    )


def normalization_scheme(doc_length: float) -> float:
    """Calculate the normalization factor given document length"""
    return (1 / doc_length) if doc_length != 0 else 0


class CourtHierarchyType(Enum):
    MOST_IMPORTANT = auto()
    IMPORTANT = auto()
    DEFAULT = auto()


class Posting:
    def __init__(
        self,
        doc_id: int,
        term_freq: int,
        normalized_tf_idf_weight: float,
        positions: Sequence[int],
    ):
        self.doc_id = doc_id
        self.term_freq = term_freq
        self.normalized_tf_idf_weight = normalized_tf_idf_weight
        self.positions = positions

    def __repr__(self):
        return f"{self.doc_id}"


class PostingList:
    def __init__(self):
        self.postings: Sequence[Posting] = []
        self._current = 0

    def __len__(self):
        return len(self.postings)

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current >= len(self):
            self._current = 0
            raise StopIteration

        self._current += 1

        return self.postings[self._current - 1]

    def __repr__(self):
        return repr(self.postings)

    def add(
        self,
        doc_id: int,
        term_freq: int,
        normalized_tf_idf_weight: float,
        positions: Sequence[int],
    ):
        """
        Append a new posting to the posting list
        """

        self.postings.append(
            Posting(
                doc_id=doc_id,
                term_freq=term_freq,
                normalized_tf_idf_weight=normalized_tf_idf_weight,
                positions=positions,
            )
        )

    def or_merge(self, other_posting_list: "PostingList") -> "PostingList":
        """
        Return a new posting list containing the union of all the doc ids
        in this posting list and the other posting list
        """

        merged_posting_list = PostingList()

        current_posting_index = 0
        other_posting_index = 0

        while current_posting_index < len(self) and other_posting_index < len(
            other_posting_list
        ):
            current_posting = self.postings[current_posting_index]
            other_posting = other_posting_list.postings[other_posting_index]

            if current_posting.doc_id <= other_posting.doc_id:
                if current_posting.doc_id == other_posting.doc_id:
                    other_posting_index += 1

                merged_posting_list.add(
                    doc_id=current_posting.doc_id,
                    term_freq=current_posting.term_freq,
                    normalized_tf_idf_weight=current_posting.normalized_tf_idf_weight,
                    positions=current_posting.positions,
                )
                current_posting_index += 1
            else:
                merged_posting_list.add(
                    doc_id=other_posting.doc_id,
                    term_freq=other_posting.term_freq,
                    normalized_tf_idf_weight=other_posting.normalized_tf_idf_weight,
                    positions=other_posting.positions,
                )
                other_posting_index += 1

        ## append any remaining postings in either list
        while current_posting_index < len(self):
            current_posting = self.postings[current_posting_index]
            merged_posting_list.add(
                doc_id=current_posting.doc_id,
                term_freq=current_posting.term_freq,
                normalized_tf_idf_weight=current_posting.normalized_tf_idf_weight,
                positions=current_posting.positions,
            )
            current_posting_index += 1

        while other_posting_index < len(other_posting_list):
            other_posting = other_posting_list.postings[other_posting_index]
            merged_posting_list.add(
                doc_id=other_posting.doc_id,
                term_freq=other_posting.term_freq,
                normalized_tf_idf_weight=other_posting.normalized_tf_idf_weight,
                positions=other_posting.positions,
            )
            other_posting_index += 1

        return merged_posting_list

    def multiple_or_merge(
        self, *other_posting_lists: "PostingList"
    ) -> Optional["PostingList"]:
        """
        Return a new posting list containing the union of all the doc ids
        in this posting list and other posting lists
        """

        stack = list(other_posting_lists)

        if not stack:
            return self

        while len(stack) > 1:
            stack.append(stack.pop().or_merge(stack.pop()))

        return self.or_merge(stack.pop())

    def and_merge(
        self, other_posting_list: "PostingList", is_multiword_merge: bool = False
    ) -> "PostingList":
        """
        Returns a new posting list containing the intersection of all the doc ids
        in this posting list and the other posting list.
        By default, term frequency and tf-idf will be set to 0 since merging posting
        lists would invalidate these values.

        However, if the merge is a multi-word merge for 2 terms, for every matching
        document id which contains both terms, it checks the list of positions to
        ensure that the terms are adjacent. The multiword positions, term frequency
        tf-idf scores will be set to the relevant values.
        """

        merged_posting_list = PostingList()

        current_posting_index = 0
        other_posting_index = 0

        ## Iterate through both postings lists
        while current_posting_index < len(self) and other_posting_index < len(
            other_posting_list
        ):
            current_posting = self.postings[current_posting_index]
            other_posting = other_posting_list.postings[other_posting_index]

            if current_posting.doc_id < other_posting.doc_id:
                current_posting_index += 1

            elif current_posting.doc_id > other_posting.doc_id:
                other_posting_index += 1

            else:
                current_posting_index += 1
                other_posting_index += 1

                ## If not multiword merge, only need to add doc id to posting list
                if not is_multiword_merge:
                    merged_posting_list.add(
                        doc_id=current_posting.doc_id,
                        term_freq=0,
                        normalized_tf_idf_weight=0,
                        positions=[],
                    )
                    continue

                ## Multiword merge: retrieve positions where terms are adjacent
                other_posting_positions = set(other_posting.positions)
                new_positions = [
                    current_posting_position
                    for current_posting_position in current_posting.positions
                    if current_posting_position + 1 in other_posting_positions
                ]

                if len(new_positions) == 0:
                    continue

                merged_posting_list.add(
                    doc_id=current_posting.doc_id,
                    term_freq=len(new_positions),
                    normalized_tf_idf_weight=(
                        current_posting.normalized_tf_idf_weight
                        + other_posting.normalized_tf_idf_weight
                    ),
                    positions=new_positions,
                )

        return merged_posting_list


class PostingListMetadata:
    def __init__(self, doc_freq: int, disk_position_offset: int, disk_data_length: int):
        self.doc_freq = doc_freq
        self.disk_position_offset = disk_position_offset
        self.disk_data_length = disk_data_length

    def load_posting_list(self, postings_file: str) -> PostingList:
        """Load posting list from disk"""
        with open(postings_file, mode="rb") as f:
            f.seek(self.disk_position_offset)
            pickled_posting_list = f.read(self.disk_data_length)

        return pickle.loads(pickled_posting_list)


class Dictionary:
    compute_idf_weight = idf_weighting_scheme
    compute_term_freq_weight = log_freq_weighting_scheme
    compute_normalization_factor = normalization_scheme

    def __init__(
        self,
        collection_size: int = 0,
        doc_id_to_court_hierarchy_type_mapping: Dict[int, CourtHierarchyType] = {},
    ):
        self.collection_size = collection_size
        self.term_to_posting_list_metadata_mapping: Dict[str, PostingListMetadata] = {}
        self.doc_id_to_court_hierarchy_type_mapping: Dict[
            int, CourtHierarchyType
        ] = doc_id_to_court_hierarchy_type_mapping

    def add(self, term: str, posting_list_metadata: PostingListMetadata):
        """Add a dictionary entry of term -> posting list metadata"""
        self.term_to_posting_list_metadata_mapping[term] = posting_list_metadata

    def get_posting_list(self, term: str, postings_file: str) -> Optional[PostingList]:
        """Return a posting list containing all the doc ids with the given term"""

        posting_list_metadata = self.term_to_posting_list_metadata_mapping.get(term)

        if not posting_list_metadata:
            return

        return posting_list_metadata.load_posting_list(postings_file)

    def get_idf(self, term: str) -> float:
        """Compute the idf of given term"""
        posting_list_metadata = self.term_to_posting_list_metadata_mapping.get(term)

        if not posting_list_metadata:
            return 0

        return Dictionary.compute_idf_weight(
            collection_size=self.collection_size,
            doc_freq=posting_list_metadata.doc_freq,
        )

    def get_court_hierarchy_type(self, doc_id: int) -> CourtHierarchyType:
        return self.doc_id_to_court_hierarchy_type_mapping.get(
            doc_id, CourtHierarchyType.DEFAULT
        )

    def to_sorted_list(self) -> Sequence[Tuple[str, PostingListMetadata]]:
        """
        Return the sorted entries of the dictionary in the form of [(term, metadata), ...]
        """
        return sorted(
            self.term_to_posting_list_metadata_mapping.items(), key=lambda pair: pair[0]
        )
