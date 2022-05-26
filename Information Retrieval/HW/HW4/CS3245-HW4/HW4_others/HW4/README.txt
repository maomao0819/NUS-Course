This is the README file for A0136134N-A0190363H-A0187836L-A0188693H submission.
Emails:
A0136134N: e0272509@u.nus.edu
A0190363H: e0325947@u.nus.edu
A0187836L: e0323420@u.nus.edu
A0188693H: e0324277@u.nus.edu 

== Python Version ==

We're using Python Version 3.9.1 for this assignment.

== General Notes about this assignment ==

This program implements indexing and searching techniques using the Vector Space Model.
It accepts both free text and boolean queries, including phrasal queries in double quotation
marks, ranked according to relevance to the query.

Note:
- All logarithmic calculations are done with log base 10.
- lnc.ltc weighting scheme is used for the computation of relevance score.
- This program uses the newer Porter2 stemming algorithm which is widely considered to be
better than the original Porter.
- This program includes a set of customisable configurations (E.g. whether to case-fold,
remove punctuation, etc). The current configurations are obtained through trial & error testing
and common heuristics. Feel free to edit the configs to examine the changes to predictive power.


This program can be described as 3-step process:
1. Preprocessing documents and extracting the terms and their positions.
2. Indexing each term and its relevant information (term frequency, positions etc) with the associated document id.
3. Querying the inverted index to obtain the documents, ranked by relevance to the query.

Outline of each step:

1. Preprocessing documents and extracting the terms and their corresponding frequencies:
- This step of the pipeline is handled by the Preprocessor class.
- Tokenization, case-folding, stemming, punctuation and stopword removal are carried out to extract the terms and
their corresponding frequencies in the documents.

2. Indexing tokens with associated document id:
- This step is handled by the Indexer class.
- For each iterm, the tf-idf score is calculated using the logarithmic frequency weighting scheme, 
with no idf, and normalized with the euclidean document length.
- A posting containing the document id, term frequency, normalized tf-idf score and list of
term positions in the document is then added to the term's posting list.
    - We augmented each posting with a list of its term positions in the document to handle
    phrasal queries.
- The indexer also maintains a count of the total number of documents in the collection during indexing.

- The indexer carries out Single-pass in-memory Indexing (SPIMI)
- The tokens and their associated document ids are first indexed in memory.
- The indexer maintains a maximum block size of 1000000 (combined postings). It generates separate
intermediate dictionaries/postings for each block, which are then written into separate files
when the block is full. 
- After all the tokens from all the document ids have been processed, there should be
multiple intermediate dictionary and posting files. The indexer merges the intermediate
dictionaries into a main dictionary, and also merges the posting lists, which are then
written to the disk as the final postings file.
    - For each term: 
        - Retrieve the posting lists from all the intermediate dictionaries.
        - Merge the posting lists from each dictionary into 1 merged posting list.
        - Add the posting list to the merged dictionary.
    - The merged posting lists are written to the specified output file.
    - The intermediate files are removed.


3. Querying the inverted index to return documents ranked by relevance to query:
- This step is handled by the Searcher class.
- The program accepts both free text queries and boolean queries, with space-separated terms or 
phrasal queries in double quotes.
- Each query is processed to extract the query terms and their corresponding frequencies in the query.
The process is similar to how terms are extracted from the documents during the indexing phase 
(tokenization, stemming, case folding and punctuation removal). 
    - For each phrasal query, the individual terms are case-folded, stemmed etc, and the phrase is preserved
    in a single space-separated string.

- For each query term:
    - The query term's tf-idf score is calculated using the logarithmic frequency weighting scheme with idf.
    - If relevance feedback is enabled by setting the ROCCHIO_RELEVANCE flag as True, the query vector is modified
    using the relevant document query vector using the Rocchio algorithm.
        - The weights for the original query (alpha) and relevant document vector (beta) are 1.0 and 0.75 respectively.
    - For phrasal queries:
        - The postings list for the phrase is obtained by merging the posting lists of each subterm iteratively, 
        and checking the positions of each subterm to ensure that they are adjacent.
        - We include scoring for subterms (for example: for phrase “A B C”, we also score “A”, “B”, “C”, “A B” and “B C”) 
        as the subterms may also be of interest and therefore relevant to the user.
        - The idf score for the phrase is calculated as a simple sum of the idf scores for each of its subterms.

- The cosine similarity score is computed as the product of the normalized tf-idf score of the term in 
the document and tf-idf score of the term in the query.
    - Note that we do not normalize the tf-idf score of the term in the query as normalization will not change the relative
    ordering of the documents' accumulated scores since the normalization factor for a particular query is constant.
- The document’s score is modified by several factors:
    - The document’s score is boosted by a court hierarchy quality score, based on the classification of the 
    different courts (0.2 for most important, 0.1 for important, and 0 for default).  
    - If the query is a boolean query, we extract the set of document ids which contain all the boolean query terms.
    If the document belongs to this set, the document’s score is boosted by a factor of 0.15. 


- The document ids are sorted by score and returned as the ranked relevant documents for the query.
- If a query contains only terms which appear in all documents in the collection, then according to the
lnc.ltc weighting scheme, all the documents will have a score of 0. In this case, we assume all the documents
are equally irrelevant and trivial, and the searcher will return an empty result.
- If pseudo relevance feedback is activated through the PSEUDO_RELEVANCE_FEEDBACK flag, another round of searching 
will be carried out using the top 5 doc ids as the relevant documents.

Additional techniques/implementations to improve the scoring are documented in the BONUS.docx attached in this submission.

Allocation of work:
A0136134N: Indexing, searching, query expansion by synonym and integration
A0190363H: Positional and biword index, phrasal queries and integration
A0187836L: Date weights and synonym co-occurence
A0188693H: Rocchio algorithm and relevance feedback

== Files included with this submission ==

- index.py: Contains the source code to build the inverted index. 
- search.py: Contains the source code to run searches on the inverted index.
- invertedindex.py: Contains the source code for the InvertedIndex and helper classes.
- models.py: Contains the the source code for the data classes. E.g. Posting, PostingList, 
PostingListMetadata and Dictionary classes.
- postings.txt: Contains the pickled postings.
- dictionary.txt: Contains the pickled invertedindex/dictionary.
- README.md: Contains the overview of the assignment submission.
- BONUS.docx: Contains a description of additional techniques and experiments conducted to improve
the performance of the system.

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] We, A0136134N-A0190363H-A0187836L-A0188693H, certify 
that we have followed the CS 3245 Information Retrieval class guidelines
for homework assignments.  In particular, we expressly vow that we have 
followed the Facebook rule in discussing with others in doing the assignment
and did not take notes (digital or printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

