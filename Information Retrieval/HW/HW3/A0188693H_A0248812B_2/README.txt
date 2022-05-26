This is the README file for A0188693H_A0248812B's submission
Email(s):
A0188693H - e0324277@u.nus.edu
A0248812B - e0934074@u.nus.edu

== Python Version ==

We're using Python Version 3.8.10 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

For index.py,
1. tokenize corpus
    1.1 split corpus using nltk.tokenize, first into sentences and then into words
    1.2 process tokens by casefolding, stemming and then removing common/ dispensable punctuation
        1.2.1 casefold by using python string.lower()
        1.2.2 stemming done using nltk PorterStemmer
        1.2.3 punctuation generated from python string.punctuation
2. for each document
    2.1 euclidean_doc_length = (summation of all terms (log_term_freq_weighting_scheme**2))**0.5
    2.2 normalized_tf_idf_weight= log_freq_weighting_scheme/euclidean_doc_length
    2.3 for every unique token
        2.1.1 calculate the normalized tf-idf weight
        2.1.2 add document id to respective postings list in inverted index
    2.4 calculate inverse document frequency
3. after processing entire corpus, save the inverted index in Dictionary object hierarchy and save the postings list data


For searching part, we first tokenize the queries.
For each query, we calculate the weights by tf_idf with the term frequency and the documents frequency.
Afterward, for each document, we collect the term frequency, multiply with the weights of query part, and divided by the vector length which stored in the terms of the dictionary.
Consequently, we get the score, add the score to the previous score which already stored and store the value with its documents ID.
After we collect the score, using heap to retrieve top 10 values and write them into the file.


== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

index.py: the index system for telling the relationship and information between documents and the terms and generating the dictionary.txt and the .ostings.txt
search.py: Getting the top 10 relevant documents with the free text queries. Tokenizing the queries first and scoring the relevant documents with lnc.ltc and cosine, and the score s are related the information in dictionary.txt and postings.txt. Finally, returning the top 10 relevant documents after scoring document.
model.py: define the classes of data types such as the term, posting list and the dictionary (inverted index)
utils.py: define the core mechanism and assisting functions
dictionary.txt: the information of dictionary and the terms information. Its keys are the certain term and the corresponding value is the term information like docuent frequency.
postings.txt: the information of postings lists and the tuple in each posting represents (doc ID, term freq).

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] We, A0188693H_A0248812B, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

We suggest that we should be graded as follows:

index.py - A0188693H
search.py - A0248812B

== References ==

sort dict by keys: https://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key
finding top k largest keys in a dictionary python: https://stackoverflow.com/questions/12266617/finding-top-k-largest-keys-in-a-dictionary-python
