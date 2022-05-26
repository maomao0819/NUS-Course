This is the README file for A0248812B's submission
e0934074@u.nus.edu

== Python Version ==

I'm (We're) using Python Version 3.8.10 for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

For building the language models, I first process the sentences with removed numbers, duplicated spaces, and non-alphabet text, 
and cast them to lower case. Afterward, with going through the sentence and fetching every 4 letters with a sliding window, I get 
lots of tuples containing 4 separate characters and collect the 4-grams from the string. Then, I store them as the key in the dictionary 
with certain labels. After calculating the occurrences of each tuple, I create the zero entries with the tuples that exist in the dictionary 
with other labels. In the end, doing add 1 smoothing and change counts to probabilities by dividing the total counts in the certain 
dictionary with corresponding keys.

For testing the language models, get the probabilities in the language models with corresponding entries and multiply the 
probabilities of the 4-grams for the string. Because of the imitation of the float, using the sum of logged value instead of  
multiplying together. Consequently, return the label that gives the highest values. Ignore the four-gram if it is not found 
in the language models, that is, skipping them when the tuples don't exist in the language models. If there are high probabilities
and proportion that tuples in the sentence is not found in the language models, it might be other languages. Thus, I use the 
proportion of tuples that don't show up in training data to the length of the sentence to tell the other label.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

build_test_LM.py: build the language models from training data and write its predictions for testing data to a file.
CS3245-hw1-check.sh: for checking.
Essay questions.txt: contains my answers to the essay questions.
eval.py: to evaluate the accuracy of the predictions.
input.correct.txt: a file containing the correct string labels for the sample test.
input.predict.txt: a file stores the predictions for the sample test.
input.test.txt: a file containing a list of strings to test the language models for sample test.
input.train.txt: a file that contains a list of strings with their labels to build the ngram language models for the sample test.
README.txt: a text-only file that describes relative information.

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I, A0248812B, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0248812B, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

create empty dictionary: https://pythonguides.com/how-to-create-an-empty-python-dictionary/
create ngrams: https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
string to separate char tuple: https://stackoverflow.com/questions/16449184/converting-string-to-tuple-without-splitting-characters
check keys if exist in the dictionary: https://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary
sum values in the dictionary: https://www.kite.com/python/answers/how-to-sum-the-values-in-a-dictionary-in-python
argmax of the dictionary: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
write files: https://www.w3schools.com/python/python_file_write.asp