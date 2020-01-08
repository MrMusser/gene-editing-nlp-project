This project uses MOST of the code used to generate the analysis for my
research paper titled "An Examination of Expert Discourse on Gene Editing
Using Natural Language Processing." A few lengthy functions are stored in
the file "functions.py," primarily those which required reasonably lengthy
dictionaries in order to function. Specifically, the three functions
in functions.py were used to convert all words occurring in a dictionary
of U.K. English terms to their American English equivalents, to remove
a pre-set list of stop words, and to remove lemmatized sentences from the
corpus if they were useless. (For instance, some documents contained headers
on each page with the sentence "All Rights Reserved"; this sentence and any
other accidental inclusion that occurred more than five times were removed
from the text used for analysis.)

The file "full project code.py" contains the remaining code used to perform
the analysis. Note the following slight modifications between the code actually
used and the code in "full project code.py": the section on preprocessing text
in the publicly accessible file has been written to make clear the steps used
in text preprocessing. However, the actual code used was structured differently
in order to make better use of list comprehensions in Python. This means that
different code was written almost entirely from scratch for different purposes
(such as preprocessing a single document vs. preprocessing and merging a list
of documents). The code in "full project code.py" instead uses unique
functions for multiple different stages of preprocessing in order to parse the
logic into single-use functions. Using the functions contained in "full project
code.py" should result in the same outputted text as the code actually used,
but because it uses functions iteratively instead of making full use of list
comprehensions, the preprocessing time would take substantially longer.

Also included in this folder are a .txt file listing all sources used in this
analysis, a folder containing the .txt files of government commission
reports used in this analysis, a folder containing .csv files with metadata
on all reports and news articles, a folder containing the outputted
figures created using the code in "full project code.py", and a folder
containing the predicted sentiments of all sentences using each of the four
sentiment classifiers discussed in the paper. Texts of books by
ethicists, books by scientists, and news articles have not been placed in
this publicly accessible folder for copyright reasons.

Any questions or inquiries can be directed to mrm311@georgetown.edu.
