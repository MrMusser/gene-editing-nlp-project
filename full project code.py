'''
Analyzing Public Discourse on Gene Editing Using Natural Language Processing
Complete Project Code

Contents in Brief

1. Begin by plotting rate of article publication by news source

2. Preprocess all text
    2.A. Write functions to preprocess text
    2.B. For each author type (plus commission types), create a file containing all preprocessed sentences
    2.C. For each author type, create a file continaing all individual, lemmatized words
    2.D. For each article, create an individual file containing the preprocessed sentences of the article

3. Build a classifier to predict text type using files of preprocessed sentences
    3.A. Build a function to preprocess and predict the author type of an input sentence
    3.B. Plot a confusion matrix showing the accuracy of the predictor

4. Examine and graph word frequencies using files of individual, lemmatized words
    4.A. Plot a graph to show Zipfian distribution of word frequencies
    4.B. Plot a graph to show frequncies of the 15 most common words by author type
    4.C. Build a function to randomly sample text and count for keywords
    4.D. Plot a graph showing the frequency distributions of sampling for keywords
    4.E. Count up individual occurrences of rare, significant words and build table

5. Build a sentiment analyzer using 5000 sentences hand-labeled by sentiment
    5.A. Fine-tune parameters as much as possible
    5.B. Plot confusion matrices for each of four model types
    5.C. Predict labels for all sentences and store in separate .txt files for easy access
    5.D. Predict label probabilities for all sentences and store in separate .txt files for easy access
    5.E. Repeat previous two steps for each individual news article and store in separate .txt files
    5.F. Repeat previous two steps for different government commissions

6. Use sentiment analyzer to examine the sentiment of different text types
    6.A. Build a table to list occurrence of each label by author type and model
    6.B. Plot distributions of label probabilities by author type and model
    6.C. Use label probabilities of individual articles to plot changes in sentiment over time
    6.D. Use list of articles by publication to examine sentiment distributions by publication
    6.E. Build a function to search articles for keywords
    6.F. Use label probabilities of individual articles to compare sentiment of articles with or without keywords
    6.G. Use .csv file of commission reports to examine sentiment by region, year
'''

import pandas as pd
import numpy as np
import re
import os
import random

import nltk
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import dates as dates
from matplotlib import gridspec
from matplotlib import font_manager
import seaborn as sns

import dateutil
from datetime import datetime

import functions

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, svm, linear_model, ensemble
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import linregress
from statsmodels.stats.multicomp import pairwise_tukeyhsd

matplotlib.rcParams['font.family'] = 'Palatino'
matplotlib.rcParams['font.size'] = 12



'''
Part 1: Begin by plotting rate of article publication by news source
'''

#First open .csv file containing information about each file
meta_news = pd.read_csv('CSV and Excel Spreadsheets/media_meta.csv')

#Then add columns and iterate by row to list total articles from each publication by the time of each new article
sources = {'Boston Globe': 'bg', 'Wall Street Journal': 'wsj', 'Financial Times': 'ft',
            'New York Times': 'nyt', 'Washington Post': 'wp', 'The Economist': 'econ',
            'The Guardian': 'guard', 'MIT Technology Review': 'mit', 'Los Angeles Times': 'lat'}
for source, abbr in sources.items():
    meta_news['num_{}'.format(abbr)] = 0
    for i in range(len(meta_news)):
        if meta_news['publication'][i] == source:
            meta_news.copy()
            meta_news['num_{}'.format(abbr)][i:] += 1

#Create a graph
sns.set_palette('Blues_r')
fig = plt.figure(figsize=(10,3.5))
ax0 = plt.subplot(1,1,1)

#Format x values and create y values from the columns created in the previous step in order to build a stacked line graph
date_vals = [dateutil.parser.parse(i) for i in meta_news['date']]
x_vals = dates.date2num(date_vals)
y = [meta_news['num_{}'.format(abbr)] for abbr in ['bg', 'wsj', 'ft', 'nyt', 'wp', 'econ', 'guard', 'mit', 'lat']]

#Create x ticks and tick labels for stacked line graph
label_ticks = ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
x_labels = ['Jan. 2013', 'Jan. 2014', 'Jan. 2015', 'Jan. 2016', 'Jan. 2017', 'Jan. 2018', 'Jan. 2019', 'Jan. 2020']
x_ticks = dates.date2num([dateutil.parser.parse(i) for i in label_ticks])

#Plot stacked line graph, format with x-tick labels, title, and font settings
ax0.stackplot(date_vals, *y)
ax0.legend(['Boston Globe', 'Wall Street Journal', 'Financial Times', 'New York Times', 'Washington Post',
           'The Economist', 'The Guardian', 'MIT Technology Review', 'Los Angeles Times'], loc=2,
           prop={'size': 12}, edgecolor='black')
ax0.set_xticks(x_ticks)
ax0.set_xticklabels(x_labels, ha='center')
ax0.set_yticklabels([0, 200, 400, 600, 800])
ax0.set_ylabel('Number of Publications', fontdict={'fontsize': 16})
ax0.set_xlabel('Date', fontdict={'fontsize': 16})
ax0.set_title('News Articles on CRISPR Over Time', fontdict={'fontsize':20})

#Create annotations
#First create a list of significant dates to be marked with vertical dashed lines
date_list = ['2015-04-22', '2015-12-01', '2016-02-01', '2016-11-15', '2017-02-14', '2017-07-25',
             '2018-07-17', '2018-11-26', '2019-03-03']
#(Also make a list of somewhat offset date for prettier annotation——these will be associated with numbers to label the dashed lines)
date_list_offset = ['2015-03-22', '2015-10-20', '2016-01-14', '2016-10-15', '2017-01-14', '2017-06-25',
             '2018-06-17', '2018-10-26', '2019-02-03']

#Format the above lists to match x-value notation
date_list = [datetime.strptime(d, "%Y-%m-%d") for d in date_list]
date_list_offset = [datetime.strptime(d, "%Y-%m-%d") for d in date_list_offset]

#Add annotations using formatted x-values
for i in range(len(date_list)):
    ax0.annotate('({})'.format(i+1), xy=(date_list_offset[i], 850), fontsize=12)
    ax0.axvline(date_list[i], ymax=0.85, linestyle='--', linewidth=0.5, color='black')

#Save and show figure
plt.savefig('Figures/news_timeline.jpg', dpi=300, bbox_inches='tight')
plt.show()



'''
    Part 2: Preprocess all text
    2.A. Write functions to preprocess text
    2.B. For each author type (plus commission types), create a file containing all preprocessed sentences
    2.C. For each author type, create a file continaing all individual, lemmatized words
    2.D. For each article, create an individual file containing the preprocessed sentences of the article
'''

lem = WordNetLemmatizer()

def preprocess_sentence(sentence):

    #First clean by removing punctuation, lower-casing, americanizing all words
    sentence = re.sub('\d|\W+', ' ', sentence).lower()
    sentence = functions.americanize(sentence)

    #Then pos-tag and lemmatize the text (use convert_part_of_speech_tag to ensure pos_tag in correct form for interpretation)
    words = nltk.word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(words)
    words_lemmatized = [lem.lemmatize(word, functions.convert_part_of_speech_tag(pos)) for word, pos in pos_tagged]

    #Remove stopwords and rejoin the sentence
    words_unstopped = functions.remove_stopwords(words_lemmatized)
    joined_sentence = ' '.join(words_unstopped)

    return joined_sentence


def preprocess_list_of_sentences(sentences, return_words=False):
    new_sentences = [preprocess_sentence(sentence) for sentence in sentences]
    without_false_sentences = functions.remove_false_sentences(new_sentences)
    if return_words==False:
        return without_false_sentences

    #Add a function to return a new file containing a list of all words in the sentences
    if return_words==True:
        word_list = []
        for sentence in without_false_sentences:
            for word in sentence:
                word_list.append(word)
        return without_false_sentences, word_list


def preprocess_and_return_file(file):

    #First open the corresponding file and preprocess it
    f = open(file).read()
    sentences = nltk.sent_tokenize(f)
    sentences_clean = preprocess_list_of_sentences(sentences)

    #Then open a new text file and copy over the preprocessed sentences on unique lines
    g = open('{}_preprocessed.txt'.format(file[:-4]), 'w+')
    g.writelines('%s\n' % sentence for sentence in sentences_clean)
    g.close()


def preprocess_and_merge_files(directory, list_of_files, title):

    #First iterate through files in the list and create a list of all sentences and a list of all words in the files
    all_sentences = []
    all_words = []
    for file in list_of_files:
        f = open(directory+file).read()
        sentences = nltk.sent_tokenize(f)
        sentences_clean = preprocess_list_of_sentences(sentences, return_words=True)
        all_sentences.extend(sentences_clean[0])
        all_words.extend(sentences_clean[1])

    #Next, write the list of all sentences to a new document, with a new sentence on each line
    g = open(directory+title, 'w+')
    g.writelines('%s\n' % sentence for sentence in all_sentences)
    g.close()

    #Also write the list of all words toa new document, with a new word on each line
    h = open(directory+title[:-4]+'_words.txt', 'w+')
    h.writelines('%s\n' % word for word in all_words)
    h.close()

#In this section, create lists of all .txt files in each of four different folders
#The preprocess_and_merge_files function will also create a .txt file including each word used by the author
#Then write a new .txt file in the same directory containing all preprocessed sentences
ethics_files = []
for filename in os.listdir('Ethics Documents'):
    if filename.endswith(".txt"):
        ethics_files.append(filename)
preprocess_and_merge_files('Ethics Documents/', ethics_files, 'ethicists.txt')

science_files = []
for filename in os.listdir('Science Books'):
    if filename.endswith('.txt'):
        science_files.append(filename)
preprocess_and_merge_files('Science Books/', science_files, 'scientists.txt')

commission_files = []
for filename in os.listdir('Government Reports'):
    if filename.endswith('.txt'):
        commission_files.append(filename)
preprocess_and_merge_files('Government Reports/', commission_files, 'commissions.txt')

journalist_files = []
for i in range(1, 905):
    journalist_files.append('News Articles/articles{}.txt'.format(i))
for file in journalist_files:
    preprocess_and_return_file(file)

#Then make preprocessed files of each individual commission report and news article
for file in commission_files:
    preprocess_and_return_file(file)

for file in journalist_files:
    preprocess_and_return_file(file)




'''
Part 3: Build a classifier to predict text type using files of preprocessed sentences
    3.A. Build a function to preprocess and predict the author type of an input sentence
    3.B. Plot a confusion matrix showing the accuracy of the predictor
'''

#First open all files with unique sentences and store them as lists

with open('Ethics Documents/ethicists.txt', 'r') as filehandle:
    ethics_sentences = [line.rstrip() for line in filehandle.readlines()]

with open('Government Reports/government_commissions.txt', 'r') as filehandle:
    commission_sentences = [line.rstrip() for line in filehandle.readlines()]

with open('News Articles/journalists_MERGED.txt', 'r') as filehandle:
    journalist_sentences = [line.rstrip() for line in filehandle.readlines()]

with open('Science Books/scientists.txt', 'r') as filehandle:
    science_sentences = [line.rstrip() for line in filehandle.readlines()]

all_sentences = journalist_sentences + commission_sentences + ethics_sentences + science_sentences

#Create a single list of all sentences for training, and randomly select an equal number of sentences from each category
np.random.seed(5)
thresh = min(len(ethics_sentences), len(commission_sentences), len(science_sentences), len(journalist_sentences))
media_sample = random.sample(journalist_sentences, thresh)
commission_sample = random.sample(commission_sentences, thresh)
ethics_sample = random.sample(ethics_sentences, thresh)
science_sample = random.sample(science_sentences, thresh)

#Combine into a list of x and y values and split between training and testing data
X = media_sample + commission_sample + ethics_sample + science_sample
Y = [0] * thresh + [1] * thresh + [2] * thresh + [3] * thresh
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.05)

#Train a count vectorizer on the list of all sentences with some parameter adjustments
vectorizer = CountVectorizer(ngram_range=(1,3), max_features=25000, max_df=0.9)
vectorizer.fit(all_sentences)

#Tranform the testing and training sentences into a count vectorized list
transformed_train_x = vectorizer.transform(train_x)
transformed_test_x = vectorizer.transform(test_x)

#Train a Naive Bayes classifier on the transformed training sentences and predict the sentiment of test values
naive = naive_bayes.MultinomialNB()
naive.fit(transformed_train_x, train_y)
predictions = naive.predict(transformed_test_x)
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions, test_y)*100)

#Plot a confusion matrix for the predictions
fig, ax = plt.subplots(figsize=(5,5))
classes=['Journalists', 'Commissions', 'Ethicists', 'Scientists']
plot_confusion_matrix(naive, transformed_test_x, test_y, display_labels=classes,
                      cmap=plt.cm.Blues, normalize='true', xticks_rotation=45, ax=ax)
ax.set_title('Author Classifier', fontdict={'fontsize': 20})
ax.set_ylabel('Actual', fontdict={'fontsize': 16})
ax.set_xlabel('Prediction', fontdict={'fontsize': 16})
plt.savefig('Figures/author_classifier_matrix.jpg', dpi=300, bbox_inches='tight')
plt.show()




'''
Part 4: Examine and graph word frequencies using files of individual, lemmatized words
    4.A. Plot a graph to show Zipfian distribution of word frequencies
    4.B. Plot a graph to show frequncies of the 15 most common words by author type
    4.C. Build a function to randomly sample text and count for keywords
    4.D. Plot a graph showing the frequency distributions of sampling for keywords
    4.E. Count up individual occurrences of rare, significant words and build table
'''

#First open the lists of preprocessed words from each file type and use nltk to create frequency distributions of those words
with open('Ethics Documents/ethicists_words.txt', 'r') as filehandle:
    ethics_words = [line.rstrip() for line in filehandle.readlines()]

with open('Government Reports/government_commissions_words.txt', 'r') as filehandle:
    commission_words = [line.rstrip() for line in filehandle.readlines()]

with open('News Articles/journalists_MERGED_words.txt', 'r') as filehandle:
    journalist_words = [line.rstrip() for line in filehandle.readlines()]

with open('Science Books/scientists_words.txt', 'r') as filehandle:
    science_words = [line.rstrip() for line in filehandle.readlines()]

freq_ethics = nltk.FreqDist(ethics_words)
freq_commissions = nltk.FreqDist(commission_words)
freq_journalists = nltk.FreqDist(journalist_words)
freq_scientists = nltk.FreqDist(science_words)

#Then write a function which receives an array of words with their number of occurrences and plots a line
#graph showing the frequency of each of these words
matplotlib.rcParams['font.family'] = 'Palatino'

def plot_function(array, ax, name):
    sns.set_style('ticks')
    sns.despine()
    words = []
    nums = []
    for word, num in array:
        words.append(word.capitalize())
        nums.append(num)
    ax.plot(range(len(words)), nums, linewidth=2.5)
    ax.set_title(name, fontdict={'fontsize': 16})
    ax.set_ylabel('Number of Occurrences', fontdict={'fontsize':12})
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90)
    matplotlib.rcParams['font.family'] = 'Palatino'

#Write a function that will receive a list of values and added shaded vertical blue backgrounds at those indices
def plot_vertical_lines(vals, ax):
    for val in vals:
        ax.axvline(val, alpha=0.25, linewidth=8)

#The following is a list of words that occurs in the top 15 most common words of only one type of author
unique_words = ['Moral', 'Enhancement', 'Life', 'Dna', 'Research', 'Science', 'Editing', 'Embryo',
                'Public', 'Also', 'Risk', 'Say', 'Scientist', 'Year']

#Create a figure with four equally sized subplots arranged in a square
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, hspace=0.6, wspace=0.3, width_ratios=[1,1], height_ratios=[1,1])

#For each figure, use the two functions to plot the most common words and highlight the unique words
#Note that the unique_words list is not used directly, as indices were selected and insert in hard code
ax1 = plt.subplot(gs[0,0])
plot_function(freq_ethics.most_common(15), ax1, 'Ethicists')
plot_vertical_lines([1,2,9], ax1)
ax2 = plt.subplot(gs[0,1])
plot_function(freq_scientists.most_common(15), ax2, 'Scientists')
plot_vertical_lines([5], ax2)
ax3 = plt.subplot(gs[1,0])
plot_function(freq_commissions.most_common(15), ax3, 'Commissions')
plot_vertical_lines([3,8,9,11,12,13,14], ax3)
ax4 = plt.subplot(gs[1,1])
plot_function(freq_journalists.most_common(15), ax4, 'Journalists')
plot_vertical_lines([0,7,12], ax4)
plt.savefig('Figures/most_common_words.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#The following code block will produce a large figure showing distributions of 19 different words from each author type

#First create a function which randomly samples 10,000 words from each text, counts the number of times that an
#inputted keyword occurs, appends this to a list of counts, and iterates the process 1,000 times
def sample_for_word(word, sources=[ethics_words, science_words, commission_words, journalist_words], num_iterations=1000, sample_size=10000):
    all_results = []
    for source in sources:
        source_results = []
        for i in range(num_iterations):
            random_words = random.sample(source, sample_size)
            trial_count = random_words.count(word)
            source_results.append(trial_count)
        all_results.append(source_results)
    return all_results

#Create a function which will receive a larger gridspec plot, use the sample_for_word function on an inputted keyword,
#and graph kdeplots of the four distributions with standard deviations above
def plot_word_frequency(word, fig, grid_first=False):
    results = sample_for_word(word)

    #This plot uses two subplots: one with the distributions themselves, and a much thinner one above where
    #the standard deviations will be plotted as horizontal lines above the corresponding distribution
    gs_inner = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=fig, height_ratios=[1,9], hspace=0)

    #The first element in the grid will instead contain a label. This if statement leaves the second subplot within
    #the first region blank and converts the first subplot to a legend
    if grid_first==True:
        ax1 = plt.subplot(gs_inner[1,0])
        heights = [np.mean(results[0]), np.mean(results[1]), np.mean(results[2]), np.mean(results[3])]
        errors = [np.std(results[0]), np.std(results[1]), np.std(results[2]), np.std(results[3])]
        for i in range(4):
            ax1.errorbar(y=0, x=0, xerr=0)
        labels = ['Ethicists', 'Scientists', 'Government\nCommissions', 'Journalists']
        ax1.legend(labels=labels, loc='center', edgecolor='black', prop={'size': 18})
        ax1.axis('off')

        ax2 = plt.subplot(gs_inner[0,0])
        ax2.axis('off')

    #The remaining code adds actual distributions to the remaning 19 subplots for each inputted word
    if grid_first==False:
        ax1 = plt.subplot(gs_inner[1,0])
        for source_results in results:
            sns.kdeplot(source_results, shade=True, bw=1, ax=ax1)
        sns.despine(ax=ax1)

        #The second, inner subplot is used to graph standard deviations using the errorbar function
        ax2 = plt.subplot(gs_inner[0,0], sharex=ax1)
        heights = [np.mean(results[0]), np.mean(results[1]), np.mean(results[2]), np.mean(results[3])]
        errors = [np.std(results[0]), np.std(results[1]), np.std(results[2]), np.std(results[3])]
        ax2.set_title('"{}"'.format(word.capitalize()), fontdict={'fontsize':20})
        for i in range(4):
            ax2.errorbar(y=(.1+.15*i), x=heights[i], xerr=errors[i], capsize=5)
        ax2.set_ylim(0,1)
        ax2.axis('off')

#This final function receives a list of words, creates a figure with 20 subplots, directs the first subplot to
#be plotted with a legend only, and iterates over the remaining 19 words, directing the plot_word_frequency
#function to sample and plot each word in the corresponding subplot
def plot_multiple_words(words):
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(nrows=5, ncols=4, figure=fig, hspace=0.3, width_ratios=[1,1,1,1], height_ratios=[1,1,1,1,1])
    for i in range(len(words)):
        if i==0:
            grid=gs[0,0]
            plot_word_frequency(words[0], grid, grid_first=True)
        else:
            if i<4:
                grid = gs[0,i]
            elif i<8:
                grid=gs[1,(i-4)]
            elif i<12:
                grid=gs[2,(i-8)]
            elif i<16:
                grid=gs[3,(i-12)]
            elif i<20:
                grid=gs[4,(i-16)]
            plot_word_frequency(words[i], grid)
    fig.savefig('Figures/word_frequencies.jpg', dpi=300, bbox_inches='tight')
    plt.show()

#Below is the list of 19 key words examined, with a blank space first to correspond to the subplot with a legend
key_words = ['', 'cure', 'therapy', 'risk',
             'china', 'international', 'moratorium', 'enhancement',
             'embryo', 'disability', 'mosquito', 'patent',
             'choice', 'democratic', 'public', 'consensus',
             'will', 'future', 'could', 'might']

sns.set_palette('muted')
plot_multiple_words(key_words)


#The following list of words are important but rare terms. The next goal is to count the total number of occurrences of each
phrases = ['gelsinger', 'aquadvantage', 'salmon', 'glybera', 'asilomar', 'oviedo', 'brave new world', 'democracy',
           'jiankui', 'niakan', 'junjiu', 'huang']

#Iterate over the key phrases and over each dataset. For each sentence, if the sentence contains the key phrase, add it
#to a running total. Then print the total number of occurrences of each word in each dataset
for phrase in phrases:
    count_eth = 0
    count_sci = 0
    count_com = 0
    count_jou = 0
    for sentence in ethics_sentences:
        if phrase in sentence:
            count_eth += 1
    for sentence in science_sentences:
        if phrase in sentence:
            count_sci += 1
    for sentence in commission_sentences:
        if phrase in sentence:
            count_com += 1
    for sentence in journalist_sentences:
        if phrase in sentence:
            count_jou += 1
    print('{}: {} {} {} {}'.format(phrase, count_eth, count_sci, count_com, count_jou))





'''
5. Build a sentiment analyzer using 5000 sentences hand-labeled by sentiment
    5.A. Fine-tune parameters as much as possible
    5.B. Plot confusion matrices for each of four model types
    5.C. Predict labels for all sentences and store in separate .txt files for easy access
    5.D. Predict label probabilities for all sentences and store in separate .txt files for easy access
    5.E. Repeat previous two steps for each individual news article and store in separate .txt files
    5.F. Repeat previous two steps for different government commissions
'''

#Begin by reading .csv file with all labeled sentences
samples = pd.read_csv('CSV and Excel Spreadsheets/sentiment_sample_sentences.csv')

#Then add a new column for preprocessed sentences
samples['SENTENCE_CLEAN'] = [preprocess_sentence(sentence) for sentence in samples['SENTENCE']]

#Print the numbers of negative, neutral, and positive labels
samples['SENTIMENT'].value_counts()

#First separate the .csv file into three dataframes based on sentiment
np.random.seed(0)
pos = samples[samples['SENTIMENT']==1.0]
neg = samples[samples['SENTIMENT']==-1.0]
neu = samples[samples['SENTIMENT']==0.0]

#Create x and y data and split into training and testing groups
X = list(pos['SENTENCE_CLEAN']) + list(neg['SENTENCE_CLEAN']) + list(neu['SENTENCE_CLEAN'])
Y = [1] * len(pos) + [-1] * len(neg) + [0] * len(neu)
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.08)

#Fit a count vectorizer to the list of all sentences created earlier
vectorizer = CountVectorizer(ngram_range=(1,1), max_df=0.9)
vectorizer.fit(all_sentences)

#Transform the training and testing sentences using the count vectorizer
transformed_train = vectorizer.transform(train_x)
transformed_test = vectorizer.transform(test_x)

#Train four separate models and print their accuracy scores
#Models are: Naive Bayes, Logistic Regression, Support Vector Machine, and Random Forests Classifier
naive = naive_bayes.MultinomialNB()
naive.fit(transformed_train, train_y)
predictions_naive = naive.predict(transformed_test)
print("Naive Bayes Accuracy Score -> ", accuracy_score(test_y, predictions_naive))

logistic = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=2000)
logistic.fit(transformed_train, train_y)
predictions_logistic = logistic.predict(transformed_test)
print("Logistic Regression Accuracy Score -> ", accuracy_score(test_y, predictions_logistic))

support = svm.SVC(max_iter=2000, kernel='linear', probability=True)
support.fit(transformed_train, train_y)
predictions_svm = support.predict(transformed_test)
print("Support Vector Machine Accuracy Score -> ", accuracy_score(test_y, predictions_svm))

forest = ensemble.RandomForestClassifier(n_estimators=100)
forest.fit(transformed_train, train_y)
predictions_forest = forest.predict(transformed_test)
print("Random Forests Accuracy Score -> ", accuracy_score(test_y, predictions_forest))

#Plot confusion matrices for each of the above models in a vertical line

fig = plt.figure(figsize=(3.5,12))
labels=['negative', 'neutral', 'positive']
im = ax.imshow(np.arange(1).reshape((1,1)))

ax1 = plt.subplot(411)
plot_confusion_matrix(naive, transformed_test, test_y, display_labels=labels,
                     cmap=plt.cm.Blues, normalize='true', xticks_rotation=45, ax=ax1)
ax1.set_title('Naive Bayes (59.25% Accuracy)', pad=0.2, fontdict={'fontsize': 16})
ax1.set_xlabel('')
ax1.set_xticklabels(['', '', ''])

ax2 = plt.subplot(412)
plot_confusion_matrix(logistic, transformed_test, test_y, display_labels=labels,
                     cmap=plt.cm.Blues, normalize='true', xticks_rotation=45, ax=ax2)
ax2.set_title('Logistic Regression (62.25% Accuracy)', pad=0.2, fontdict={'fontsize': 16})
ax2.set_xlabel('')
ax2.set_xticklabels(['','',''])

ax3 = plt.subplot(413)
plot_confusion_matrix(support, transformed_test, test_y, display_labels=labels,
                     cmap=plt.cm.Blues, normalize='true', xticks_rotation=45, ax=ax3)
ax3.set_title('Support Vector Machine (56.25% Accuracy)', pad=2, fontdict={'fontsize': 16})
ax3.set_xlabel('')
ax3.set_xticklabels(['','',''])

ax4 = plt.subplot(414)
plot_confusion_matrix(forest, transformed_test, test_y, display_labels=labels,
                     cmap=plt.cm.Blues, normalize='true', xticks_rotation=45, ax=ax4)
ax4.set_title('Random Forest (58.25% Accuracy)', pad=0.2, fontdict={'fontsize':16})

plt.subplots_adjust(wspace=0.3, hspace=0.45)
plt.savefig('Figures/sentiment_matrices_line.jpg', dpi=300, bbox_inches='tight')
plt.show()

#Create functions to predict labels

#First function predicts the label of a sentence using each of the four classifiers
def label_sentence_simple(sentence):
    transformed_sentence = vectorizer.transform([sentence])
    prediction_naive = naive.predict(transformed_sentence)
    prediction_logistic = logistic.predict(transformed_sentence)
    prediction_svm = support.predict(transformed_sentence)
    prediction_forest = forest.predict(transformed_sentence)
    return prediction_naive[0], prediction_logistic[0], prediction_svm[0], prediction_forest[0]

#Second function predicts the label probability by subtracting the probability of a negative label from the
#probability of a positive label, using each of the four classifiers
def label_sentence_complex(sentence):
    transformed_sentence = vectorizer.transform([sentence])

    prediction_naive_raw = naive.predict_proba(transformed_sentence)
    prediction_naive = prediction_naive_raw[0][2] - prediction_naive_raw[0][0]

    prediction_logistic_raw = logistic.predict_proba(transformed_sentence)
    prediction_logistic = prediction_logistic_raw[0][2] - prediction_logistic_raw[0][0]

    prediction_svm_raw = support.predict_proba(transformed_sentence)
    prediction_svm = prediction_svm_raw[0][2] - prediction_svm_raw[0][0]

    prediction_forest_raw = forest.predict_proba(transformed_sentence)
    prediction_forest = prediction_forest_raw[0][2] - prediction_forest_raw[0][0]

    return prediction_naive, prediction_logistic, prediction_svm, prediction_forest

#Third function uses only logistic regression to predict both label and label probability of a sentence
def label_with_logistic(sentence):
    transformed_sentence = vectorizer.transform([sentence])
    prediction_simple = logistic.predict(transformed_sentence)[0]

    prediction_complex_raw = logistic.predict_proba(transformed_sentence)
    prediction_complex = prediction_complex_raw[0][2] - prediction_complex_raw[0][0]

    return prediction_simple, prediction_complex

#Next, create labels and label probabilities for all sentences and write to new files

ethics_array_simple = [label_sentence_simple(sentence) for sentence in ethics_sentences]
ethics_array_complex = [label_sentence_complex(sentence) for sentence in ethics_sentences]

commission_array_simple = [label_sentence_simple(sentence) for sentence in commission_sentences]
commission_array_complex = [label_sentence_complex(sentence) for sentence in commission_sentences]

journalist_array_simple = [label_sentence_simple(sentence) for sentence in journalist_sentences]
journalist_array_complex = [label_sentence_complex(sentence) for sentence in journalist_sentences]

science_array_simple = [label_sentence_simple(sentence) for sentence in science_sentences]
science_array_complex = [label_sentence_complex(sentence) for sentence in science_sentences]


arrays = [ethics_array_simple, ethics_array_complex, commission_array_simple, commission_array_complex,
          journalist_array_simple, journalist_array_complex, science_array_simple, science_array_complex]

names = ["ethics", "commission", "journalist", "science"]

for i in range(4):
    f = open('Sentiment Arrays/{}_simple'.format(names[i]), 'w+')
    f.writelines("{}\n".format(datapoint) for datapoint in arrays[i*2])
    f.close()

    g = open('Sentiment Arrays/{}_complex'.format(names[i]), 'w+')
    g.writelines("{}\n".format(datapoint) for datapoint in arrays[i*2 + 1])
    g.close()

#Use the logistic classifier to iterate through all news articles and create a new file containing
#a tuple with both a simple label prediction and a composite probability prediction

for i in range(1,905):
    with open('News Articles/article{}_preprocessed.txt'.format(i), 'r') as filehandle:
        article_sentences = [line.rstrip() for line in filehandle.readlines()]
    output_tuples = []
    for sentence in article_sentences:
        output_tuples.append(label_with_logistic(sentence))
    g = open('Sentiment Arrays/article{}_data'.format(i), 'w+')
    g.writelines('{}\n'.format(datapoint) for datapoint in output_tuples)
    g.close()

#Repeat the last step with the government commission reports

report_list = []
for filename in os.listdir('Government Reports/Preprocessed Reports'):
    if filename.endswith(".txt"):
        report_list.append(filename)

for report in report_list:
    with open('Government Reports/Preprocessed Reports/'+report, 'r') as filehandle:
        report_sentences = [line.rstrip() for line in filehandle.readlines()]
    output_tuples = []
    for sentence in report_sentences:
        output_tuples.append(label_with_logistic(sentence))
    g = open('Sentiment Arrays/{}_data'.format(report[:-17]), 'w+')
    g.writelines('{}\n'.format(datapoint) for datapoint in output_tuples)
    g.close()






'''
6. Use sentiment analyzer to examine the sentiment of different text types
    6.A. Build a table to list occurrence of each label by author type and model
    6.B. Plot distributions of label probabilities by author type and model
    6.C. Use label probabilities of individual articles to plot changes in sentiment over time
    6.D. Use list of articles by publication to examine sentiment distributions by publication
    6.E. Build a function to search articles for keywords
    6.F. Use label probabilities of individual articles to compare sentiment of articles with or without keywords
    6.G. Use .csv file of commission reports to examine sentiment by region, year
'''

#First open the predicted probabilties for all sentences for each type of author
with open('Sentiment Arrays/ethics_complex', 'r') as filehandle:
    ethics_complex_data = [eval(line.rstrip()) for line in filehandle.readlines()]
with open('Sentiment Arrays/commission_complex', 'r') as filehandle:
    commission_complex_data = [eval(line.rstrip()) for line in filehandle.readlines()]
with open('Sentiment Arrays/journalist_complex', 'r') as filehandle:
    journalist_complex_data = [eval(line.rstrip()) for line in filehandle.readlines()]
with open('Sentiment Arrays/science_complex', 'r') as filehandle:
    science_complex_data = [eval(line.rstrip()) for line in filehandle.readlines()]

#The following function takes a list of tuples consisting of predicted label, predicted label probability
#and calculates the average predicted label probability of the array and the standard deviation of the probabilities
def calculate_article_sentiment(article_num):
    with open('Sentiment Arrays/article{}_data'.format(article_num)) as f:
        sentiments = [eval(datapoint.rstrip()) for datapoint in f.readlines()]
    sentiment_probs = [b for a, b in sentiments]
    mean = np.mean(sentiment_probs)
    std = np.std(sentiment_probs)
    return mean, std, len(sentiment_probs)

#Next calculate mean and standard deviation of sentiment probabilities for all articles
article_sentiments = []
for i in range(1, 905):
    article_sentiments.append(calculate_article_sentiment(i))

#Format x values for a scatter plot:
date_vals = [dateutil.parser.parse(i) for i in meta_news['date']]
x_vals = dates.date2num(date_vals)

#Set y values equal to mean for each article and error bars equal to standard deviation:
y_vals = [a for a, b, c in article_sentiments]
errors = [b for a, b, c in article_sentiments]
sizes = [c for a, b, c in article_sentiments]

#Plot a error bar scatterplot showing sentiment of articles over time
fig, ax = plt.subplots(figsize=(16, 4))
ax.scatter(x_vals, y_vals, s=sizes, edgecolor='black', linewidth=0.5)

#Add a linear regression line and p-value
gradient, intercept, r_value, p_value, std_err = linregress(x_vals,y_vals)
ax.plot(x_vals, [gradient*x_val + intercept for x_val in x_vals], linestyle='--', color='black', linewidth=1.5)

#Create x ticks and tick labels and add graph elements
label_ticks = ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
x_labels = ['Jan. 2013', 'Jan. 2014', 'Jan. 2015', 'Jan. 2016', 'Jan. 2017', 'Jan. 2018', 'Jan. 2019', 'Jan. 2020']
x_ticks = dates.date2num([dateutil.parser.parse(i) for i in label_ticks])
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, ha='center', fontdict={'fontsize':16})
ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
ax.set_yticklabels(['-0.4', '-0.2', '0.0', '0.2', '0.4'], fontdict={'fontsize': 16})
ax.set_ylabel('Article Sentiment', fontdict={'fontsize': 16})
ax.annotate('R-squared value: {}\np-value: {}'.format(round(r_value**2, 5), round(p_value, 5)),
            xy=(dates.date2num(dateutil.parser.parse('01/01/2013')), 0.35), fontsize=16)

plt.savefig('Figures/sentiment_over_time.jpg', dpi=300, bbox_inches='tight')
plt.show()

#Examine average sentiment of each publication and look at pairwise comparisons using a Tukey test
meta_news['average_sentiment'] = [article_sentiments[i][0] for i in range(len(meta_news))]
meta_news.head(20)
tukey = pairwise_tukeyhsd(meta_news['average_sentiment'], meta_news['publication'], 0.05)
print(tukey)

#Next, build lists of article indices for articles including specific keywords
includes_china = []
includes_jiankui = []
includes_kathy_niakan = []
includes_junjiu_huang = []
includes_moratorium = []
includes_ban = []

for i in range(1, 905):
    f = open("News Articles/article{}_preprocessed.txt".format(i)).read()
    if 'china' in f or 'chinese' in f:
        includes_china.append(i)
    if 'jiankui' in f:
        includes_jiankui.append(i)
    if 'kathy niakan' in f:
        includes_kathy_niakan.append(i)
    if 'junjiu huang' in f or 'huang junjiu' in f:
        includes_junjiu_huang.append(i)
    if 'moratorium' in f:
        includes_moratorium.append(i)
    if ' ban ' in f:
        includes_ban.append(i)

#Use boxplots to display the distribution of sentiments with and without keywords
#Create three boxplots showing the distribution of three three-way comparisions

#First write a function to create data arrays
#Will take two lists and output three arrays of average sentiments in (1) articles in list 1 but not list 2,
#(2) articles in list 2 but not list 1, and (3) articles in neither list 1 nor list 2
def compare_three_way(list1, list2, list1_exclusive=True):
    list1_data = []
    if list1_exclusive==True:
        for i in [j for j in list1 if j not in list2]:
            list1_data.append(calculate_article_sentiment(i)[0])
    if list1_exclusive==False:
        for i in list1:
            list1_data.append(calculate_article_sentiment(i)[0])

    list2_data = []
    for i in [j for j in list2 if j not in list1]:
        list2_data.append(calculate_article_sentiment(i)[0])

    remainder = []
    for i in [j for j in range(1, 905) if j not in list1 and j not in list2]:
        remainder.append(calculate_article_sentiment(i)[0])

    return list1_data, list2_data, remainder

#Create plot for comparisons
fig, ax = plt.subplots(figsize=(16,4))
sns.set_palette('pastel')

#Use lists of articles with keywords to create lists of average article sentiments
with_jiankui, with_china, without_china = compare_three_way(includes_jiankui, includes_china, list1_exclusive=False)
with_niakan, with_huang, neither1 = compare_three_way(includes_kathy_niakan, includes_junjiu_huang)
with_ban, with_moratorium, neither2 = compare_three_way(includes_ban, includes_moratorium)

#Use the lists of average sentiments to create a dataframe
values = with_jiankui + with_china + without_china + with_huang + with_niakan + neither1 + with_ban + with_moratorium + neither2
names = ['a']*len(with_jiankui) + ['b']*len(with_china) + ['c']*len(without_china) + ['a']*len(with_huang) + ['b']*len(with_niakan) + ['c']*len(neither1) + ['a']*len(with_ban) + ['b']*len(with_moratorium) + ['c']*len(neither2)
types = ['1']*len(with_jiankui+with_china+without_china) + ['2']*len(with_niakan+with_huang+neither1) + ['3']*len(with_ban+with_moratorium+neither2)
data = {'Sentiment Values':values, 'Iteration':names, 'Condition':types}
frame=pd.DataFrame(data)

#Use the dataframe to plot boxplots and adjust figure appearance
sns.boxplot(y='Sentiment Values', x='Condition', hue='Iteration', data=frame, ax=ax).legend_.remove()
ax.set_xticks([-0.275,0,0.275,0.725,1,1.275,1.725,2,2.275])
ax.set_xticklabels(['"Jiankui"', '"China"', 'Neither',
                   '"Junjiu Huang"', '"Kathy Niakan"', 'Neither',
                   '"Ban"', '"Moratorium"', 'Neither'],
                  rotation=30, ha='right', fontdict={'fontsize': 16})
ax.set_ylabel('Article Sentiments', fontdict={'fontsize': 16})
ax.set_xlabel('')
ax.set_yticklabels(['-0.4', '-0.4', '- 0.2', '0.0', '0.2', '0.4'], fontdict={'fontsize': 16})
plt.savefig('Figures/sentiment_with_keywords.jpg', dpi=300, bbox_inches='tight')
plt.show()

#Then use Tukey Pairwise tests to examine the statistical significance of variation in the distributions
v1 = np.concatenate([with_jiankui, with_china, without_china])
labels1 = ['jiankui']*len(with_jiankui) + ['china']*len(with_china) + ['neither']*len(without_china)
v2 = np.concatenate([with_niakan, with_huang, neither1])
labels2 = ['niakan']*len(with_niakan) + ['huang']*len(with_huang) + ['neither']*len(neither1)
v3 = np.concatenate([with_ban, with_moratorium, neither2])
labels3 = ['ban']*len(with_ban) + ['moratorium']*len(with_moratorium) + ['neither']*len(neither2)
tukey1 = pairwise_tukeyhsd(v1, labels1, 0.05)
tukey2 = pairwise_tukeyhsd(v2, labels2, 0.05)
tukey3 = pairwise_tukeyhsd(v3, labels3, 0.05)
print(tukey1)
print(tukey2)
print(tukey3)

#In this final sub-section, open the .csv file containing metadata about the commission reports
gov_meta = pd.read_csv('CSV and Excel Spreadsheets/metadata.csv')
gov_meta['sentiment_file'] = [name[:-4]+'_data' for name in gov_meta['file_name']]

#Add a column including the arrays of sentiment, another including the average sentiment, and another with number
#of sentences
def add_array(filename):
    with open('Sentiment Arrays/{}'.format(filename)) as f:
        sentiments = [eval(datapoint.rstrip()) for datapoint in f.readlines()]
    return sentiments

#Add columns including the array of sentence probabilities for all sentences, the average probability, and the number of sentences
gov_meta['sentiment_array'] = [add_array(filename) for filename in gov_meta['sentiment_file']]
gov_meta['average_sentiment'] = [np.mean(array) for array in gov_meta['sentiment_array']]
gov_meta['num_sentences'] = [len(array) for array in gov_meta['sentiment_array']]

#Now create a scatterplot, similar to the one above, showing the average sentiment over time

#Format x values for a scatter plot:
date_vals = [dateutil.parser.parse(i) for i in gov_meta['year']]
x_vals = dates.date2num(date_vals)

#Plot a error bar scatterplot showing sentiment of commissions over time
fig, ax = plt.subplots(figsize=(16, 4))
sizes = [len(gov_meta['sentiment_array'][i]) for i in range(len(gov_meta))]
ax.scatter(x_vals, gov_meta['average_sentiment'], s=sizes, edgecolor='black')

#Create a data array that is weighted by report length, then calculate and print the gradient, r_value, and p_value
#of a linear regression
full_y_data = []
full_x_vals = []
for i in range(len(gov_meta)):
    y_values = [gov_meta['average_sentiment'][i]]*gov_meta['num_sentences'][i]
    full_y_data.extend(y_values)
    x_values = [x_vals[i]]*gov_meta['num_sentences'][i]
    full_x_vals.extend(x_values)
gradient, intercept, r_value, p_value, std_err = linregress(full_x_vals, full_y_data)
print(gradient, r_value**2, p_value)

#Then plot a linear regression with r- and p-values using the unweighted sentiment averages
gradient, intercept, r_value, p_value, std_err = linregress(x_vals, gov_meta['average_sentiment'])
ax.plot([x_vals[3], x_vals[-2]], [(gradient*x_vals[3]+intercept), (gradient*x_vals[-2]+intercept)], linestyle='--', color='black')

#Create x ticks and tick labels and add graph elements
label_ticks = ['2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01']
x_labels = ['Jan. 2016', 'Jan. 2017', 'Jan. 2018', 'Jan. 2019']
x_ticks = dates.date2num([dateutil.parser.parse(i) for i in label_ticks])
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontdict={'fontsize': 16})
ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1])
ax.set_yticklabels(['-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1'], fontdict={'fontsize': 16})
ax.set_ylabel('Average Sentiment', fontdict={'fontsize': 16})
ax.annotate('R-squared value: {}\np-value: {}'.format(round(r_value**2, 4), round(p_value, 5)),
            xy=(dates.date2num(dateutil.parser.parse('11/01/2015')), 0.05), fontsize=16)
plt.savefig('Figures/commission_sentiment_over_time.jpg', dpi=300, bbox_inches='tight')
plt.show()

#Finally, write a function that takes a column and a label of interest and returns an array of all predicted
#sentence sentiment probabilities for all reports with that label
def array_from_subgroup(column, label, mean=False):
    df = gov_meta[gov_meta[column]==label].reset_index()
    data = []
    for i in range(len(df)):
        array_items = df['sentiment_array'][i]
        data.extend([b for a, b in array_items])
    if mean==False:
        return data
    if mean==True:
        return label, np.mean(data), len(data)

#Use the above function to print the array for all reports from each geogrpahical region
#Then format the returned values into a dataframe
regions = ['united kingdom', 'united states', 'germany', 'france', 'international', 'australia', 'new zealand',
          'europe', 'oecd', 'denmark', 'netherlands', 'spain']
y_vals = []
x_vals = []
for i in regions:
    y_vals.extend(array_from_subgroup('country', i))
    x_vals.extend([i]*len(array_from_subgroup('country', i)))

data = {'Country': x_vals, 'Sentiments': y_vals}
frame = pd.DataFrame(data)


#Then use Tukey Pairwise tests to examine the statistical significance of variation in the distributions
tukey = pairwise_tukeyhsd(frame['Sentiments'], frame['Country'], 0.05)
print(tukey)
