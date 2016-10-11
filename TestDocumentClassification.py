
import nltk 
from sklearn import cross_validation
import csv
import numpy


f = open('tweetsentiment.csv')
twittersentiment_data = csv.reader(f)


twitter_data = []
for (words, classofdata) in twittersentiment_data:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    twitter_data.append((words_filtered, classofdata))

#print twitter_data

def get_words_in_twitter_data(twitter_data):
    all_words = []
    for (words, classofdata) in twitter_data:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


word_features = get_word_features(get_words_in_twitter_data(twitter_data))

#print word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#print extract_features(twitter_data)


training_set = nltk.classify.apply_features(extract_features, twitter_data)

#print training_set

#cross validation
cv = cross_validation.KFold(len(training_set), n_folds=3, shuffle=True, random_state=None)

variable = []
for traincv, testcv in cv:
    classifier = nltk.NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    variable.append(nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]]))
    
    
AvgAcc = (numpy.mean(variable))*100     
print 'Accuracy of Classifier:', AvgAcc , '%'

    
    
new_twitter_data = raw_input("New tweet: ")
print 'New tweet classified as: ', classifier.classify(extract_features(new_twitter_data.split()))



