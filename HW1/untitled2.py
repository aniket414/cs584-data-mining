#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:25:36 2021

@author: aniket
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords 
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer
import matplotlib.pyplot as plt
from collections import Counter

#Read Train and Test File
with open("train_new.dat", "r") as trainingFileReader:
    trainingFile = trainingFileReader.readlines()
    
with open("test_new.dat", "r") as testingFileReader:
    testingFile = testingFileReader.readlines()
    
#Spliting the training file into two files: Reviews and Sentiments 
trainSentiments = [x.split("\t", 1)[0] for x in trainingFile]
trainReviews = [x.split("\t", 1)[1] for x in trainingFile]

#Removing HTML Scripts
def removeHTML(review):
    beautifulSoup = BeautifulSoup(review, "html.parser")
    return beautifulSoup.get_text()

#Removing Brackets
def removeBrackets(review):
    return re.sub("\[[^]]*\]", "", review)

#Removing Special Characters
def removeSpecialCharacters(review):
    pattern = r"[^a-zA-z0-9\s]"
    review = re.sub(pattern, "", review)
    return review

#Removing Noisy data
def removeNoisyData(review):
    review = removeHTML(review)
    review = removeBrackets(review)
    review = removeSpecialCharacters(review)
    return review

#Remove Noise from Review column of both Train and Test data
trainReviewsWithoutNoise = list(map(removeNoisyData, trainReviews))
testReviewsWithoutNoise = list(map(removeNoisyData, testingFile))


#Tokenizing text initialization outside function so that multiple instances are not created
tokenizer = ToktokTokenizer()

#Stemming text initialization outside function so that multiple instances are not created
lancasterStemmer = LancasterStemmer()

#List of stopwords
listOfStopwords = stopwords.words("english")

#Removing the stopwords from Review
def removeStopwords(review):
    reviewTokens = tokenizer.tokenize(review)
    reviewTokens = [token.strip() for token in reviewTokens]
    cleanTokens = [token for token in reviewTokens if token.lower() not in listOfStopwords]
    cleanReview = " ".join(cleanTokens)    
    return cleanReview

#Stemming the Review
def stemming(review):
    review = " ".join([lancasterStemmer.stem(word) for word in review.split()])
    return review

#Preprocess the Review once Noise has been removed
def preProcess(review):
    review = removeStopwords(review)
    review = stemming(review)
    return review

#Preprocess Review column of both Train and Test data
trainReviewsPreProcessed = list(map(preProcess, trainReviewsWithoutNoise))
testReviewsPreProcessed = list(map(preProcess, testReviewsWithoutNoise))

def termFrequencyIDF(trainReview, testReview):
    """Takes in processed training and testing data, outputs respective L2-normalized sparse matrices with TF-IDF values"""
    
    vectorizer = TfidfVectorizer(norm = "l2")
    trainVectorMatrix = vectorizer.fit_transform(trainReview)
    
    #parameters generated from fit() method on train data applied upon model to generate transformed data set of test data
    testVectorMatrix = vectorizer.transform(testReview)
    return trainVectorMatrix, testVectorMatrix

def cosineSimilarities(trainVectorMatrix, testVectorMatrix):
    """Takes in the entire training data and the testing data (both sparse matrices) and 
        gives the cosine similarity between the two as a numpy array.
        Numpy arrays are fastest to work with for sorting while finding nearest neighbors"""
    transposedTrainVector = np.transpose(trainVectorMatrix)
    cosineSimilaritiesOfReviews = np.dot(testVectorMatrix, transposedTrainVector)
    return cosineSimilaritiesOfReviews.toarray()        

def kNearestNeighbours(reviewSimilarity, k):
    """Takes in the similarity vector (numpy array) and number of neighbors to find, to return the K Nearest Neighbors indices.
        The input array gets sorted in descending order and the first k indices returned.
        The argsort function has been used to preserve the indices of the training reviews so that their respective labels
        can be easily referenced in the training labels list"""
   
    return np.argsort(-reviewSimilarity)[:k]
     

def predictSentiment(kNearestNeighbours, sentiments):
    """Takes in the list of K nearest Neighbors and the full training labels list, and 
        calculates the count of positive and negative reviews. 
        If positive reviews are more, then the test review is positive and vice-versa"""
    
    positiveSentiment, negativeSentiment = 0, 0
    for nearestNeighbour in kNearestNeighbours:
        if int(sentiments[nearestNeighbour]) == 1:
            positiveSentiment += 1
        else:
            negativeSentiment += 1
    return "+1" if (positiveSentiment >= negativeSentiment) else "-1"
    
    
trainReviewsMatrix, testReviewsMatrix = termFrequencyIDF(trainReviewsPreProcessed, testReviewsPreProcessed)
reviewSimilarities = cosineSimilarities(trainReviewsMatrix, testReviewsMatrix)

#Actual Prediction for the Test Reviews
k = 158 #sqrt(2500) = 158.11 (Approx)
testSentiments = list()

for reviewSimilarity in reviewSimilarities:
    kNN = kNearestNeighbours(reviewSimilarity, k)
    prediction = predictSentiment(kNN, trainSentiments)
    testSentiments.append(prediction)
        
#Writing the predicted Sentiments of Test Review in a file
sentimentFileWriter = open("output-k-300.dat", "w")
sentimentFileWriter.writelines("%s\n" % sentiment for sentiment in testSentiments)
sentimentFileWriter.close()

#Plotting Pie Chart for Visualization of Predicted Sentiments
sentiments = Counter(testSentiments).keys()
sentimentsCount = Counter(testSentiments).values()
plt.pie(sentimentsCount, labels = sentimentsCount, shadow = True, autopct = "%1.1f%%")
plt.legend(sentiments, title = "Sentiments", loc = "center right", bbox_to_anchor = (1, 0, 0.5, 1))
plt.title("Movie Review Sentiments")
plt.show()