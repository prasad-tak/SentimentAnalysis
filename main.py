# importing required packages

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import pyphen
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re


#--------------------------------------------------------------------------------------------------------------------------------------


# We will create stopwords list and lower it down so that we won't miss any stopwords

StopWords = []

# function to extract stopwords from text files

def getStopWords(path,StopWords):
    with open(path,'r') as file:
        StopWords.extend(file.read().split())
        
#extracting stopwords from the text file

getStopWords('StopWords/StopWords_Auditor.txt',StopWords)
getStopWords('StopWords/StopWords_Currencies.txt',StopWords)
getStopWords('StopWords/StopWords_DatesandNumbers.txt',StopWords)
getStopWords('StopWords/StopWords_Generic.txt',StopWords)
getStopWords('StopWords/StopWords_GenericLong.txt',StopWords)
getStopWords('StopWords/StopWords_Geographic.txt',StopWords)
getStopWords('StopWords/StopWords_Names.txt',StopWords)


# converting all words to lower case and removing duplicates

StopWords = list(set([x.lower() for x in StopWords]))


#----------------------------------------------------------------------------------------------------------------------------------

# Creating Positive and Negative dictionaries

# creating list of positive words from positive-words.txt and converting it to lower case in order to remove duplicates

with open('MasterDictionary/positive-words.txt','r') as file:
    positive = file.read().split()
positive = list(set([word.lower() for word in positive]))

# creating list of positive words from positive-words.txt and converting it to lower case in order to remove duplicates

with open('MasterDictionary/negative-words.txt','r') as  file:
    negative = file.read().split()
negative = list(set([word.lower() for word in negative]))


# Creating positive and negative dictionaries for words not present in StopWords

PositiveDictionary = {item:0 for item in positive if item not in StopWords}
NegativeDictionary = {item:0 for item in negative if item not in StopWords}

#----------------------------------------------------------------------------------------------------------------------------------

# Creating list of all the output variables
#title
Title = []

# Number of Positive Words
PositiveScore = []

#Number of negative words
NegativeScore = []

# score that determines if a given text is positive or negative in nature.
PolarityScore = []

# score that determines if a given text is objective or subjective. 
SubjectivityScore = []

#Avg number of words per sentence
AvgSentLength = []

# percentage of number of complex words
perComplex = []

# fogIndex for readability
fogIndex = []

#Complex word count
ComplexWordCount= []

# word count without nltk stopwords
WordCount = []

# Avg Syllables per word
SyllablePerWord = []

#Count of perosnal pronouns
PersonalPronouns = []

#avg word length
AvgWordLength = []

#-----------------------------------------------------------------------------------------------------------------------------------

# Defining functions to calculate the above variables

# for positive score 
def get_positiveScore(words,PositiveDictionary=PositiveDictionary):
    for word in words:
        if word in PositiveDictionary.keys():
            PositiveDictionary[word] += 1
    PositiveScore = 0
    for key in PositiveDictionary.keys():
        PositiveScore += PositiveDictionary[key]
        PositiveDictionary[key] = 0
    return PositiveScore
 
 #--------------------------------------------------------------------------------------------------------------------------------
 
 # for negative score 

def get_negativeScore(words,NegativeDictionary=NegativeDictionary):
    for word in words:
        if word in NegativeDictionary.keys():
            NegativeDictionary[word] += 1
    NegativeScore = 0
    for key in NegativeDictionary.keys():
        NegativeScore += NegativeDictionary[key]
        NegativeDictionary[key] = 0
    return NegativeScore

#------------------------------------------------------------------------------------------------------------------------------------

# for polarity score

def get_polarityScore(PositiveScore,NegativeScore):
    return (PositiveScore - NegativeScore)/ ((PositiveScore + NegativeScore) + 0.000001)

#-----------------------------------------------------------------------------------------------------------------------------------

# for subjectivity score
def get_subjectivityScore(PositiveScore,NegativeScore,words):
    return (PositiveScore + NegativeScore)/ (len(words) + 0.000001)

#------------------------------------------------------------------------------------------------------------------------------------

# for avg sent length
def get_avgSentLen(sentences,words):
    return len(words)/len(sentences)

#--------------------------------------------------------------------------------------------------------------------------------------

# for complex count

def nSyllables(word):
    dic = pyphen.Pyphen(lang='en_US') 
    syllables = dic.inserted(word).count('-') + 1
    return syllables

def get_complexCount(words):
    complexWords = [word for word in words if nSyllables(word) > 2]
    return len(complexWords)

#-------------------------------------------------------------------------------------------------------------------------------------

#percentage of complex words in  the article

def get_complexPercent(complexCount,wordCount):
    return (complexCount / wordCount) * 100
 
 #------------------------------------------------------------------------------------------------------------------------------------
 
 # calculating fog index by passing complex percent and avg  sentence length

def get_fogIndex(avg,percent):
    return 0.4 * (avg + percent)

#----------------------------------------------------------------------------------------------------------------------------------------

# This is the word count given by removing nltk stopwords form the words
def get_WordCount(words):
    new_words = [x for x in words if x not in stopwords.words('english')]
    return len(new_words)

#--------------------------------------------------------------------------------------------------------------------------------------

def get_avgsyllable(words):
    new_words = [nSyllables(word) for word in words]
    return np.mean(new_words)

#--------------------------------------------------------------------------------------------------------------------------------------

def get_avgwordlength(words):
    count = [len(word) for word in words]
    return np.mean(count)

#---------------------------------------------------------------------------------------------------------------------------------------

def get_personalPronouns(text):
    personal_pronouns = ["I", "we", "my", "ours", "us", "We", "My", "Ours"]

# Create a regex pattern 
    pattern = r'\b(?:' + '|'.join(personal_pronouns) + r')\b'

    pronouns = re.findall(pattern, text, flags=re.IGNORECASE)

    return len(pronouns)
    
#--------------------------------------------------------------------------------------------------------------------------------------

# Main function to run for each url

def calculateAll(url,id):
    http = requests.get(url)
    soup = BeautifulSoup(http.content,'html.parser')
    title = soup.title.string.replace(' | Blackcoffer Insights','')
    Title.append(title)
    paragraph = ""
    paraSet = soup.find_all('p')[16:-3]
    for para in paraSet:
        paragraph+= para.text
    
    # Exporting text to txt file
    
    fileName = 'Output/'+str(id)+'.txt'  
    with open(fileName,'w',encoding='utf-8') as file:
        file.write(title)
        file.write('\n')
        file.write(paragraph)

    
    # Calculating number of personal pronouns and then appending the list
    PersonalPronouns.append(get_personalPronouns(paragraph))
    
    
    sentences = sent_tokenize(paragraph)
    tokens = []
    for sentence in sentences:
        tokens.extend(word_tokenize(sentence))
    
    
    #Removing punctuations from the tokens and saving it to words
    punctuations = set(string.punctuation)
    
    
    
    words = [token for token in tokens if token not in punctuations and token not in ['’','s','ve','t','']]
    
    #removing . between words
    Words = []
    for word in words:
        if '.' in word:
            Words.extend(word.split('.'))
    
        else:
            Words.append(word)
    
    
    
    # getting average length of sentence 
    avgSentencelen = get_avgSentLen(sentences,Words)
    
    AvgSentLength.append(round(avgSentencelen,2))
    
    
    # converting all words to lower case for better fitting
    
    words = [word.lower() for word in Words]
    
    
    #calculating average word length and appending the list
    
    wordLenAvg = get_avgwordlength(words)
    AvgWordLength.append(round(wordLenAvg,2))
    
    # getting count of complex words and appending the list
    
    complex = get_complexCount(words)
    ComplexWordCount.append(complex)
    
    #calculate percent complex words and append perComplex
    
    percent = get_complexPercent(complex,len(words))
    perComplex.append(round(percent,2))
    
    #calculating fog index 
    fog = get_fogIndex(avgSentencelen,percent)
    fogIndex.append(round(fog,2))

    
    # getting word count by removing the nltk stop words
    
    wordCount = get_WordCount(words)
    WordCount.append(wordCount)
    
    # average syllable count
    syllable = get_avgsyllable(words)
    SyllablePerWord.append(round(syllable,2))
    
    
    # now we only need to calculate the four scores
    
    # for that we need to remove stopwords from our words list
    
    words = [word for word in words if word not in StopWords]
    
    # now calculating the four scores and updating the results int the list
    
    positivescore = get_positiveScore(words)
    PositiveScore.append(positivescore)
    
    negativescore = get_negativeScore(words)
    NegativeScore.append(negativescore)
    
    polarityscore = get_polarityScore(positivescore,negativescore)
    PolarityScore.append(round(polarityscore,2))
    
    subjectivityscore = get_subjectivityScore(positivescore,negativescore,words)
    SubjectivityScore.append(round(subjectivityscore,2))
    
    
# now we import urls from input.xlsx and rum calculateALL for each url

urls = pd.read_excel('Input.xlsx')
for id, row in urls.iterrows():
    calculateAll(row['URL'],row['URL_ID'])
    
    
# Now after processing we will add new column to our dataframe urls

urls['Title'] = Title
urls['Positive score'] = PositiveScore
urls['Negative score'] = NegativeScore
urls['Polarity score'] = PolarityScore
urls['Subjectivity score'] = SubjectivityScore
urls['Average words/sentence'] = AvgSentLength
urls['Complex words'] = ComplexWordCount
urls['Complex percent'] = perComplex
urls['Fog index'] = fogIndex
urls['Word count'] = WordCount
urls['Syllables/word'] = SyllablePerWord
urls['Personal Pronouns'] = PersonalPronouns
urls['Average word length'] = AvgWordLength

# finally export dataframe as xlsx file

urls.to_excel('Output.xlsx',index=False)

print("Done")