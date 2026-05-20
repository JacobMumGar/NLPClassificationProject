import re
import nltk
from sklearn.model_selection import train_test_split

def preprocess(speech):
    #Clean the speech field by removing html tags, new line and carriage characters and trailing whitespace
    speech = re.sub(r"<.*?>"," ", speech)
    speech = re.sub("\n", " ", speech)
    speech = re.sub("\r", " ", speech)
    speech = speech.strip().lower()
    speech = re.sub(r"[^a-z\s]", "", speech)
    return speech

def corpus(speech):
    #tokenise a speech into individual words, and return the word list
    wordList = nltk.word_tokenize(speech)
    return(wordList)

def trainTestSplit(df):
    #splitting the data into training and test using sklearn module, stratifying since classes arent of equal size
    dfTrain, dfTest = train_test_split(df, test_size=0.2, random_state=123, stratify=df["party"])
    return dfTrain, dfTest