import re
import nltk
from sklearn.model_selection import train_test_split

def preprocess(speech):
    #Clean the speech field by removing html tags, new line and carriage characters and trailing whitespace
    speech = re.sub(r"<.*?>"," ", speech)
    speech = re.sub("\n", " ", speech)
    speech = re.sub("\r", " ", speech)
    speech = speech.strip().lower()
    return speech

def trainTestSplit(df):
    dfTrain, dfTest = train_test_split(df, test_size=0.2, random_state=123, stratify=df["party"])
    return dfTrain, dfTest

def chunkify(df, chunkSize, minChunk=75):
    chunkList = []
    
    for _, row in df.iterrows():
        words = nltk.word_tokenize(row["transcript"])
        for idx in range(0, len(words), chunkSize):
            chunk = words[idx:idx+chunkSize]
            if len(chunk) >= 75:
                chunkList.append([" ".join(chunk),row["party"]])

    return chunkList