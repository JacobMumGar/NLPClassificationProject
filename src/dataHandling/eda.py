from collections import Counter
from nltk.corpus import stopwords
from nltk import ngrams
from src.dataHandling.dataPreprocessing import preprocess
import matplotlib.pyplot as plt
import seaborn as sns

def speechLengths(df):
    from src.dataHandling.dataPreprocessing import corpus

    #counts the lengths of each party to figure out the average speech length 
    counts = df.groupby("party").agg("count")
    print(counts)
    df["cleaned"] = df["transcript"].apply(preprocess)
    df["wordList"] = df["cleaned"].apply(corpus)
    df["length"] = df["wordList"].apply(len)

    #return the average speech length for each party
    return df.groupby("party")["length"].mean()

def wordCounts(df):
    from src.dataHandling.dataPreprocessing import corpus 

    #create column in df containing a list of all the words in the speech
    df["cleaned"] = df["transcript"].apply(preprocess)
    df["wordList"] = df["cleaned"].apply(corpus)

    #make list of all the words across all speeches
    allBigrams = []
    stopWords = set(stopwords.words("english"))
    myList = ["q", "president"]

    #
    for wordList in df["wordList"]:
        biGrams = list(ngrams(wordList, 2))
        filtered = [(word,word2) for (word,word2) in biGrams if word not in stopWords and word not in myList and word2 not in myList and word2 not in stopWords]
        allBigrams += filtered

    #return the most common words
    mostCommon = Counter(allBigrams).most_common(10)
    return mostCommon

def graphs(top10,party):
    #make a figure with the top 10 used bigrams for visualisation purposes
    bigrams = []
    frequency = []
    for words, count in top10:
        bigrams.append(words[0] + " " + words[1]) 
        frequency.append(count)

    sns.barplot(x=frequency, y=bigrams)
    plt.title(f"{party} Top 10 Used Bigrams")
    plt.xlabel("Frequency")
    plt.show()