import tweepy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

#Twitter Developer API Auth
access_token = "<twitter developer API access token here>"
access_token_secret = "<twitter developer API access token secret here>"
consumer_key = "<twitter developer consumer key here>"
consumer_key_secret = "<twitter developer consumer key secret here>"
auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)

#Getting Tweets
api = tweepy.API(auth)
tweets = api.search_tweets(q="europe inflation",  lang = "en", result_type = "recent", count = 100)
df = pd.DataFrame()
df["tweet"] = [tweet.text for tweet in tweets]
df["retweet_count"] = [tweet.retweet_count for tweet in tweets]
df["user_followers_count"] = [tweet.author.followers_count for tweet in tweets]
df["user_location"] = [tweet.author.location for tweet in tweets]
df["sources"] = [tweet.source for tweet in tweets]

#Cleaning Tweets
df['cleanedTweets'] = df['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split())) #Lowercase

def cleanTweets(tweets): #url, @ and # removal
    clean_tweet = re.sub(r"https([^\s]+)|@([^\s]+)","",tweets)
    clean_tweet = re.sub(r"#[a-zA-Z0-9\s]+","",clean_tweet)
    return clean_tweet
df["cleanedTweets"] = df["cleanedTweets"].apply(cleanTweets)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['cleanedTweets'] = df['cleanedTweets'].str.replace('[^\w\s]','')#Punctuation removal
df['cleanedTweets'] = df['cleanedTweets'].apply(lambda x: " ".join(x for x in x.split() if x not in sw)) #Stopwords
df['cleanedTweets'] = df['cleanedTweets'].str.replace('rt','') #RT remove
df['cleanedTweets'] = df['cleanedTweets'].str.replace('\d','') #Numbers Removal


#Polarity & Subjectivity Detection and Analysis

from textblob import TextBlob
import matplotlib.pyplot as plt

def calculatePolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def calculateSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def segmentation(tweet):
    if tweet > 0:
        return "Positive"
    if tweet < 0:
        return "Negative"
    else:
        return "Neutral"

df["tPolarity"] = df["cleanedTweets"].apply(calculatePolarity) 
df["tSegmentation"] = df ["tPolarity"].apply(segmentation)
df["tSubjectivity"] = df["cleanedTweets"].apply(calculateSubjectivity)


print(df.pivot_table(index = ['tSegmentation'],aggfunc={'tSegmentation':'count'}))

#Sorting Positive Tweets first
positive = df.sort_values(by=["tPolarity"],ascending=False)[["tweet","cleanedTweets","tPolarity","tSegmentation","tSubjectivity"]]
positive.to_excel("positiveSorting.xlsx")


#Sorting Negative Tweets
negative = df.sort_values(by=["tPolarity"],ascending=True)[["tweet","cleanedTweets","tPolarity","tSegmentation","tSubjectivity"]]
negative.to_excel("negativeSorting.xlsx")

#Visualization
from wordcloud import WordCloud
import seaborn as sns

sns.countplot(data = df, x='tSegmentation')
plt.show(block=True)

sns.set_style("whitegrid")
sns.scatterplot(data = df, x = 'tPolarity', y = 'tSubjectivity', s = 75, hue = 'tSegmentation')
plt.show(block=True)

source_freq = df.groupby("sources").count()["tweet"] #Tweet Sources 
source_freq.head()
source_freq.plot.bar(x = "Sources", y="tweet")
plt.show(block = True)


consolidated_tweets = "".join(word for word in df["cleanedTweets"])


wordcloud = WordCloud(background_color="black",width=1900, height=1000,).generate(consolidated_tweets)
fig = plt.figure(figsize=(19,10), dpi=10000)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
wordcloud.to_image().show()
wordcloud.to_file("wordcloud.png")









#Polarity & Subjectivity Detection and Analysis

from textblob import TextBlob
import matplotlib.pyplot as plt

def calculatePolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def calculateSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def segmentation(tweet):
    if tweet > 0:
        return "Positive"
    if tweet < 0:
        return "Negative"
    else:
        return "Neutral"

df["tPolarity"] = df["cleanedTweets"].apply(calculatePolarity) 
df["tSegmentation"] = df ["tPolarity"].apply(segmentation)
df["tSubjectivity"] = df["cleanedTweets"].apply(calculateSubjectivity)


print(df.pivot_table(index = ['tSegmentation'],aggfunc={'tSegmentation':'count'}))

#Sorting Positive Tweets first
positive = df.sort_values(by=["tPolarity"],ascending=False)[["tweets","cleanedTweets","tPolarity","tSegmentation","tSubjectivity"]]
positive.to_excel("positiveSorting.xlsx")


#Sorting Negative Tweets
negative = df.sort_values(by=["tPolarity"],ascending=True)[["tweets","cleanedTweets","tPolarity","tSegmentation","tSubjectivity"]]
negative.to_excel("negativeSorting.xlsx")

#Visualization
from wordcloud import WordCloud
import seaborn as sns

sns.countplot(data = df, x='tSegmentation')
plt.show(block=True)

sns.set_style("whitegrid")
sns.scatterplot(data = df, x = 'tPolarity', y = 'tSubjectivity', s = 75, hue = 'tSegmentation')
plt.show(block=True)

consolidated_tweets = "".join(word for word in df["cleanedTweets"])


wordcloud = WordCloud(background_color="black",width=1900, height=1000,).generate(consolidated_tweets)
fig = plt.figure(figsize=(10,10), dpi=10000)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show(block = True)



