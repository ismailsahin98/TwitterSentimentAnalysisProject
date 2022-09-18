from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep

"""
Tip : Dont Use your personal twitter acoount for web scrapping, create or use any twitter account for scrapping & development purposes only.
"""


my_user = "<Enter Your Twitter Username Here>"  #Twitter Username 
my_pass = "<Enter Your Twitter Password Here>"  #Twitter Password


search_item = "Europe inflation"   #Topic that what you want to search

PATH = "chromedriver.exe"        # Attention This chromedriver is just for Chrome version 105
driver = webdriver.Chrome(PATH)
driver.get("https://twitter.com/i/flow/login")
sleep(3)
user_id = driver.find_element(By.XPATH,"//input[@type='text']")
user_id.send_keys(my_user)
user_id.send_keys(Keys.ENTER)
sleep(3)
user_password = driver.find_element(By.XPATH,"//input[@type='password']")
user_password.send_keys(my_pass)
user_password.send_keys(Keys.ENTER)
sleep(5)

search = driver.find_element(By.XPATH,"//input[@enterkeyhint='search']")
search.send_keys(search_item)
search.send_keys(Keys.ENTER)
sleep(5)

#Scrapping Tweets

tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
all_tweets = set()
while True:
    for tweet in tweets:
        all_tweets.add(tweet.text)

    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    sleep(3)
    tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
    sleep(1)
    if len(all_tweets)>150:
        break

all_tweets = list(all_tweets)

len(all_tweets)

#Cleaning Tweets
import pandas as pd
import re
import nltk 
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

df = pd.DataFrame(all_tweets, columns=["tweets"])
df['tweets'] = df['tweets'].apply(lambda x: " ".join(x.lower() for x in x.split())) #lower tweet text

def CleanTweets(tweet):   #Stopwords, links, # and @ Removal
    cleanTweet = re.sub(r"@[a-zA-Z0-9]+","",tweet)  
    cleanTweet = re.sub(r"#[a-zA-Z0-9\s]+","",cleanTweet)
    cleanTweet = re.sub(r"https[^\s]+","",cleanTweet)
    cleanTweet = " ".join(word for word in cleanTweet.split() if word not in stop_words)
    return cleanTweet

df["cleanedTweets"] = df["tweets"].apply(CleanTweets)
df['cleanedTweets'] = df['cleanedTweets'].str.replace('[^\w\s]','') #Punctuation Removal 
# df['cleanedTweets'] = df['cleanedTweets'].str.replace('\d','') #Numbers Removal ---> i don't use it because topic is inflation but you can use it for any topic you want!



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



