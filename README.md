
## Unit 7 | Assignment - Distinguishing Sentiments

## Background

__Twitter__ has become a wildly sprawling jungle of information&mdash;140 characters at a time. Somewhere between 350 million and 500 million tweets are estimated to be sent out _per day_. With such an explosion of data, on Twitter and elsewhere, it becomes more important than ever to tame it in some way, to concisely capture the essence of the data.

Choose __one__ of the following two assignments, in which you will do just that. Good luck!

## News Mood

In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: __BBC, CBS, CNN, Fox, and New York times__.


The first plot will be and/or feature the following:

* Be a scatter plot of sentiments of the last __100__ tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
* Each plot point will reflect the _compound_ sentiment of a tweet.
* Sort each plot point by its relative timestamp.

The second plot will be a bar plot visualizing the _overall_ sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

* Pull last 100 tweets from each outlet.
* Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet. 
* Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
* Export the data in the DataFrame into a CSV file.
* Save PNG images for each plot.



### Observed Trend

#### 1.  Till 04/03/17, it seems only FOX News has positive overall media sentiment 

#### 2.  Tweets from BBC News have worst overall media sentiment 

#### 3.  The scatter plot provides a general idea of how the individual compound scores reached both highs and lows, but there is too much information to make an overall statement about the data.


```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
% matplotlib inline

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
consumer_key = 'RF125avZqsqHFrWYZs88EYNop'
consumer_secret = 'sSryRp4HttVXySNs5MqbULE9oR9l4Lnkn3ULGnLIPoRxcQDM9A'
access_token = '892258092108333056-9EE89SpCevqFwkg61vBGJoHYqbfb0A5'
access_token_secret = 'Dzuou3NA6QaRipajIUyCl5xFAFVEFUN1iwhztsbrWt9jD'

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())



```


```python
# select news source from twitter
news_source = ["FoxNews", "CNN", "BBCWorld", "CBSNews", "nytimes"]



```


```python
tweet = api.user_timeline("FoxNews", page=0)




```


```python
# store data in dictionary
tweet_data = {"source": [], 
              "text": [], 
              "date": [], 
              "compound": [], 
              "negative": [], 
              "neutral": [], 
              "positive": []}

# grab 100 tweets from each site
for x in range(5):
    
    # loop through news source:
    for source in news_source:
        
        # grab the tweet
        tweets = api.user_timeline(source, page=x)
        
        for tweet in tweets:
            # grab data from tweets API call (20 for each page)
            tweet_data["source"].append(tweet["user"]["name"])
            tweet_data["text"].append(tweet["text"])
            tweet_data["date"].append(tweet["created_at"])
            
            # calculate for sentiment
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            
            # add to data dictionary
            tweet_data["compound"].append(compound)
            tweet_data["positive"].append(pos)
            tweet_data["neutral"].append(neu)
            tweet_data["negative"].append(neg)
            
```


```python
# convert dictionary to dataframe
tweet_df = pd.DataFrame(tweet_data)




```


```python
tweet_df.head()




```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>date</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>source</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.2960</td>
      <td>Wed Apr 04 23:48:26 +0000 2018</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>WATCH: @edhenry sits down with EPA Chief Scott...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.2263</td>
      <td>Wed Apr 04 23:41:21 +0000 2018</td>
      <td>0.091</td>
      <td>0.909</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>RT @FoxNewsResearch: Every president since Ron...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.6249</td>
      <td>Wed Apr 04 23:32:42 +0000 2018</td>
      <td>0.215</td>
      <td>0.785</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>.@JonathanTurley: “If they’re waiting for a co...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5095</td>
      <td>Wed Apr 04 23:18:29 +0000 2018</td>
      <td>0.000</td>
      <td>0.852</td>
      <td>0.148</td>
      <td>Fox News</td>
      <td>.@GovMikeHuckabee: “@POTUS needs to have his m...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Wed Apr 04 23:13:27 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>.@GovMikeHuckabee: “We have borders. Every cou...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert date from string to datetime
tweet_df["date"] = pd.to_datetime(tweet_df["date"])


```


```python
tweet_df.head()

tweet_df.to_csv("sentiment.csv",sep=',')

```


```python
# sort dataframe (hint: use .sort_values(by="column name", inplace=True))
tweet_df.sort_values(by="date", ascending=True)



```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>date</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>source</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459</th>
      <td>-0.6908</td>
      <td>2018-04-03 16:08:03</td>
      <td>0.266</td>
      <td>0.734</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>RT @BBCDomC: BBC News Reality Check: Has Londo...</td>
    </tr>
    <tr>
      <th>458</th>
      <td>0.0000</td>
      <td>2018-04-03 16:54:08</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Israel and Saudi Arabia: The relationship emer...</td>
    </tr>
    <tr>
      <th>457</th>
      <td>-0.8225</td>
      <td>2018-04-03 17:36:27</td>
      <td>0.521</td>
      <td>0.479</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Waterslide designer arrested after boy killed ...</td>
    </tr>
    <tr>
      <th>456</th>
      <td>0.0000</td>
      <td>2018-04-03 17:54:02</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Trump vows to put US military in charge of sou...</td>
    </tr>
    <tr>
      <th>455</th>
      <td>-0.2960</td>
      <td>2018-04-03 17:57:13</td>
      <td>0.306</td>
      <td>0.694</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Tearful reunion with missing daughter https://...</td>
    </tr>
    <tr>
      <th>454</th>
      <td>0.5574</td>
      <td>2018-04-03 18:15:59</td>
      <td>0.000</td>
      <td>0.714</td>
      <td>0.286</td>
      <td>BBC News (World)</td>
      <td>Nigerian 'migrant hero' baptised by Pope after...</td>
    </tr>
    <tr>
      <th>453</th>
      <td>0.0000</td>
      <td>2018-04-03 19:00:28</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>RT @BBCNorthAmerica: Donald Trump says he will...</td>
    </tr>
    <tr>
      <th>452</th>
      <td>0.2263</td>
      <td>2018-04-03 19:30:22</td>
      <td>0.000</td>
      <td>0.826</td>
      <td>0.174</td>
      <td>BBC News (World)</td>
      <td>Man pardoned by hotel after seagull and pepper...</td>
    </tr>
    <tr>
      <th>451</th>
      <td>-0.2023</td>
      <td>2018-04-03 19:38:36</td>
      <td>0.184</td>
      <td>0.816</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Swastika removed from controversial Nazi bell ...</td>
    </tr>
    <tr>
      <th>450</th>
      <td>0.4796</td>
      <td>2018-04-03 20:07:31</td>
      <td>0.119</td>
      <td>0.677</td>
      <td>0.204</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: GOAL! Juventus 0-2 Real Madrid\n...</td>
    </tr>
    <tr>
      <th>449</th>
      <td>-0.5574</td>
      <td>2018-04-03 20:16:48</td>
      <td>0.340</td>
      <td>0.660</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>'Shots fired' near YouTube HQ in US https://t....</td>
    </tr>
    <tr>
      <th>448</th>
      <td>0.0000</td>
      <td>2018-04-03 20:33:26</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Latest from San Bruno: https://t.co/a5RDa4FFVl...</td>
    </tr>
    <tr>
      <th>447</th>
      <td>0.4019</td>
      <td>2018-04-03 20:53:17</td>
      <td>0.000</td>
      <td>0.881</td>
      <td>0.119</td>
      <td>BBC News (World)</td>
      <td>RT @BBCBreaking: What we know so far:\n\n- Pol...</td>
    </tr>
    <tr>
      <th>446</th>
      <td>-0.8481</td>
      <td>2018-04-03 21:52:11</td>
      <td>0.305</td>
      <td>0.695</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>RT @BBCBreaking: Female suspect dead in shooti...</td>
    </tr>
    <tr>
      <th>445</th>
      <td>-0.7579</td>
      <td>2018-04-03 22:21:51</td>
      <td>0.394</td>
      <td>0.606</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>YouTube shooting: Four shot at California HQ, ...</td>
    </tr>
    <tr>
      <th>444</th>
      <td>0.0000</td>
      <td>2018-04-03 22:44:59</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>"You'd think that after... Las Vegas, Parkland...</td>
    </tr>
    <tr>
      <th>443</th>
      <td>0.0000</td>
      <td>2018-04-03 22:54:36</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>US plans 25% tariffs on 1,300 Chinese imports ...</td>
    </tr>
    <tr>
      <th>442</th>
      <td>-0.4215</td>
      <td>2018-04-03 23:41:43</td>
      <td>0.359</td>
      <td>0.641</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Russia's bitter taste of capitalism https://t....</td>
    </tr>
    <tr>
      <th>441</th>
      <td>-0.5106</td>
      <td>2018-04-03 23:44:16</td>
      <td>0.292</td>
      <td>0.708</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>The doctor who really feels his patients' pain...</td>
    </tr>
    <tr>
      <th>440</th>
      <td>-0.2263</td>
      <td>2018-04-03 23:46:56</td>
      <td>0.174</td>
      <td>0.826</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>The 'baby warehouses' which cater for Israel's...</td>
    </tr>
    <tr>
      <th>359</th>
      <td>0.0000</td>
      <td>2018-04-03 23:49:21</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Martin Luther King: How diverse is US politics...</td>
    </tr>
    <tr>
      <th>358</th>
      <td>-0.6486</td>
      <td>2018-04-04 00:06:07</td>
      <td>0.371</td>
      <td>0.629</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Winnie Mandela - the young mother who refused ...</td>
    </tr>
    <tr>
      <th>357</th>
      <td>0.0000</td>
      <td>2018-04-04 01:30:16</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>WPP boss Martin Sorrell faces misconduct inves...</td>
    </tr>
    <tr>
      <th>356</th>
      <td>0.0000</td>
      <td>2018-04-04 02:14:20</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>RT @BBCNewsAsia: What are the most common myth...</td>
    </tr>
    <tr>
      <th>355</th>
      <td>-0.2263</td>
      <td>2018-04-04 02:29:37</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>RT @BBCNewsAsia: The story of one Chinese coup...</td>
    </tr>
    <tr>
      <th>354</th>
      <td>-0.3818</td>
      <td>2018-04-04 03:02:57</td>
      <td>0.133</td>
      <td>0.867</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>RT @BBCNewsAsia: Still annoyed about Malaysia ...</td>
    </tr>
    <tr>
      <th>353</th>
      <td>0.3400</td>
      <td>2018-04-04 03:13:54</td>
      <td>0.000</td>
      <td>0.870</td>
      <td>0.130</td>
      <td>BBC News (World)</td>
      <td>RT @bbcdavideades: Fifty years on from #Martin...</td>
    </tr>
    <tr>
      <th>352</th>
      <td>0.0000</td>
      <td>2018-04-04 03:28:50</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Ukraine: On patrol with the far-right National...</td>
    </tr>
    <tr>
      <th>351</th>
      <td>-0.6705</td>
      <td>2018-04-04 03:37:59</td>
      <td>0.478</td>
      <td>0.522</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Commonwealth Games: Delegate accused of assaul...</td>
    </tr>
    <tr>
      <th>350</th>
      <td>0.0000</td>
      <td>2018-04-04 04:01:09</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Russian spy: Chemical watchdog to meet at Russ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>185</th>
      <td>0.0000</td>
      <td>2018-04-04 23:21:50</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>Watch live: Sally Yates, the former acting att...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.0000</td>
      <td>2018-04-04 23:21:50</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>Watch live: Sally Yates, the former acting att...</td>
    </tr>
    <tr>
      <th>141</th>
      <td>-0.1531</td>
      <td>2018-04-04 23:25:33</td>
      <td>0.242</td>
      <td>0.758</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Spanish royals in awkward moment https://t.co/...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>-0.1531</td>
      <td>2018-04-04 23:25:33</td>
      <td>0.242</td>
      <td>0.758</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Spanish royals in awkward moment https://t.co/...</td>
    </tr>
    <tr>
      <th>184</th>
      <td>0.4215</td>
      <td>2018-04-04 23:26:40</td>
      <td>0.000</td>
      <td>0.865</td>
      <td>0.135</td>
      <td>The New York Times</td>
      <td>George Nader, an adviser to the United Arab Em...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.4215</td>
      <td>2018-04-04 23:26:40</td>
      <td>0.000</td>
      <td>0.865</td>
      <td>0.135</td>
      <td>The New York Times</td>
      <td>George Nader, an adviser to the United Arab Em...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>-0.4588</td>
      <td>2018-04-04 23:27:08</td>
      <td>0.103</td>
      <td>0.897</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>RT @jestei: Asked if she Is the leader of the ...</td>
    </tr>
    <tr>
      <th>183</th>
      <td>-0.4588</td>
      <td>2018-04-04 23:27:08</td>
      <td>0.103</td>
      <td>0.897</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>RT @jestei: Asked if she Is the leader of the ...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.0000</td>
      <td>2018-04-04 23:29:53</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>George Nader has connections to both the Persi...</td>
    </tr>
    <tr>
      <th>182</th>
      <td>0.0000</td>
      <td>2018-04-04 23:29:53</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>George Nader has connections to both the Persi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.6249</td>
      <td>2018-04-04 23:32:42</td>
      <td>0.215</td>
      <td>0.785</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>.@JonathanTurley: “If they’re waiting for a co...</td>
    </tr>
    <tr>
      <th>102</th>
      <td>-0.6249</td>
      <td>2018-04-04 23:32:42</td>
      <td>0.215</td>
      <td>0.785</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>.@JonathanTurley: “If they’re waiting for a co...</td>
    </tr>
    <tr>
      <th>122</th>
      <td>0.0000</td>
      <td>2018-04-04 23:34:50</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CNN</td>
      <td>Yolanda Renee King, the eldest granddaughter o...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0000</td>
      <td>2018-04-04 23:34:50</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CNN</td>
      <td>Yolanda Renee King, the eldest granddaughter o...</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.0772</td>
      <td>2018-04-04 23:38:06</td>
      <td>0.118</td>
      <td>0.717</td>
      <td>0.165</td>
      <td>The New York Times</td>
      <td>RT @NYTSports: Tiger Woods is back at the Mast...</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0.0772</td>
      <td>2018-04-04 23:38:06</td>
      <td>0.118</td>
      <td>0.717</td>
      <td>0.165</td>
      <td>The New York Times</td>
      <td>RT @NYTSports: Tiger Woods is back at the Mast...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>-0.0516</td>
      <td>2018-04-04 23:39:32</td>
      <td>0.159</td>
      <td>0.691</td>
      <td>0.150</td>
      <td>CBS News</td>
      <td>On the anniversary of Martin Luther King Jr.'s...</td>
    </tr>
    <tr>
      <th>160</th>
      <td>-0.0516</td>
      <td>2018-04-04 23:39:32</td>
      <td>0.159</td>
      <td>0.691</td>
      <td>0.150</td>
      <td>CBS News</td>
      <td>On the anniversary of Martin Luther King Jr.'s...</td>
    </tr>
    <tr>
      <th>121</th>
      <td>0.5267</td>
      <td>2018-04-04 23:40:34</td>
      <td>0.000</td>
      <td>0.793</td>
      <td>0.207</td>
      <td>CNN</td>
      <td>Liverpool stun Manchester City to take command...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.5267</td>
      <td>2018-04-04 23:40:34</td>
      <td>0.000</td>
      <td>0.793</td>
      <td>0.207</td>
      <td>CNN</td>
      <td>Liverpool stun Manchester City to take command...</td>
    </tr>
    <tr>
      <th>101</th>
      <td>-0.2263</td>
      <td>2018-04-04 23:41:21</td>
      <td>0.091</td>
      <td>0.909</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>RT @FoxNewsResearch: Every president since Ron...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.2263</td>
      <td>2018-04-04 23:41:21</td>
      <td>0.091</td>
      <td>0.909</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>RT @FoxNewsResearch: Every president since Ron...</td>
    </tr>
    <tr>
      <th>180</th>
      <td>-0.4404</td>
      <td>2018-04-04 23:47:29</td>
      <td>0.132</td>
      <td>0.868</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>RT @melbournecoal: NEW: David Smith, the chair...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>-0.4404</td>
      <td>2018-04-04 23:47:29</td>
      <td>0.132</td>
      <td>0.868</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>RT @melbournecoal: NEW: David Smith, the chair...</td>
    </tr>
    <tr>
      <th>100</th>
      <td>-0.2960</td>
      <td>2018-04-04 23:48:26</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>WATCH: @edhenry sits down with EPA Chief Scott...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-0.2960</td>
      <td>2018-04-04 23:48:26</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>WATCH: @edhenry sits down with EPA Chief Scott...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.0000</td>
      <td>2018-04-04 23:50:30</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>What next for Trump's trade agenda? https://t....</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.0000</td>
      <td>2018-04-04 23:50:30</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>What next for Trump's trade agenda? https://t....</td>
    </tr>
    <tr>
      <th>120</th>
      <td>-0.8225</td>
      <td>2018-04-04 23:53:35</td>
      <td>0.323</td>
      <td>0.677</td>
      <td>0.000</td>
      <td>CNN</td>
      <td>Sheriff on the fatal wreck after a family's SU...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.8225</td>
      <td>2018-04-04 23:53:35</td>
      <td>0.323</td>
      <td>0.677</td>
      <td>0.000</td>
      <td>CNN</td>
      <td>Sheriff on the fatal wreck after a family's SU...</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>




```python
# filter data
BBC = tweet_df[tweet_df["source"] == "BBC News (World)"]
CNN = tweet_df[tweet_df["source"] == "CNN"]
NYT = tweet_df[tweet_df["source"] == "The New York Times"]
FOX = tweet_df[tweet_df["source"] == "Fox News"]
CBS = tweet_df[tweet_df["source"] == "CBS News"]
```


```python
# matplotlib
# BBC News world
plt.scatter(np.arange(-len(BBC), 0, 1), BBC["compound"], edgecolor="black", marker="o", color="skyblue", s=100, alpha=0.8, label="BBC")
plt.scatter(np.arange(-len(CNN), 0, 1), CNN["compound"], edgecolor="black", marker="o", color="red", s=100, alpha=0.8, label="CNN")
plt.scatter(np.arange(-len(NYT), 0, 1), NYT["compound"], edgecolor="black", marker="o", color="yellow", s=100, alpha=0.8, label="NYT ")
plt.scatter(np.arange(-len(FOX), 0, 1), FOX["compound"], edgecolor="black", marker="o", color="blue", s=100, alpha=0.8, label="FOX")
plt.scatter(np.arange(-len(CBS), 0, 1), CBS["compound"], edgecolor="black", marker="o", color="green", s=100, alpha=0.8, label="CBS")



#graph properties
plt.title("Sentiment Analysis of Media Tweets (04/03/17)")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.xlim([-105, 5])
plt.xticks([-100, -80, -60, -40, -20, 0], [100, 80, 60, 40, 20, 0])
plt.savefig("Sentiment_scatter.png")
```


![png](output_13_0.png)



```python
# group your sources and take mean of compound
grouped_mean = tweet_df.groupby(["source"]).mean()["compound"]
```


```python
grouped_mean
```




    source
    BBC News (World)     -0.122237
    CBS News             -0.102156
    CNN                  -0.078656
    Fox News              0.023760
    The New York Times   -0.039914
    Name: compound, dtype: float64




```python
BBC_mean = grouped_mean["BBC News (World)"]
CNN_mean = grouped_mean["CNN"]
NYT_mean = grouped_mean["The New York Times"]
FOX_mean = grouped_mean["Fox News"]
CBS_mean = grouped_mean["CBS News"]

```


```python
tweets_polarity = [BBC_mean, CNN_mean, NYT_mean, FOX_mean, CBS_mean]

# generate bars for barplot
fig, ax = plt.subplots()

#generate indexes for bar plots
ind = np.arange(len(tweets_polarity))
width = 0.75

BBC_bar = ax.bar(ind[0], tweets_polarity[0], width, color="skyblue")
CNN_bar = ax.bar(ind[1], tweets_polarity[1], width, color="red")
NYT_bar = ax.bar(ind[2], tweets_polarity[2], width, color="yellow")
FOX_bar = ax.bar(ind[3], tweets_polarity[3], width, color="blue")
CBS_bar = ax.bar(ind[4], tweets_polarity[4], width, color="green")
# Generate labels for each news source
def autolabelneg(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, -0.05,
                '-%.2f' % float(height), 
                ha='center', va='bottom')
        
def autolabelpos(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                '+%.2f' % float(height), 
                ha='center', va='bottom')
        
# call function to apply on our bars
autolabelneg(BBC_bar)
autolabelneg(CNN_bar)
autolabelneg(NYT_bar)
autolabelneg(CBS_bar)
autolabelpos(FOX_bar)
# set graph parameters
ax.set_ylabel("Tweet Polarity")
ax.set_title("Overall Media Sentiment based on Twitter (04/03/17)")
ax.set_xticks(ind)
ax.set_xticklabels(["BBC", "CNN", "New York Times", "FOX", "CBS"])
ax.set_autoscaley_on(True)
ax.grid(True)

plt.savefig("Sentiment_bar.png")
```


![png](output_17_0.png)

