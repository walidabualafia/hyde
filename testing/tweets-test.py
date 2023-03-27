import os
import sys
from datetime import datetime
from multiprocessing import Pool, set_start_method, get_context
from random import randint

import pandas as pd
import snscrape.modules.twitter as sntwitter
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pd.set_option('max_colwidth', 150)

tweets = []
limit = 25

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']


def run_roberta(tweet):
    """
    Run sentiment analysis on the given tweet using RoBERTa model.
    """
    tweet_proc = proc_tweet(tweet)

    encoded_tweet = tokenizer(tweet_proc, max_length=512, truncation=True, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return scores[0], scores[1], scores[2]


def proc_tweet(tweet):
    """
    Preprocess the tweet by replacing usernames and
    URLs with generic placeholders.
    """
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)

    return " ".join(tweet_words)


def get_tweets(query):
    """
    Scrape tweets based on the provided query.
    """
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.rawContent])


if __name__ == '__main__':
    set_start_method("spawn")

    word = "islam"
    month = sys.argv[1]
    year = sys.argv[2]

    query = f"islam religion -mma -ufc -makhachev (#islam OR #religion) lang:en until:{year}-{month}-28 since:{year}-{month}-01"

    get_tweets(query)
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

    with get_context("spawn").Pool(16) as pool:
        results = pool.map(run_roberta, df['Tweet'].to_numpy())
    pool.close()

    name_id = f"{word}.{month}.{year}.{randint(0, 100000)}"
    out_dir = f"out/{name_id}"
    outfile_name = f"{out_dir}/{name_id}.out"

    os.makedirs(out_dir, exist_ok=True)

    with open(outfile_name, "w") as outfile:
        outfile.write(f"(QUERY): '{word.upper()}' '{month}'\n")
        outfile.write(f"{labels[0]} {labels[1]} {labels[2]}\n")

        for result in results:
            for item in result:
                outfile.write(f"{item} ")
            outfile.write("\n")

    for i, label in enumerate(labels):
        with open(f"{out_dir}/{label.lower()}.{name_id}.out", "w") as outfile:
            for result in results:
                outfile.write(str(result[i]))
                outfile.write("\n")

