from datetime import datetime
from multiprocessing import Pool, set_start_method, get_context
import os
import pandas as pd
from random import randint
from scipy.special import softmax
import snscrape.modules.twitter as sntwitter
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification


pd.set_option('max_colwidth', 150)

tweets = []
limit = 50

# roberta global vars
tweet_words = []
# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']
# values = {}
# values['Negative'] = []
# values['Neutral'] = []
# values['Positive'] = []


def run_roberta(tweet):
    # process the tweet
    tweet_proc = proc_tweet(tweet)

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, max_length=512,
            truncation=True, return_tensors='pt')

    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    print("scores: {}".format(scores))

    return scores[0], scores[1], scores[2]

#    for i in range(len(scores)):
#        if labels[i] == 'Negative':
#            values["negative"].append(scores[i])
#            print("negative: {}".format(values["negative"]))
#        elif labels[i] == 'Neutral':
#            values["neutral"].append(scores[i])
#            print("neutral: {}".format(values["neutral"]))
#        else:
#            values["positive"].append(scores[i])
#            print("positive: {}".format(values["positive"]))


def proc_tweet(tweet):
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'

        elif word.startswith('http'):
            word = 'http'

        tweet_words.append(word)
    return " ".join(tweet_words)


def get_tweets(query):
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        # print(vars(tweet))
        # break
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.rawContent])


if __name__ == '__main__':
    set_start_method("spawn")

    # snscrape global vars
    word = "islam"
    month = sys.argv[1]

    query = "islam religion -mma -ufc -makhachev (#islam OR #religion) lang:en until:2021-{}-28 since:2021-{}-01".format(month, month)

    # now = datetime.now()
    # islam
    get_tweets(query)
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
    # muslim/muslims
    # American muslim/muslims
    # hijab/headscarf
    # women in islam/muslim women

    # parallel worker pool
    with get_context("spawn").Pool(4) as pool:
        results = pool.map(run_roberta, df['Tweet'].to_numpy())
    pool.close()

    name_id = "{}.{}.{}".format(word, month, randint(0, 100000))

    outfile_name = f"out/{name_id}/{name_id}.out"

    os.system(f"mkdir out/{name_id}")
    os.system(f"touch out/{name_id}/{name_id}.out")
    os.system("touch out/{name_id}/neg.{}".format(outfile_name))
    os.system("touch out/{name_id}/neu.{}".format(outfile_name))
    os.system("touch out/{name_id}/pos.{}".format(outfile_name))

    with open(outfile_name, "w") as outfile:
        outfile.write("(QUERY): '{}' '{}'\n".format(word.upper(), month))
        outfile.write("{} {} {}\n".format(labels[0], labels[1], labels[2]))

        for result in results:
            for item in result:
                outfile.write("{} ".format(item))
            outfile.write("\n")

    with open("neg.{}".format(outfile_name), "w") as outfile:
        for result in results:
            outfile.write(str(result[0]))
            outfile.write("\n")
    with open("neu.{}".format(outfile_name), "w") as outfile:
        for result in results:
            outfile.write(str(result[1]))
            outfile.write("\n")
    with open("pos.{}".format(outfile_name), "w") as outfile:
        for result in results:
            outfile.write(str(result[2]))
            outfile.write("\n")
