import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

#process_tweet
'''
> cleans the text, 
> tokenizes it into separate words,
> removes stopwords, 
> and converts words to stems. 
'''

def process_tweet(tweet):
    '''
    Funtion to process tweets:
    --------------------------
    Input:
    ------
        tweet: A string containing a tweet.
    Output:
    -------
        tweets_clean: A list of words from the processed tweet.
    '''
    
    #for stemming
    stemmer = PorterStemmer()
    
    #to remove stopwords
    stopwords_english = stopwords.words("english") 
    
    #remove stock market tickers like $GE, $20, etc;
    tweet = re.sub(r'\$\w*', '', tweet)
    
    #remove old style retweet text RT
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, 
                               strip_handles=True,
                               reduce_len=True)
    
    tweet_tokens = tokenizer.tokenize(tweet)
    
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

#build_freqs
'''
> counts how often a word in the 'corpus' (the entire set of tweets) was associated with a positive label '1' or a negative label '0', 
> then builds the freqs dictionary, where each key is a (word,label) tuple, and the value is the count of its frequency within the corpus of tweets.
'''
def build_freqs(tweets, ys):
    import numpy as np
    """
    Build frequencies.
    ------------------
    Input:
    ------
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
    -------
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
    
    