import tweepy
import pandas as pd
from tweepy import OAuthHandler
import contractions
from sklearn import model_selection
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
nltk.download('stopwords')

def transform_data(data_path, X_output, y_output):
    """
    Cleaning text (removing punctuation, stop words, etc.), and splitting into train-test-validation dataset. Divides into equal thirds with random state 42.
    """
    data = pd.read_csv(data_path)
    cleaned_data = clean_data(data)
    
    return split(cleaned_data, X_output, y_output)

def clean_data(data):
    """
    Cleaning text, removing punctuation, stop words, etc.
    """
    data['tweet_text'] = data['tweet_text'].str.replace(r'https://[\w\W].*', '') # removing urls
    data['tweet_text'] = data['tweet_text'].str.replace('[\n]',' ') # removing \n
    
    # expanding contractions
    data['tweet_text'] = data['tweet_text'].map(contractions.fix)
    data['tweet_text'] = data['tweet_text'].str.replace('[\']','')
    
    data['tweet_text'] = data['tweet_text'].str.replace('[^\w\s\n]',' ') # removing unicode characters, punctuation
    data['tweet_text'] = data['tweet_text'].str.replace('(  )',' ') # remove double whitespace
    data['tweet_text'] = data['tweet_text'].str.strip() # remove trailing whitespace
    data['tweet_text'] = data['tweet_text'].str.lower() # changing to lowercase
    data['tweet_text'] = data['tweet_text'].map(unidecode) # standardizing font
    
    # removing stop words
    stop_words = '|'.join(stopwords.words('english'))
    data['tweet_text'] = data['tweet_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data = data.dropna()
    
    return data

def split(data, X_output, y_output):
    """
    Split into train-test-validation dataset. Divides into equal thirds. Random state 42.
    """
    testsize = 2/3 # divides into train and test+val
    valsize = 1/2 # divides test and validation
    
    X_train, X, y_train, y = model_selection.train_test_split(data['tweet_text'], data['party'], test_size=testsize, random_state=42)
    X_test, X_validation, y_test, y_validation = model_selection.train_test_split(X, y, test_size=valsize, random_state=42)
    
    X_train.to_csv(X_output[0], index=False)
    X_test.to_csv(X_output[1], index=False)
    X_validation.to_csv(X_output[2], index=False)
    
    y_train.to_csv(y_output[0], index=False)
    y_test.to_csv(y_output[1], index=False)
    y_validation.to_csv(y_output[2], index=False)
    
    return

def scrape_data(access_token, access_token_secret, api_key, api_key_secret, output):
    """
    Scraping tweets from Twitter.
    """
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True)

    # politician method (chris smith NJ, jefferson van drew NJ have deleted twitter accounts)
    handles = pd.read_excel('data/raw/congress.xlsx') # credit to https://ucsd.libguides.com/congress_twitter/home 
    handles['Link'] = handles['Link'].str.replace('https://twitter.com/', '')

    count = 300
    tweets_lst = []

    for userID in handles['Link']:
        try:
            user = api.user_timeline(screen_name=userID, count=count, include_rts=False, tweet_mode='extended')

        except Exception as e:
            print(userID)
            print(repr(e))
            break

        party = handles.loc[handles['Link'] == userID, 'Party'].iloc[0]
        
        for tweet in user:
            tweets_lst.append((tweet.full_text, party))

    tweets = pd.DataFrame(tweets_lst, columns = ['tweet_text', 'party'])
    tweets.to_csv(output, index=False)
    
    return
