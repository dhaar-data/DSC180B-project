import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

sys.path.insert(0, 'src')
import etl

def conduct_eda(input_path, output_path):
    raw = pd.read_csv(input_path)
    tweets = pd.read_csv(input_path)
    etl.clean_data(tweets)
    
    raw['party'].replace({'D': 'Democrat', 'R': 'Republican'}, inplace=True)
    tweets['party'].replace({'D': 'Democrat', 'R': 'Republican'}, inplace=True)
    
    party_count = tweets['party'].value_counts().to_dict()
    
    print('Number of tweets:', len(tweets))
    print('Avg. number of tweets per user:', len(tweets) / 538)
    print('Number of tweets per party:\n', 'Democrat:', party_count['Democrat'], ', Republican:', party_count['Republican'])
    
    raw['num_words'] = raw['tweet_text'].str.split().apply(len)
    
    # plot number of words in a tweet by party
    sns.histplot(raw[raw['party'].str.contains('Democrat')]['num_words'], color='blue', alpha=0.5, bins=10, edgecolor='blue', stat='percent')
    sns.histplot(raw[raw['party'].str.contains('Republican')]['num_words'], color='red', alpha=0.5, bins=10, edgecolor='red', stat='percent')
    plt.legend(['Democrat', 'Republican'])
    plt.xlabel('Number of words in a tweet')
    plt.ylabel('Frequency (%)')
    plt.title('Number of words in a tweet by party')
    plt.savefig(output_path[0])
    
    # plot most frequent words by party
    dems = tweets[tweets['party'].str.contains('D')]['tweet_text']
    reps = tweets[tweets['party'].str.contains('R')]['tweet_text']
    
    keys = list(dems.str.split(expand=True).stack().value_counts()[1:11].to_dict().keys())
    vals = list(dems.str.split(expand=True).stack().value_counts()[1:11].to_dict().values())
    sns.barplot(x=keys, y=vals, dodge=False)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('Most frequent words used by Democrats')
    plt.savefig(output_path[1])
    
    keys = list(reps.str.split(expand=True).stack().value_counts()[1:11].to_dict().keys())
    vals = list(reps.str.split(expand=True).stack().value_counts()[1:11].to_dict().values())
    sns.barplot(x=keys, y=vals, dodge=False)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('Most frequent words used by Republicans')
    plt.savefig(output_path[2])
