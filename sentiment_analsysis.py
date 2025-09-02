
# Year 12 Level Sentiment Analyzer using NLTK's VADER

import pandas as pd # to read CSV file as pd means you dont have to reference pandas everytime you can just say pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk #   nltk is this huge libary still very lightweight compared to others used to process text in lots of different ways the main ones are 
# 1) Tokenization (splitting text into words)
# 2) Part-of-speech tagging (is this word a noun, verb etc).
# 3) sentiment analysis
# 4) Removing stop words (common words like “the” or “and”)

# - a transformer isnt just a large corpus its a type of nural network architecture designed for processing sequences of data ( like text) - throughout ITS TRAINED CORPUS its made up of three things 1) Encode procceses inputs into embedeed sequences (remember first numbers then sequences)r 2) Decoder- generates output sequence 3) Attentionn Laters - prediction of sentiment

# download the vader lexicon if not already installed. =
nltk.download('vader_lexicon')
#vadar lexicon is just a predefined dictionary of words INSIDE THE LIBARY NLTK- where each word in the "lexicon" has a sentiment score from -4 to 4 - understand vadar is kinda outdated now in the sense its more simple and a more limted trasfomrer (but it's not really a tranformer its too basic)
#RoBERTa/BERT is advanced tranfromer based modelling : Can handle subtle context, negation, sarcasm, and long text.
#Hugging Face is a platfrom / libary that hosts thousands of pretrained tranfromer models BERT, RoBERTA, GPT_ makes it easy to download and use start if gthe art NLP models Its somewhat free but for large scale that use GPU (for parallelism + speed + memory bandwidth) its paid


# loading the CSV file
df = pd.read_csv('reviews.csv') #reads the CSV file into a data frame - df its basically another sturcure in python like a database



# ceating sentiment analyzer using libary
sia = SentimentIntensityAnalyzer()

# analysing sentiments
def analyze_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        sentiment = 'Positive'
    elif score < -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, score

# Applying sentiment to every review
df['sentiment'], df['compound_score'] = zip(*df['review_text'].apply(analyze_sentiment))

# Print Individual Results for each review
print("----- Individual Review Analysis -----\n")
for index, row in df.iterrows():
    print(f"Review: {row['review_text']}")
    print(f"Sentiment: {row['sentiment']} | Score: {row['compound_score']}")
    print("-" * 50)

# Printing Summary Table 
summary = df['sentiment'].value_counts()
print("\n----- Sentiment Summary -----")
print(summary.to_string())
