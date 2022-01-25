# !pip install nltk
# !pip install re
# !pip install kaggle
# !pip install gensim
import pandas as pd
import numpy as np
import os
import re
import spacy
import gensim.downloader
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPooling1D, GlobalMaxPooling1D, Flatten, Input, Dropout
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import EarlyStopping

PATH = r'D:\uchi\2021Fall\PPHA30536_Data and Programming for Public Policy II\final-project-tianling-luo'

"""
(Can skip this step if one do not want to use kaggle API)
If using kaggle API, please uncomment the following 4 lines
"""
# import kaggle
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files('lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
#                                   path=os.path.join(PATH, 'Data'), unzip=True)

# read data from csv file
def read_csv(fname):
    df = pd.read_csv(os.path.join(PATH, 'Data', fname))
    return df

# label the reviews by 1 and 0
def label_sentiment(df):
    df['label'] = 0
    for ind in df.index:
        if df.loc[ind, ['sentiment']].values[0] == 'positive':
            df.loc[ind, ['label']] = 1
        if df.loc[ind, ['sentiment']].values[0] == 'negative':
            df.loc[ind, ['label']] = 0
    return df

# remove undesired signs and characters
def clean_text(sen):
    # remove html tags
    re.sub(r'<[^>]+>', ' ', sen)
    # remove punctuations and numbers
    sen = re.sub('[^a-zA-Z]', ' ', sen)
    # single character removal
    sen = re.sub(r'\s+[a-zA-Z]\s+', ' ', sen)
    # remove multiple spaces
    sen = re.sub(r'\s+', ' ', sen)
    return sen

# change capital letters to lower ones
def lower_token(token):
    lower_token = []
    for word in token:
        lower_token.append(word.lower())
    return lower_token

# remove stopwords
def remove_stopwords(token):
    stoplist = stopwords.words('english')
    filtered_token = []
    for word in token:
        if word not in stoplist:
            filtered_token.append(word)
    return filtered_token

def overall_clean_text(column_name, df):
    cleaned_column = column_name + '_clean'
    final_column = column_name + '_final'

    df[cleaned_column] = df[column_name].apply(lambda sen: clean_text(sen))
    # transfer sentences to tokens
    tokens = [word_tokenize(sen) for sen in df[cleaned_column]]
    # change tokens to lower letters
    lower_tokens = [lower_token(token) for token in tokens]
    # remove stopwords
    filtered_tokens = [remove_stopwords(token) for token in lower_tokens]
    # combine results
    result = [' '.join(token) for token in filtered_tokens]
    df[final_column] = result
    df['tokens'] = filtered_tokens
    return df

# get x and y data for training and testing
def get_x_y(t_x, t_data, maxlen):
    t_sequences = tokenizer.texts_to_sequences(t_x.tolist())
    x = pad_sequences(t_sequences, maxlen=maxlen)
    y = t_data['label'].values
    return x, y

# function for embeddings
def embeddings(tokens_list, word2vec, dim=300):
    if len(tokens_list)<1:
        return np.zeros(dim)
    vec = [word2vec[word] if word in word2vec else np.random.rand(dim) for word in tokens_list]
    average = len(vec) / np.sum(vec, axis=0)
    return average

# CNN architechture
def cnn_construct(vocab_size, dim, embedding_matrix, maxlen):
    input_data = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size,
                                dim,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)
    embedding = embedding_layer(input_data)
    conv1 = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding)
    pool1 = MaxPooling1D(2)(conv1)
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(pool1)
    pool2 = GlobalMaxPooling1D()(conv2)
    dropout = Dropout(0.5)(pool2)
    dense1 = Dense(64, activation='relu')(dropout)
    pred = Dense(1, activation='sigmoid')(dense1)

    model = Model(input_data, pred)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model

# transfer texts to lists of sentences
def txt_to_sents(txts):
    text_sents = []
    text_sents_num = []
    for num in range(len(txts)):
        filename = os.path.join(PATH, 'Data\SOTU texts', txts[num])
        text_file = open(filename, 'r', encoding='UTF-8-sig', errors='ignore')
        text = text_file.read()
        text = text.replace('\n', '')
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sents = list(doc.sents)
        text_sents.append(sents)
        text_sents_num.append(len(sents))
    return text_sents, text_sents_num

# transfer text sentences into list of DataFrames
def df_list(num_txts, text_sents):
    df_text = []
    for num in range(num_txts):
        text_sents[num] = [sen.text for sen in text_sents[num]]
        df = pd.DataFrame(text_sents[num])
        df.columns = ['address']
        df_text.append(df)
    return df_text

# prepare address texts for CNN prediction
def prep_address(df, maxlen):
    df = overall_clean_text('address', df)
    address = df['address_final']
    address_sequences = tokenizer.texts_to_sequences(address.tolist())
    address_x = pad_sequences(address_sequences, maxlen=maxlen)
    return address_x

# get predictions through trained CNN
def get_predictions(address_x, batch_size):
    # predict sentiments
    predictions = model.predict(address_x, batch_size=batch_size, verbose=1)
    prediction_labels = []
    for p in predictions:
        if p[0] < 0.5:
            prediction_labels.append(0)
        else:
            prediction_labels.append(1)
    return prediction_labels

# read data from excel file
def read_excel(fname, sname, header):
    df = pd.read_excel(os.path.join(PATH, 'Data', fname), header=header, sheet_name=sname)
    return df

# Prepare Dataset for Training

# If you fail to download the file from kaggle, you can directly use this one.
movie_reviews = read_csv("IMDB Dataset.csv")
# label the reviews by 0 and 1
movie_reviews = label_sentiment(movie_reviews)

# clean the texts
movie_reviews = overall_clean_text('review', movie_reviews)
movie_reviews = movie_reviews[['review_final', 'tokens', 'label']]

# Building Up CNN Model

# split train and test data
training_data, testing_data = train_test_split(movie_reviews, test_size=0.2)
training_x = training_data['review_final']
testing_x = testing_data['review_final']

"""
NOTICE: The following line may take a bit long since the vector file is large.
If the storage is not enough for this download, you might directly use the result in sentiment.xlsx data.
"""
word2vec = gensim.downloader.load('word2vec-google-news-300')

training_embeddings = list(training_data['tokens'].apply(lambda x: embeddings(x, word2vec)))

# set variables
maxlen = 100
dim = 300

training_words = [word for tokens in training_data["tokens"] for word in tokens]
training_wordlist = sorted(list(set(training_words)))

# tokenize and padding
tokenizer = Tokenizer(num_words=len(training_wordlist), lower=True, char_level=False)
tokenizer.fit_on_texts(training_x.tolist())
train_x, train_y = get_x_y(training_x, training_data, maxlen)
test_x, test_y = get_x_y(testing_x, testing_data, maxlen)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# get embedding matrix
embedding_matrix = np.zeros((vocab_size, dim))
for word, index in word_index.items():
    if word in word2vec:
        embedding_matrix[index, :] = word2vec[word]
    else:
        np.random.rand(dim)

# construct CNN model
model = cnn_construct(vocab_size, dim, embedding_matrix, maxlen)
callbacks_list = [EarlyStopping(monitor='var_loss', min_delta=0.01, patience=4, verbose=1)]

# train model
epochs = 16
batch_size = 64
hist = model.fit(train_x, train_y, epochs=epochs, verbose=1, validation_split=0.15, shuffle=True, batch_size=batch_size)

# test model
evaluation = model.evaluate(test_x, test_y, verbose=1)
accuracy = evaluation[1]
print("Accuracy is: ", accuracy)

# Predict on the SOTU Texts

# texts of sotu address
txts = os.listdir(PATH + '\Data\SOTU texts')
num_txts = len(txts)

# break the whole text into sentences
text_sents, text_sents_num = txt_to_sents(txts)
# transfer each text to a DataFrame and get a list of DataFrames
df_text = df_list(num_txts, text_sents)

scores = []
for df in df_text:
    # prepare address texts
    address_x = prep_address(df, maxlen)
    # get predictions
    prediction_labels = get_predictions(address_x, 1024)
    # get a sentiment score of the whole address
    score = sum(prediction_labels) / len(prediction_labels)
    scores.append(score)

# uncomment the next line to see scores
# scores

# Save to Excel File

# get the dates for sotu and save to excel
sotu_date = read_excel('Data_Daily_1.1-2.4.xlsx', 'address', [0])
sotu_date['sentiment'] = scores
sotu_date = sotu_date[['Date', 'sentiment']]
sotu_date.to_excel(os.path.join(PATH, 'Data', 'sentiment.xlsx'))