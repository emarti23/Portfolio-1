import pandas as pd                    
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from string import punctuation
from langdetect import detect #detect english and non-english languages
import matplotlib.pyplot as plt
from wordcloud import WordCloud


dataset = pd.read_csv('../final_dataset.csv')

dataset['published'] = pd.to_datetime(dataset['published'])
dataset['collection date'] = pd.to_datetime(dataset['collection date'])
dataset['days']= (dataset['collection date'] - dataset['published']).dt.days.astype('int16')
dataset['tags'] = dataset['tags'].fillna('')

dataset= dataset.drop(columns=['ID', 'published', 'collection date'])

print(dataset.info())
print(dataset[['tags', 'title']].head())

######## text pre-processing 

#convert to lower case
dataset['title'] = dataset['title'].str.lower()
dataset['tags'] = dataset['tags'].str.lower()
dataset['title'] = dataset['title'].str.replace('[{}]'.format(punctuation), ' ', regex = True)
dataset['tags'] = dataset['tags'].str.replace('[{}]'.format(punctuation), ' ', regex = True) 

#remove non-ascii characters
dataset['title']  = dataset['title'] .str.encode('ascii', 'ignore').str.decode('ascii')
dataset = dataset[dataset['title'] != '']


dataset['tags']  = dataset['tags'] .str.encode('ascii', 'ignore').str.decode('ascii') 
dataset = dataset[dataset['tags'] != '']

#remove numbers
dataset['title'] = dataset['title'].str.replace('\d+', '', regex = True)
dataset['tags'] = dataset['tags'].str.replace('\d+', '', regex = True)
print(dataset[['tags', 'title']])

#detect title language and drop all non-english languages
def language_detection(text):
   try:
       return detect(text)
   except:
       return 'unknown'

dataset['language'] = dataset['title'].apply(language_detection)


dataset = dataset[dataset['language'] == 'en']
dataset= dataset.drop(columns=['language'])

print(dataset.head())
print(dataset.info())

#remove stopwords
stop = text.ENGLISH_STOP_WORDS
dataset['title'] = dataset['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
dataset['tags'] = dataset['tags'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) #i dont need to do this in tags


#tokenize title words
t = keras.preprocessing.text.Tokenizer(split = ' ')
t.fit_on_texts(dataset['title'])
sequence = t.texts_to_sequences(dataset['title'])
title_padded_train = keras.utils.pad_sequences(sequence, padding='post', maxlen =10 )
word_index_title = t.word_index
print('Title vocab size:', len(word_index_title))
print(title_padded_train)

print(len(title_padded_train))
print(len(title_padded_train[0]))

#tokenize tags 
t2 = keras.preprocessing.text.Tokenizer(split = ' ')
t2.fit_on_texts(dataset['tags'], )
sequence = t2.texts_to_sequences(dataset['tags'])
tags_padded_train = keras.utils.pad_sequences(sequence, padding='post', maxlen =10 )
word_index_tags = t2.word_index
print('Tags vocab size:', len(word_index_tags))
print(tags_padded_train)

print(len(tags_padded_train))
print(len(tags_padded_train[0]))


#####word cloud

wordcloud_title = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(' '.join(t.word_index))

# plot the WordCloud image for title                    
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_title)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


wordcloud_tags = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(' '.join(t2.word_index))
 
# plot the WordCloud image for tags                    
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_tags)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


#####train/test split of nmercal data and save to csv

training_data, testing_data = train_test_split(dataset, test_size=0.3, random_state=30)
validation_data, testing_data = train_test_split(testing_data, test_size=0.5, random_state=30)

training_title, testing_title = train_test_split(title_padded_train, test_size=0.3, random_state=30)
validation_title, testing_title = train_test_split(testing_title, test_size=0.5, random_state=30)
training_tags, testing_tags = train_test_split(tags_padded_train, test_size=0.3, random_state=30)
validation_tags, testing_tags = train_test_split(testing_tags, test_size=0.5, random_state=30)

training_data.to_csv(r'../training data .csv', index=False)
validation_data.to_csv(r'../validation data .csv', index=False)
testing_data.to_csv(r'../testing data .csv', index=False)

scaler = preprocessing.StandardScaler()
scaler1 = preprocessing.StandardScaler()
scaler2 = preprocessing.StandardScaler()

#scale padded sequences
scaler1.fit(training_title)
training_title = scaler1.transform(training_title)
valid_title = scaler1.transform(validation_title)
test_title = scaler1.transform(testing_title)

training_tags = scaler2.fit_transform(training_tags)
valid_tags = scaler2.transform(validation_tags)
test_tags = scaler2.transform(testing_tags)

print(training_title.shape)
print(f"No. of numerical training examples: {training_data.shape[0]}")
print(f"No. of numerical testing examples: {testing_data.shape[0]}")
print(f"No. of titles examples: {training_title.shape[0]}")
print(f"No. of titles examples: {testing_title.shape[0]}")
print(f"No. of tags examples: {training_tags.shape[0]}")
print(f"No. of tags examples: {testing_tags.shape[0]}")

print(training_data.head())
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)



####scaling numerical data
train_data = training_data[['title_length', 'duration','tag_count','days','like_count','comment_count', 'view_count']]
scaler.fit(train_data)
scale_train = scaler.transform(train_data)
names = train_data.columns
train_data_norm = pd.DataFrame(scale_train,columns= names)
print(train_data_norm.head())

valid_data= validation_data[['title_length','duration','tag_count','days','like_count','comment_count', 'view_count']]
scale_val = scaler.transform(valid_data)
valid_data_norm = pd.DataFrame(scale_val,columns= names)
print(valid_data_norm.head())

test_data = testing_data[['title_length','duration','tag_count','days','like_count','comment_count', 'view_count']]
scale_test = scaler.transform(test_data)
test_data_norm = pd.DataFrame(scale_test,columns= names)
print(test_data_norm.head())


########X,Y train/test split for numerical data

X_train = train_data_norm[['title_length','duration','tag_count','days','like_count','comment_count']]
Y_train = train_data_norm['view_count']

X_val = valid_data_norm[['title_length','duration','tag_count','days','like_count','comment_count']]
Y_val = valid_data_norm['view_count']

X_test = test_data_norm[['title_length','duration','tag_count','days','like_count','comment_count']]
Y_test = test_data_norm['view_count']

print(Y_test.iloc[:1])
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#######multiple input layers network (keras Functional API) 

#define inputs
title_input = layers.Input((10,))
#tag_input = layers.Input((10,))
numerical_input = layers.Input((6,))

####model

bias = keras.initializers.HeUniform()

concat_layer = layers.Concatenate(axis=1)([title_input, numerical_input])
x = layers.Dense(12, kernel_initializer='he_uniform', activation = 'relu', use_bias=True, bias_initializer=bias)(concat_layer)
x = layers.Dense(8, kernel_initializer='he_uniform',  activation = 'relu', use_bias=True, bias_initializer=bias)(x)
x = layers.Dense(6, kernel_initializer='he_uniform',  activation = 'relu', use_bias=True, bias_initializer=bias)(x)
output = layers.Dense(1,  kernel_initializer='he_uniform', activation = 'linear')(x)

model = keras.Model(inputs=[title_input, numerical_input], outputs = output)

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

adamOpt = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=adamOpt,
              loss='mean_squared_error', metrics=['cosine_proximity'])

training = model.fit([training_title, X_train], Y_train,
          batch_size=128, epochs=100, #callbacks=[callback], 
          validation_data=([valid_title,  X_val], Y_val))



print(len(training.history['loss']))


#evaluate model
evaluate = model.evaluate([test_title,  X_test], Y_test)
print("MSE loss on testing data:", evaluate)
print("Generate a prediction")
prediction= model.predict([test_title,  X_test])
prediction = pd.DataFrame(prediction, columns = ['view_count'])
print(prediction.shape)
prediction = pd.concat([X_test.reset_index(drop=True),prediction.reset_index(drop=True)], axis=1)
unscaled_df = scaler.inverse_transform(prediction)
unscaled_df = unscaled_df.astype(int)
unscaled_df = pd.DataFrame(unscaled_df, columns = prediction.columns)
print(prediction)
print(unscaled_df)
print("prediction shape:", prediction.shape)

print('Number of negative values for view count:', unscaled_df['view_count'].lt(0).sum().sum())

print(model.summary())


#############model performance visualisation

plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.plot(training.history['cosine_proximity'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val', 'cosine prox'], loc='upper right')
plt.show()



############save model

model.save('saved_model.h5')

