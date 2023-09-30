#!/usr/bin/env python
# coding: utf-8

import numpy as np

from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop



#Loading the dataset
text=open("dataset.txt","r",encoding= "utf8")

# Store the data in a list
lines=[]
for i in text:
    lines.append(i)
    
# Preapring a raw data for the list
data=""
for i in lines:
    data= " ".join(lines)

#Replacing unwanted symbols with spaces
data=data.replace('\n','').replace('\t', '').replace('??', '').replace('“','').replace('”','')

#Removing unnecessary spaces in the data
data=data.split()
data= ' '.join(data)

# Just for checking,taking up smaaler quantity of data
#test=data[:10000]

    
#Applying Tokenization
tokenizer= RegexpTokenizer(r"\w+")  #Object of tokenizer
tokens=tokenizer.tokenize(data.lower()) #Passing the data

#Dictionary to match tokens with an index
unique_tokens= np.unique(tokens)
unique_token_index={token:idx for idx,token in enumerate(unique_tokens)}

#Looking at the last 5 words to predict the next word
n_words=5
ip_words= []
next_words= []

for i in range(len(tokens) - n_words):
    ip_words.append(tokens[i:i+n_words])
    next_words.append(tokens[i+n_words])

x= np.zeros((len(ip_words),n_words,len(unique_tokens)),dtype=bool)
y=np.zeros((len(next_words),len(unique_tokens)),dtype=bool)

for i,words in enumerate(ip_words):
    for j,word in enumerate(words):
        x[i,j,unique_token_index[word]]=1
    y[i,unique_token_index[next_words[i]]]=1



#Training the model with the dataset
model= Sequential()
model.add(LSTM(128,input_shape=(n_words,len(unique_tokens)),return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",optimizer=RMSprop(learning_rate=0.01),metrics=["accuracy"])
model.fit(x,y,batch_size=125,epochs=30,shuffle=True)



#Saving the model created
model.save("model.h5")
model=load_model("model.h5")



#Prediction
def predict_next_word(input_text,n_best):
    input_text=input_text.lower()
    x=np.zeros((1,n_words,len(unique_tokens)))
    for i,word in enumerate(input_text.split()):
        x[0,i,unique_token_index[word]]=1
    predictions = model.predict(x)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

#Trail Input
possible=predict_next_word("They must be able to",5) #5 possible next words
print([unique_tokens[idx] for idx in possible])



model=load_model("model.h5")

