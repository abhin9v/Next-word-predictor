import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('next_word_predictor.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_next_word(model,tokenizer,text,max_seq_len):
    tokenlist=tokenizer.texts_to_sequences([text])[0]
    if len(tokenlist)>=max_seq_len:
        tokenlist=tokenlist[-(max_seq_len-1):]
    tokenlist=pad_sequences([tokenlist],maxlen=max_seq_len-1,padding='pre')
    predicted=model.predict(tokenlist,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

st.title('Next Word Predictor')
text=st.text_input('Enter text:','I am')
if st.button('Predict Next Word'):
    max_seq_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,text,max_seq_len)
    st.write(f'Next word: {next_word}')