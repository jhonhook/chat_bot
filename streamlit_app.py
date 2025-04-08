import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model
import streamlit as st

nltk.download('punkt')

# Load necessary resources
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents1.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) 
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if len(ints) == 0:
        return "Sorry, I didn't understand that."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, something went wrong."

# Streamlit UI setup
st.set_page_config(page_title="Financial Assistant Chatbot", page_icon=":speech_balloon:")
st.title(":speech_balloon: Financial Assistant Chatbot")
st.markdown("Chat with the assistant by typing below.")

# Chat state initialization
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

# Chat input
user_input = st.text_input("You:", "", key="input")

if user_input:
    st.session_state.chat_log.append(("You", user_input))
    ints = predict_class(user_input, model)
    response = get_response(ints, intents)
    st.session_state.chat_log.append(("Bot", response))

# Display chat log
for sender, message in st.session_state.chat_log:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
