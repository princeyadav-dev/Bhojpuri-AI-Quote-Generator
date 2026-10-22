
import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Bhojpuri AI Quote Generator",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(
        "bhojpuri_quote_model.h5",
        compile=False
    )
    with open("bhojpuri_tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
max_sequence_len = model.input_shape[1] + 1

def generate_quote(seed_text, next_words=12, temperature=0.8):
    result = seed_text.strip()

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.asarray(predictions, dtype=np.float64)

        # Temperature sampling
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions - np.max(predictions))
        probabilities = predictions / np.sum(predictions)

        next_token = np.random.choice(
            len(probabilities),
            p=probabilities
        )

        output_word = next(
            (word for word, index in tokenizer.word_index.items()
             if index == next_token),
            ""
        )

        if not output_word or output_word == "<OOV>":
            continue

        result += " " + output_word

    return result.strip().capitalize() + "."

st.title("💬 Bhojpuri AI Quote Generator")
st.write("Apna topic likhi aur Bhojpuri quote generate kari!")

topic = st.text_input(
    "Quote ke topic likhi",
    placeholder="Jaise: Jindagi, Mehnat, Dosti"
)

temperature = st.slider(
    "Creativity",
    min_value=0.3,
    max_value=1.2,
    value=0.8,
    step=0.1
)

if st.button("✨ Generate Quote", type="primary"):
    if not topic.strip():
        st.warning("Pahile koi topic likhi!")
    else:
        with st.spinner("Quote banawat bani..."):
            quote = generate_quote(
                topic,
                next_words=12,
                temperature=temperature
            )

        st.success("Tohar Bhojpuri quote taiyar ba!")
        st.markdown(f"> {quote}")

        st.caption(
            "Note: Quote model ke prediction se generate bhail ba; "
            "Bhojpuri grammar aur quality har baar sahi ho, zaroori na ba."
        )
