import streamlit as st
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from PIL import Image

# ---------- Page Config ----------
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")

# ---------- Load Caption Model ----------
@st.cache_resource
def load_caption_model():
    return load_model('/home/wicky/workspace/second/model4.keras')  # Use correct model path

model = load_caption_model()

# ---------- Load Tokenizer ----------
@st.cache_resource
def load_tokenizer():
    with open('/home/wicky/workspace/second/tokenizer1 (3).pickle', 'rb') as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

# ---------- Set Max Caption Length ----------
max_length = 35  # Set as per your training

# ---------- Load VGG16 (CNN feature extractor) ----------
@st.cache_resource
def load_cnn_model():
    base_model = VGG16(weights='imagenet')
    model_new = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model_new

cnn_model = load_cnn_model()

# ---------- Extract Features ----------
def extract_features(image, model):
    img = image.resize((224, 224))  # VGG16 expects 224x224
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img, verbose=0)
    return features

# ---------- Beam Search Captioning ----------
def beam_search_predictions(model, tokenizer, photo, max_length, beam_index=3):
    start = [tokenizer.word_index['start']]
    sequences = [[start, 0.0]]  # (sequence, score)

    while len(sequences[0][0]) < max_length:
        temp = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([photo, padded], verbose=0)
            top_words = np.argsort(yhat[0])[-beam_index:]

            for word_id in top_words:
                new_seq = seq + [word_id]
                new_score = score + np.log(yhat[0][word_id] + 1e-10)
                temp.append([new_seq, new_score])

        sequences = sorted(temp, key=lambda tup: tup[1], reverse=True)[:beam_index]

        # Stop early if all sequences end with 'endseq'
        if all(tokenizer.index_word.get(seq[-1], '') == 'end' for seq, _ in sequences):
            break

    final_seq = sequences[0][0]
    
    # Convert to caption by removing startseq and cutting off at endseq
    result = []
    for idx in final_seq:
        word = tokenizer.index_word.get(idx, '')
        if word == 'end':
            break
        if word != 'start':
            result.append(word)
    
    return ' '.join(result)

# ---------- Streamlit UI ----------
st.title("🖼️ Image Caption Generator (VGG16 + RNN)")
st.markdown("Upload an image, and the model will describe it for you.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner('🔍 Generating caption...'):
        features = extract_features(image, cnn_model)
        caption = beam_search_predictions(model, tokenizer, features, max_length)

    st.markdown("### 📝 Generated Caption:")
    st.success(caption)
