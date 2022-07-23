import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import seaborn as sns
import gzip
import keras
import tensorflow as tf
from PIL import Image


# WHAT_SHOULD_I_DO?
# 1. MAKE BEST MODEL.
# 2. TAKE IMAGE, AND FIT MODEL.
# 3. TAH-DAH!
# 4. RESULT : 80% NON-MASK -> YOU SHOULD WEAR MASK, 15% N95 -> KF94, 5% SURGERY

# Use page wide.
# st.set_page_config(layout="wide")

# For EDA

st.set_page_config(
    page_title="Face Mask Types | Team_PureSoul",
    page_icon="😷",
)


st.title("Face Mask Types")
st.header("LIKELION AI SCHOOL-6th, ♡영혼이맑아영♡")
st.markdown('''
The Dataset includes more than 2000 images with 300x300 resolution, which are totally clean.
Perfect for multiclass classification.

## Mask Type
- __cloth__ : Cloth face mask images
- __n95__ : N95 face mask images
- __n95v__ : N95 with Valve face mask images
- __nfm__ : No Face Mask images
- __srg__ : Surgical face mask images
''')

new_file = st.file_uploader('Upload Your Masked or Non-Masked Image')

best_model = keras.models.load_model('best_model_wholeB0.h5')


img_list = ['cloth', 'n95', 'n95v', 'nfm', 'srg']

label_maps = {}
label_maps_rev = {}

for i, v in enumerate(img_list):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})

if new_file is not None:
    img = Image.open(new_file)
    img = img.convert('RGB')
    img = img.resize((300, 300))
    print(img.size)
    # show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')

    probs = best_model.predict(np.expand_dims(img, axis=0))
    st.image(img, use_column_width=False)
    # print(probs)
    for idx in probs.argsort()[0][::-1][:3]:
        st.text("{:.2f}%".format(probs[0][idx]*100) +" "+ label_maps_rev[idx].split("-")[-1])

st.markdown('##')
