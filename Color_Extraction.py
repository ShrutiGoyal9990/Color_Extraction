import streamlit as st
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import webcolors
import numpy as np
import PIL
from PIL import Image as im
import matplotlib.patches as patches

html_temp = '<div style="background-color:pink;padding:10px";bold:True><h1 style="color:black;text-align:left;"> Dominant Color Extraction </h1></div>'
st.markdown(html_temp,unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<span style="color: green;font-size: 24px;">**HELLO**</span>', unsafe_allow_html=True)
    st.markdown('<span style="color: violet;font-size: 24px;">**Welcome to IMAGineColor**</span>', unsafe_allow_html=True)
    st.markdown('''<span style="color: blue;font-size: 24px;">**Let's play with the images!**</span>''', unsafe_allow_html=True)
    image = im.open("play.jpg")
    st.sidebar.image(image, use_column_width=True)

st.subheader('Input Image')

img = st.file_uploader('Choose an image')

if img is not None:
    st.header('Original Image')
    print(img)
    st.image(img)
    img = plt.imread(img)

    num_of_colors = st.slider("Select number of colors that you want to use:", min_value=1, max_value=100, step=1)
    
    n = img.shape[0]*img.shape[1] 
    all_pixels = img.reshape((n, 3))
    
    model = KMeans(n_clusters=num_of_colors)
    model.fit(all_pixels)
    centers = model.cluster_centers_.astype('uint8')
    
    new_img = np.zeros((n, 3), dtype='uint8')

    for i in range(n):
        group_idx = model.labels_[i] 
        new_img[i] = centers[group_idx]
    new_img = new_img.reshape(*img.shape)

    clicked = st.button('Generate')
    print(clicked)
    if clicked==True:
        st.header('Modified Image') 
        st.image(new_img)
        
        st.header("Summary of the colors used")
        
        def create_colored_rectangle(rgb_tuple, width, height):
            image = im.new("RGB", (width, height), rgb_tuple)
            return image

        print(centers)
        st.subheader("The dominant colors used :")
        for i in centers:
            print(i)
            st.write(f"The color for RGB({i[0]}, {i[1]}, {i[2]}) is:")
            width = 150
            height = 75
            colored_rectangle = create_colored_rectangle(tuple(i), width, height)
            st.image(colored_rectangle)

        new_img = im.fromarray(new_img)
        new_img.save("new_image.jpg")
        with open("new_image.jpg", "rb") as file:
            btn=st.download_button(
                label="Download Image",
                data=file,
                file_name="new_image.jpg",
                mime="image/jpg",
            )
