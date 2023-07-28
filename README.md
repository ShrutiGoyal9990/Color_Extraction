# Color_Extraction
This project is based on dominant color extraction which can be used in the camouflage design, image segmentation, color editing, palette generation  and image compression.


The user provides uploads an image either through camera or any donloaded file and the number of colors that they want to use in generation of the same image. The image is generated by using the dominant colors used in the original image. Dominant colors are extracted using clustering or histogram-based methods. The color palette is displayed and this modified image can be downloaded as well.


K-means algorithm is used that calculates the color features such as the saturation, contrast,and area of each cluster, based on which it extracts the dominant colors. It is a clustering algorithm that is more efficient than the conventional clustering method. It is an unsupervised learning model in which there is no training data but the model analyze and cluster unlabeled datasets by identifying patterns.


Streamlit is used for model deployment so as to provide an interactive user interface.
