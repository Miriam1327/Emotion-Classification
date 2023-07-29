# Emotion Classification

This repository contains the data and code for the implementation of the emotion classification task from the CL team lab in summer 2023 at the University of Stuttgart by Miriam Segiet and Linnet Moxon.

The baseline consists of a Naive Bayes classifier in a bag of words approach with Laplace smoothing.<br>
The advanced models incorporate more information for emotion classification to investigate the impact.


In the folder [**data_representation**](EmotionClassification/data_representation) the preprocessing is performed. The data is read in and preprocessed into the information needed for further calculation. <br>
The baseline approach is mainly implemented in the file [**data_representation.py**](EmotionClassification/data_representation/data_representation.py).  <br>
The file [**emotion_info.py**](EmotionClassification/data_representation/emotion_info.py) processes further emotion-dependent information while the file [**demographic_info.py**](EmotionClassification/data_representation/demographi_info.py) processes the demographic information further. <br>

Our main naive bayes implementation is included in the folder [**main_work**](EmotionClassification/main_work).  <br>
Finally, the evaluation based on (macro) F1-scores is contained in the folder [**evaluation**](EmotionClassification/evaluation). 
