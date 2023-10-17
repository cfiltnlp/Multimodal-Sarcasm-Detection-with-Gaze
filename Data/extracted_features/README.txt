Name: Apoorva Nunna
Roll: 203050028
------------------------------------------------
This folder contains the features extracted and saved as pickle files. These pickle files are accessed in 
our python models and used in dictionary format
------------------------------------------------

text/audio/video_features.pickle - Pickle files or Features created in the previous iteration of the project using
BERT, Low-level and ResNet152 respectively

features_Tbart_Vkey_Audio.pickle - Combination of BART for text, old audio features and ResNet152 of key frames taken as video features

audio_features_extended.pickle - Audio features tried beyond the low-level prosodic

key_video_features.pickle - Contains ResNET152 embeddings extracted from the keyframes of videos

text_features_bart.pickle - Contains text features extracted using BART

aug-XYZ.pickles - Augmented data features