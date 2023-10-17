import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
import csv
from gensim.models import KeyedVectors

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('/home/development/divyankt/DT_MTP/cikm/data/Trial Reports/Final feature reports/train-test/train_11.csv') #change

# Extract the desired column as a Series
column = df['sentence']
lab = df['IA_REGRESSION_OUT_FULL_COUNT'] #change
# Convert the Series to a list
sent = column.tolist()
sent_lab = lab.tolist()
# print(type(column_list[0]))


# Generate sample data
# texts = ['i love being ignored', 'i love working at 4am in the morning','its a long tiring weekend','its good to walk every day']
embeddings = model.encode(sent)
X = np.zeros((len(sent), 384))
for i, embedding in enumerate(embeddings):
    X[i] = embedding
    # print(embedding)
    

# Normalize the word embeddings
X = X / np.linalg.norm(X, axis=1, keepdims=True)

y = np.array(sent_lab)

# Fit regression model
regr = SVR(kernel='linear', C=1e3)
regr.fit(X, y)


# Predict
X_test = np.zeros((12, 384))
# For creating key-val dict of key and sentences of mustard++

# with open('/home/development/divyankt/DT_MTP/cikm/data/mustard++_complete.csv', 'r') as file:
#     reader = csv.reader(file)
#     header = next(reader)
#     data = {}
#     for row in reader:
#         key = row[2]
#         value = row[3]
#         data[key] = value

# df = pd.read_csv('/home/development/divyankt/DT_MTP/cikm/data/Predicted_Gaze_with_regression_features - Sheet1.csv', header=None)

# # Extract the desired column as a Series
# scenes = df[0]

# sceneslst = scenes.tolist()

test_text = ["You're gonna have to tell me how you did that.","Cause all I ever get to do now is pregnant stuff, it just bums me out.","I've never loved anybody as much as I love you.","Umm, excuse me, we switched apartments. You can't eat are food anymore, that-that gravy train had ended.","I didn't know you and Carol were getting divorced, I'm sorry.","Alright, I'm gonna go pick up a few things for the trip.","Hey. Sorry if I scared you. I know I have somewhat ghost-like features.","Richard, look, I'm sorry, but I think the box is here to stay.","This plan better fucking work with a sacrifice like this.","He was oddly tall, don't you think?","He's our most profitable developer by far.","Look, we cannot take blood money."]

# for k in sceneslst:
#     test_text.append(data[k+'_u'])
# test_text = ["Hey. Sorry if I scared you. I know I have somewhat ghost-like features.","I've never loved anybody as much as I love you.","This plan better fucking work with a sacrifice like this."]
test_embeddings = model.encode(test_text)
for i, t_embedding in enumerate(test_embeddings):
    X_test[i] = t_embedding
    # print(t_embedding)

# Normalize the word embeddings
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

y_pred = regr.predict(X_test)

print(y_pred)
print(len(y_pred))

# np.savetxt("/home/development/divyankt/DT_MTP/cikm/data/array.csv", y_pred.reshape(-1, 1), delimiter=",", fmt="%.2f")
# print(X_test)




##########################################################


# ["You're gonna have to tell me how you did that."
# ,"Cause all I ever get to do now is pregnant stuff, it just bums me out."
# ,"I've never loved anybody as much as I love you."
# ,"Umm, excuse me, we switched apartments. You can't eat are food anymore, that-that gravy train had ended."
# ,"I didn't know you and Carol were getting divorced, I'm sorry."
# ,"Alright, I'm gonna go pick up a few things for the trip."
# ,"Hey. Sorry if I scared you. I know I have somewhat ghost-like features."
# ,"Richard, look, I'm sorry, but I think the box is here to stay."
# ,"This plan better fucking work with a sacrifice like this."
# ,"He was oddly tall, don't you think?"
# ,"He's our most profitable developer by far."
# ,"Look, we cannot take blood money."]
