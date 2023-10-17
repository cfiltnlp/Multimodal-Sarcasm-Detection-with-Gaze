import numpy as np
import pandas as pd
import time
import sys
import random
import pickle


path = "/home/development/divyankt/DT_MTP/cikm/data/"
#path = "/home/development/apoorvan/AN_MTP/cikm/data/"
#mustard_input = pd.read_csv('/home/development/divyankt/DT_MTP/cikm/data/final_datasets/mustard++_sarcasm_detection_forgaze_with_context.csv', index_col=0)
temp = open(path+'extracted_features/an_merged/features_Tbart_Vkey_Audio_sarcasm.pickle', 'rb')

temp2 = open(path+'divyank/features_VTAG_231.pickle', 'rb')
# temp3 = open(path+'divyank/features_VTAG_copy1.pickle', 'rb')
data = pickle.load(temp)
data2= pickle.load(temp2)
# data3= pickle.load(temp3)








# with open('/home/development/divyankt/DT_MTP/cikm/data/divyank/Mustard-pp-keys-with-gaze.txt', 'w') as f:
#     for item in data2:
#         f.write("%s\n" % data2[item]['uText'])


# with open('/home/development/divyankt/DT_MTP/cikm/data/divyank/231_scenes.txt', 'w') as f:
#     for item in data2:
#         f.write("%s\n" % item)


df=pd.read_csv(r'/home/development/divyankt/DT_MTP/cikm/data/Gaze_pred_svm- Sheet1.csv',header=None)

        
# print(data2['1_105_1']['gaze'])
datasets={}

for key in list(data2.keys()):
    # print(key)
    track1={}
    track1['uText']=data2[key]['uText']
    track1['cText']=data2[key]['cText']
    track1['uAudio']=data2[key]['uAudio']
    track1['cAudio']=data2[key]['cAudio']
    track1['uVideo']=data2[key]['uVideo']
    track1['cVideo']=data2[key]['cVideo']
    track1['gaze']=data2[key]['gaze']
    
    datasets[key]=track1
    

print(len(datasets))
l=[]
for key in list(data2.keys()):
    l.append(key[:-2])

gaze_list=[*set(l)] # 231 unique SCENE names......
print(len(gaze_list))
for key in list(data.keys()):

    if(key not in gaze_list):
        
        track={}
        track['uText']=data[key]['uText']
        track['cText']=data[key]['cText']
        track['uAudio']=data[key]['uAudio']
        track['cAudio']=data[key]['cAudio']
        track['uVideo']=data[key]['uVideo']
        track['cVideo']=data[key]['cVideo']
        datasets[key]= track

    
print(len(datasets))

#   #Adding gaze features of 971 samples to features_VTAG.pickle
for index, row in df.iterrows():
    gaze_features=[]
    for i in range(25):
        gaze_features.append(row[i+1])
        datasets[row[0]]['gaze']=np.array(gaze_features)




# print(datasets['1_105'])

# print("============================")

# print(datasets['1_10495'])


# print(datasets['2_288'])
# print(datasets['3_S06E06_143'])
print(datasets['3_S06E07_272'])

print(len(datasets))


with open('/home/development/divyankt/DT_MTP/cikm/data/divyank/finalmpp.txt', 'w') as f:
    for key in datasets:

        f.write(str(key)+'\n')

with open('/home/development/divyankt/DT_MTP/cikm/data/divyank/features_VTAG.pickle', 'wb') as f:
    pickle.dump(datasets, f)



