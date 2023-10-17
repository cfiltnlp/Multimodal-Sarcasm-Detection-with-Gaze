
#%%
import torch
import os
import time
import sys
import torch.nn as nn
import pretrainedmodels.utils as utils
import pretrainedmodels
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
# import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#%%
path = "/home/development/apoorvan/AN_MTP/cikm/data/final_datasets/"
ukeyframes_path = 'selected_utterance_frames'
ckeyframes_path = 'selected_context_frames'


# '''There are a few videos which were too small to extract key frames
# So we extract all their frames using the ffmpeg'''

# additional_utterance_frames = ['2_55_u',
#   '1_S11E11_217_u',
#   '3_S01E06_054_u',
#   '1_S12E03_174_u',
#   '2_463_u',
#   '2_609_u',
#   '1_S12E03_095_u',
#   '1_S11E11_403_u',
#   '2_569_u',
#   '3_S05E05_119_u',
#   '1_S12E14_125_u',
#   '2_385_u',
#   '1_S10E17_287_u',
#   '1_S10E07_198_u',
#   '1_S10E02_123_u',
#   '2_474_u',
#   '2_504_u',
#   '2_602_u',
#   '1_S12E03_118_u',
# ]

# additional_context_frames = ['1_S11E05_402_c']

# print(len(additional_utterance_frames))
# print(len(additional_context_frames))


# for x in additional_utterance_frames:
#     os.system('ffmpeg -i '+path+'final_utterance_videos/'+x+' '+path+ukeyframes_path+'/'+x+'_%05d.jpg')

# for x in additional_context_frames:
#     os.system('ffmpeg -i '+path+'final_context_videos/'+x+' '+path+ckeyframes_path+'/'+x+'_%05d.jpg')
    

'''With all the key frames obtained from katna and some frames from ffmpeg, let's 
create a dictionary with each video_name holding 
its corresponding frames'''


ufilelist = os.listdir(path+ukeyframes_path)
cfilelist = os.listdir(path+ckeyframes_path)

dictionary = {}  
print(len(ufilelist)-1)
print(len(cfilelist))

for x in ufilelist: 
    if(x[0]=='.'):
        continue
    keys = os.path.basename(x).split('_')[:-1]
    key = '_'.join(keys)
    group = dictionary.get(key,[])
    group.append(x)  
    dictionary[key] = group

print(len(dictionary.keys()))

for x in cfilelist:  
    if(x[0]=='.'):
        continue
    keys = os.path.basename(x).split('_')[:-1]
    key = '_'.join(keys)
    group = dictionary.get(key,[])
    group.append(x)  
    dictionary[key] = group

print(len(dictionary.keys()))


'''Use resnet152 to now obtain the image features in a particular video
and perform mean'''

model_name = 'resnet152'
device = torch.device('cuda:0')
print("Loading model...")
model = pretrainedmodels.__dict__[model_name](
    num_classes=1000, pretrained='imagenet').to(device)
print("Model loaded...")
rgb = {}
load_img = utils.LoadImage()
tf_img = utils.TransformImage(model)

df = pd.read_csv(path+'mustard++_all_cu.csv')
video_names = list(df['KEY'])
c=0
for key in video_names:
    # print(key)
    feature = []
    for image in tqdm(dictionary[key]):
        if(key[-1]=='c'):
            path_img = path+ckeyframes_path+'/'+image
        elif(key[-1]=='u'):
            path_img = path+ukeyframes_path+'/'+image
        input_img = load_img(path_img)
        input_tensor = tf_img(input_img)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input = torch.autograd.Variable(input_tensor, requires_grad=False)
        output_features = model.features(input).mean(2).mean(2).squeeze()
        feature.append(output_features.detach().cpu().numpy())
        del output_features
        del input_tensor
    feature = torch.tensor(feature)
    rgb[key] = feature.mean(0).numpy()
    sys.stdout.flush()

    

pickle_out = open('/home/development/apoorvan/AN_MTP/cikm/data/extracted_features/'+'key_video_features.pickle', "wb")

pickle.dump(rgb, pickle_out)
pickle_out.close()
