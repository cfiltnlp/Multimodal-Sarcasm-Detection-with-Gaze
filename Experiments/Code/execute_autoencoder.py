import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time
import sys
import random
import pickle


from numpy import genfromtxt
my_data = genfromtxt('/home/development/divyankt/DT_MTP/cikm/data/divyank/231samples_autoencoder.txt', delimiter=' ')


t = T.from_numpy(my_data)




# device = T.device("cuda:1")


# # Loading data
# path = "/home/development/divyankt/DT_MTP/cikm/data/"
# #path = "/home/development/apoorvan/AN_MTP/cikm/data/"
# mustard_input = pd.read_csv('/home/development/divyankt/DT_MTP/cikm/data/final_datasets/mustard++_sarcasm_detection_forgaze_with_context.csv', index_col=0)
# #temp = open(path+'extracted_features/an_merged/features_Tbart_Vkey_Audio_sarcasm.pickle', 'rb')
# temp = open(path+'divyank/features_VTAG.pickle', 'rb')
# data = pickle.load(temp)
# count=0
# for key in list(data.keys()):
#     count=count+1
#     for idx in ['cText', 'uText', 'cAudio', 'uAudio', 'cVideo', 'uVideo', 'gaze']:
#         data[key][idx] /= np.max(abs(data[key][idx]))



# Actual_value=[]
# Predicted_value=[]

# temp=[]
# remp=[]
# dataframe_x = pd.DataFrame(temp)
# # with open('/home/development/divyankt/DT_MTP/cikm/data/divyank/text_gaze_mapping.txt', 'w') as f:
# for i in range(1024):
#     temp=[]
#     for j,key in enumerate(list(data.keys())):
#             #f.write("%s\n" % data[key]['uText'][i])
#         # if((j>=171 and j<=230) or (j>=402 and j<=461) or (j>=633 and j<=692) or (j>=864 and j<=923) or (j>=1095 and j<=1154)):
#         #   continue
#         temp.append(data[key]['uText'][i])
#     dataframe_x[i]=temp    

# dataframe_x.to_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_train_autoencoder.txt', header=None, index=None, sep=' ', mode='a')




# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5), (0.5))
#     ])

# #transform = transforms.ToTensor()

# mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
#                                           batch_size=64,
#                                           shuffle=True)

# dataiter = iter(t)
# utext_torch= dataiter.next()
# print(torch.min(utext_torch), torch.max(utext_torch))


#repeatedly reduce the size
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512), # (N, 1024) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Tanh()
        )
        self.double()
        
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded,encoded
    
# Input [-1, +1] -> use nn.Tanh


model = Autoencoder()

criterion = nn.MSELoss()
optimizer = T.optim.Adam(model.parameters(),
                             lr=0.0005, 
                             weight_decay=1e-5)



# Point to training loop video
num_epochs = 10
outputs = []




for epoch in range(num_epochs):
    outputs=[]
    for img in t:

        # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
        recon,encod = model(img) 
        loss = criterion(recon, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.append(encod)

        


        


    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    
    #outputs.append((epoch, img, recon, encod))

with open(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/final_8_text_features.txt', 'w') as fp:
    for item in outputs:
            # write each item on a new line
        fp.write("%s\n" % item)

    print('Done')


            



   
    










