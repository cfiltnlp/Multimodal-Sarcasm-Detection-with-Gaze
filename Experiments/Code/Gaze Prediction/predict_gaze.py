import numpy as np
import pandas as pd
import time
import sys
import random
import pickle



import torch as T

device = T.device("cuda:1")


# Loading data
path = "/home/development/divyankt/DT_MTP/cikm/data/"
#path = "/home/development/apoorvan/AN_MTP/cikm/data/"
mustard_input = pd.read_csv('/home/development/divyankt/DT_MTP/cikm/data/final_datasets/mustard++_sarcasm_detection_forgaze_with_context.csv', index_col=0)
#temp = open(path+'extracted_features/an_merged/features_Tbart_Vkey_Audio_sarcasm.pickle', 'rb')
temp = open(path+'divyank/features_VTAG_231.pickle', 'rb')
data = pickle.load(temp)
count=0
# for key in list(data.keys()):
#     for idx in ['uText','uVideo']:
#         data[key][idx] /= np.max(abs(data[key][idx]))



path2 = "/home/development/divyankt/DT_MTP/cikm/data/"
#path = "/home/development/apoorvan/AN_# 201
# 166
# 235MTP/cikm/data/"
#mustard_input = pd.read_csv('/home/development/divyankt/DT_MTP/cikm/data/final_datasets/mustard++_sarcasm_detection_forgaze_with_context.csv', index_col=0)
temp2 = open(path+'extracted_features/an_merged/features_Tbart_Vkey_Audio_sarcasm.pickle', 'rb')
#temp2 = open(path+'divyank/features_VTAG_copy1.pickle', 'rb')
#temp = open(path+'divyank/features_VTAG.pickle', 'rb')
data2 = pickle.load(temp2)
count=0

temp=[]
remp=[]
dataframe_x=pd.DataFrame(temp)
dataframe_x2=pd.DataFrame(temp)
dataframe_x = pd.read_csv("/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_train.txt", sep=" ",header=None)
dataframe_x2 = pd.read_csv("/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_test.txt", sep=" ",header=None)
dataframe_x = dataframe_x.drop(dataframe_x.columns[[1024]], axis=1)
dataframe_x2 = dataframe_x2.drop(dataframe_x2.columns[[1024]], axis=1)
# for key in list(data2.keys()):
#     count=count+1
#     for idx in ['cText', 'uText', 'cAudio', 'uAudio', 'cVideo', 'uVideo']:
#         data2[key][idx] /= np.max(abs(data2[key][idx]))




Actual_value=[]
Predicted_value=[]
temp=[]

remp=[]
# dataframe_x=pd.DataFrame(temp)
# dataframe_x2=pd.DataFrame(temp)

# for i in range(1024):
#     temp=[]
#     for j,key in enumerate(list(data.keys())):
#             #f.write("%s\n" % data[key]['uText'][i])
#         if((j>=171 and j<=230) or (j>=402 and j<=461) or (j>=633 and j<=692) or (j>=864 and j<=923) or (j>=1095 and j<=1154)):
#           continue
#         temp.append(data[key]['uText'][i])
#     dataframe_x[i]=temp   

# for i in range(1024):
#     temp=[]
#     for j,key in enumerate(list(data.keys())):
#             #f.write("%s\n" % data[key]['uText'][i])
#         if((j>=171 and j<=230) or (j>=402 and j<=461) or (j>=633 and j<=692) or (j>=864 and j<=923) or (j>=1095 and j<=1154)):
#           continue
#         temp.append(data[key]['cText'][i])
#     dataframe_x[i+1024]=temp
 


        
for k,key in enumerate(list(data.keys())):

        if((k>=171 and k<=230) or (k>=402 and k<=461) or (k>=633 and k<=692) or (k>=864 and k<=923) or (k>=1095 and k<=1154)):
          continue
        remp.append(data[key]['gaze'][24])

dataframe_x[1024]=remp

# dataframe_x.to_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_train_t_withcontext.txt', header=None, index=None, sep=' ', mode='w')
dataframe_x.to_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_train.txt', header=None, index=None, sep=' ', mode='w') # for only text features

remp=[]

# for i in range(1024):
#     temp=[]
#     for j,key in enumerate(list(data.keys())):
#             #f.write("%s\n" % data[key]['uText'][i])
#         if((j>=402 and j<=461)):
#           temp.append(data[key]['uText'][i])
#         else:
#           continue
#     dataframe_x2[i]=temp 

# for i in range(1024):
#     temp=[]
#     for j,key in enumerate(list(data.keys())):
#             #f.write("%s\n" % data[key]['uText'][i])
#         if((j>=402 and j<=461)):
#           temp.append(data[key]['cText'][i])
#         else:
#           continue
#     dataframe_x2[i+1024]=temp    

        
for k,key in enumerate(list(data.keys())):

        if((k>=402 and k<=461)):
          remp.append(data[key]['gaze'][24])
        else:
          continue
        
dataframe_x2[1024]=remp

# dataframe_x2.to_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_test_t_withcontext.txt', header=None, index=None, sep=' ', mode='w')
dataframe_x2.to_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_test.txt', header=None, index=None, sep=' ', mode='w')


# # print(data['1_105_2']['gaze'])
# # print(data['1_105_2']['gaze'][18])
epoch_loss=0
class HouseDataset(T.utils.data.Dataset):
  # AC  sq ft   style  price   school
  # -1  0.2500  0 1 0  0.5650  0 1 0
  #  1  0.1275  1 0 0  0.3710  0 0 1
  # air condition: -1 = no, +1 = yes
  # style: art_deco, bungalow, colonial
  # school: johnson, kennedy, lincoln
# content dataset for x values Text embeddings.
  def __init__(self, src_file, m_rows=None):
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols= range(0,1025), delimiter=" ",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float32)

    tmp_x = all_xy[:,range(0,1024)]
    tmp_y = all_xy[:,1024].reshape(-1,1)    # 2-D required

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float32).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float32).to(device)


  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx,:]  # or just [idx]
    price = self.y_data[idx,:] 
    return (preds, price)       # tuple of two matrices 



# -----------------------------------------------------------

# linear layers in the neural network.
class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
  #   self.hid1 = T.nn.Linear(8, 10)  # 8-(10-10)-1
  #   self.hid2 = T.nn.Linear(10, 10)
  #   self.oupt = T.nn.Linear(10, 1)

  #   T.nn.init.xavier_uniform_(self.hid1.weight)
  #   T.nn.init.zeros_(self.hid1.bias)
  #   T.nn.init.xavier_uniform_(self.hid2.weight)
  #   T.nn.init.zeros_(self.hid2.bias)
  #   T.nn.init.zeros_(self.oupt.bias)

  # def forward(self, x):
  #   z = T.relu(self.hid1(x))
  #   z = T.relu(self.hid2(z))
  #   z = self.oupt(z)  # no activation
  #   return z
    self.hid1 = T.nn.Linear(1024, 512)  # 8-(10-10)-1
    self.hid2 = T.nn.Linear(512, 512)
    self.hid3 = T.nn.Linear(512, 256)
    self.hid4 = T.nn.Linear(256, 128)
    self.oupt = T.nn.Linear(128, 1)

    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.hid3.weight)
    T.nn.init.zeros_(self.hid3.bias)
    T.nn.init.xavier_uniform_(self.hid4.weight)
    T.nn.init.zeros_(self.hid4.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    z = T.relu(self.hid3(z))
    z = T.relu(self.hid4(z))
    z = self.oupt(z)  # no activation
    return z
  #   # self.hid1 = T.nn.Linear(3072, 1024)  # 8-(10-10)-1
  #   # self.hid2 = T.nn.Linear(1024, 512)  # 8-(10-10)-1
  #   # self.hid3 = T.nn.Linear(512, 512)
  #   # self.hid4 = T.nn.Linear(512, 256)
  #   # self.hid5 = T.nn.Linear(256, 128)
  #   # self.oupt = T.nn.Linear(128, 1)
  #   self.hid1 = T.nn.Linear(2048, 1024)  # 8-(10-10)-1
  #   self.hid2 = T.nn.Linear(1024, 1024)  # 8-(10-10)-1
  #   self.hid3 = T.nn.Linear(1024, 512)
  #   self.hid4 = T.nn.Linear(512, 256)
  #   self.hid5 = T.nn.Linear(256, 128)
  #   self.oupt = T.nn.Linear(128, 1)

  #   T.nn.init.xavier_uniform_(self.hid1.weight)
  #   T.nn.init.zeros_(self.hid1.bias)
  #   T.nn.init.xavier_uniform_(self.hid2.weight)
  #   T.nn.init.zeros_(self.hid2.bias)
  #   T.nn.init.xavier_uniform_(self.hid3.weight)
  #   T.nn.init.zeros_(self.hid3.bias)
  #   T.nn.init.xavier_uniform_(self.hid4.weight)
  #   T.nn.init.zeros_(self.hid4.bias)
  #   T.nn.init.xavier_uniform_(self.hid5.weight)
  #   T.nn.init.zeros_(self.hid5.bias)
  #   T.nn.init.xavier_uniform_(self.oupt.weight)    
  #   T.nn.init.zeros_(self.oupt.bias)

  # def forward(self, x):
  #   z = T.relu(self.hid1(x))
  #   z = T.relu(self.hid2(z))
  #   z = T.relu(self.hid3(z))
  #   z = T.relu(self.hid4(z))
  #   z = T.relu(self.hid5(z))
  #   z = self.oupt(z)  # no activation
# -----------------------------------------------------------

def accuracy(model, ds, pct):
  # assumes model.eval()
  # percent correct within pct of true house price
  n_correct = 0; n_wrong = 0
  Actual_value=[]
  Predicted_value=[]
  for i in range(len(ds)):
    (X, Y) = ds[i]            # (predictors, target)
    Actual_value.append(Y)
    with T.no_grad():
      oupt = model(X)         # computed price
      Predicted_value.append(oupt)
      
    abs_delta = np.abs(oupt.item() - Y.item())
    max_allow = np.abs(pct * Y.item())
    if abs_delta < max_allow:
      n_correct +=1
    else:
      n_wrong += 1
  
  df_actual= pd.DataFrame(Actual_value)
  df_pred= pd.DataFrame(Predicted_value)

  # df_actual.to_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/Actual_values_jaynik.txt', header=None, index=None, sep=' ', mode='w')
  # df_pred.to_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/Predicted_values_jaynik.txt', header=None, index=None, sep=' ', mode='w')
  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  return acc

# -----------------------------------------------------------

def accuracy_quick(model, dataset, pct):
  # assumes model.eval()
  n = len(dataset)
  X = dataset[0:n][0]  # all predictor values
  Y = dataset[0:n][1]  # all target prices
  with T.no_grad():
    oupt = model(X)      # all computed prices

  max_deltas = T.abs(pct * Y)    # max allowable deltas
  abs_deltas = T.abs(oupt - Y)   # actual differences
  
  results = abs_deltas < max_deltas  # [[True, False, . .]]
  acc = T.sum(results, dim=0).item() / n  # dim not needed
  return acc

# -----------------------------------------------------------

def baseline_acc(ds, pct):
  # linear regression model accuracy using just sq. feet
  # y = 1.9559x + 0.0987 (from separate program)
  n_correct = 0; n_wrong = 0
  for i in range(len(ds)):
    (X, Y) = ds[i]           # (predictors, target)
    x = X[1].item()          # sq feet predictor
    y = 1.9559 * x + 0.0987  # computed

    abs_delta = np.abs(oupt.item() - Y.item())
    max_allow = np.abs(pct * Y.item())
    if abs_delta < max_allow:
      n_correct +=1
    else:
      n_wrong += 1

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  return acc   

# -----------------------------------------------------------

def main():
  # 0. get started
  print("\nBegin predict House price \n")
  T.manual_seed(4)  # representative results 
  np.random.seed(4)
  
  # 1. create DataLoader objects
  print("Creating Houses Dataset objects ")
  train_file = "/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_train.txt"
  train_ds = HouseDataset(train_file)  # all 200 rows

  test_file = "/home/development/divyankt/DT_MTP/cikm/data/divyank/predict_test.txt"
  test_ds = HouseDataset(test_file)  # all 40 rows

  bat_size = 16
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True)

  # 2. create network
  net = Net().to(device)

  # 3. train model
  max_epochs = 200#modified no. of epochs
  ep_log_interval = 20
  lrn_rate = 0.005

  loss_func = T.nn.MSELoss()
  # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
  optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)

  print("\nbat_size = %3d " % bat_size)
  print("loss = " + str(loss_func))
  print("optimizer = Adam")
  print("max_epochs = %3d " % max_epochs)
  print("lrn_rate = %0.3f " % lrn_rate)

  print("\nStarting training with saved checkpoints")
  net.train()  # set mode
  for epoch in range(0, max_epochs):
    T.manual_seed(1+epoch)  # recovery reproducibility
    epoch_loss = 0  # for one full epoch
    for (batch_idx, batch) in enumerate(train_ldr):
      (X, Y) = batch                 # (predictors, targets)
      optimizer.zero_grad()          # prepare gradients
      oupt = net(X)                  # predicted prices
      loss_val = loss_func(oupt, Y)  # avg per item in batch
      epoch_loss += loss_val.item()  # accumulate avgs
      loss_val.backward()            # compute gradients
      optimizer.step()               # update wts

    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" % \
       (epoch, epoch_loss))

      # save checkpoint
      # dt = time.strftime("%Y_%m_%d-%H_%M_%S")
      # fn = ".\\Log\\" + str(dt) + str("-") + \
      #  str(epoch) + "_checkpoint.pt"

      # info_dict = { 
      #   'epoch' : epoch,
      #   'net_state' : net.state_dict(),5987 -0.030520143 -0.019784413 0.12985075 -0.029715663 0.051180992 -0.047423102 -0.08909312 -0.05405913 -0.1418366 0.113607034 -0.10281831 -0.2

  # 4. evaluate model accuracy
  print("\nComputing model accuracy")
  net.eval()
#   acc_train = accuracy(net, train_ds, 0.20) 
#   print("Accuracy (within 0.10) on train data = %0.4f" % \
#     acc_train)

  acc_test = accuracy(net, test_ds, 0.40) 
  print("Accuracy (within 0.40) on test data  = %0.4f" % \
    acc_test)
  
  # base_acc_train = baseline_acc(train_ds, 0.10) 
  # print("%0.4f" % base_acc_train)  # 0.7000
  # base_acc_test = baseline_acc(test_ds, 0.10)    
  # print("%0.4f" % base_acc_test)   # 0.7000

  # 5. make a prediction  for all 1024 text features
  # print("\nPredicting gaze features")

  # gaze_list=[]

  # for key in list(data.keys()):
  #   gaze_list.append(key[:-2])

  # for key in list(data2.keys()):
  #   if(key not in gaze_list):
  #     x=np.concatenate((data2[key]['uText'], data2[key]['cText']), axis=None)
  #     unk = x    
  #     unk = T.tensor(unk, dtype=T.float32).to(device) 

  #     with T.no_grad():
  #       pred_gaze = net(unk)
  #     pred_gaze = pred_gaze.item()  # scalar
  #     # str_gaze = \
  #     #   "${:,.2f}".format(pred_gaze)
  #     # print(str_gaze)
  #     with open('/home/development/divyankt/DT_MTP/cikm/data/divyank/gaze_features_pca.txt', 'a') as f:
  #       f.write(str(pred_gaze)+'\n')
# 5. make a prediction  for 8 feature size by pca
  print("\nPredicting gaze features")

  gaze_list=[]

  for key in list(data.keys()):
    gaze_list.append(key[:-2])

  for key in list(data2.keys()):
    if(key not in gaze_list):
      x=data2[key]['uText']
      unk = x    
      
      unk = T.tensor(unk, dtype=T.float32).to(device) 

      with T.no_grad():
        pred_gaze = net(unk)
      pred_gaze = pred_gaze.item()  # scalar
      # str_gaze = \
      #   "${:,.2f}".format(pred_gaze)
      # print(str_gaze)
      with open('/home/development/divyankt/DT_MTP/cikm/data/divyank/gaze_features_pca.txt', 'a') as f:
        f.write(str(pred_gaze)+'\n')



  # 6. save final model (state_dict approach)
  print("\nSaving trained model state")
  fn = "/home/development/divyankt/DT_MTP/cikm/code/models/predict_gaze.pth"
  T.save(net.state_dict(), fn)

  # saved_model = Net()
  # saved_model.load_state_dict(T.load(fn))
  # use saved_model to make prediction(s)

  print("\nEnd House price demo")
  

if __name__ == "__main__":
  main()
