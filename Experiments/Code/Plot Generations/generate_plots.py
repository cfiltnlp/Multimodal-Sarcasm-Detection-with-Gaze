import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
#===========================================================================================================================================
# Line Plot for comparing predicted features.
# aarsh=[]
jaynik=[0.21,0.12,0.15]
# predicted_aarsh=[]
predicted_jaynik=[0.27,0.09,0.19]


# df= pd.DataFrame(pd.read_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/Actual_values.txt'))

# df2= pd.DataFrame(pd.read_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/Predicted_values.txt'))

# df3= pd.DataFrame(pd.read_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/Actual_values_jaynik.txt'))

# df4= pd.DataFrame(pd.read_csv(r'/home/development/divyankt/DT_MTP/cikm/data/divyank/Predicted_values_jaynik.txt'))

# for index, rows in df.iterrows():
#     aarsh.append(rows)

# for index, rows in df2.iterrows():
#     predicted_aarsh.append(rows)

# for index, rows in df3.iterrows():
#     jaynik.append(rows)

# for index, rows in df4.iterrows():
#     predicted_jaynik.append(rows)

# diff_sq=0
# for i in range(len(jaynik)-1):
#     diff= jaynik[i]-predicted_jaynik[i]
#     diff_sq=diff_sq + diff*diff

# print(diff_sq)  

#y1 = np.array(aarsh)
# y2 = np.array(predicted_aarsh)
y3 = np.array(jaynik)
y4 = np.array(predicted_jaynik)

# plt.plot(y1)
# plt.plot(y2)
plt.plot(y3,linestyle='-.')
plt.plot(y4)

plt.legend(["Pearson corr","spearmann corr"], loc ="upper right")

# plt.legend(["annotator1_predicted","annotator2_predicted"], loc ="upper right")


plt.savefig('/home/development/divyankt/DT_MTP/cikm/data/divyank/correlation_gaze.png', bbox_inches='tight')

#========================================================================================================================================================
#Bar plot for kbest features comparison with baseline......
# set width of bar
# barWidth = 0.12
# fig = plt.subplots(figsize =(8, 6))
 
# # set height of bar
# Precision = [0.689, 0.642,0.661,0.651]
# Recall = [0.67, 0.64, 0.657, 0.650]
# F1_score = [0.689, 0.642, 0.661, 0.651]

# # Set position of bar on X axis
# br1 = np.arange(len(Precision))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# # Make the plot
# plt.bar(br1, Precision, color ='r', width = barWidth,
#         edgecolor ='grey', label ='Precision')
# plt.bar(br2, Recall, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Recall')
# plt.bar(br3, F1_score, color ='g', width = barWidth,
#         edgecolor ='grey', label ='F1-Score')
 
# # Adding Xticks
# plt.xlabel('Model', fontweight ='bold', fontsize = 15)
# plt.ylabel('Score', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(Precision))],
#         ['FFNN','Roberta-base','Roberta + SA adapter','Roberta + QE adapter'])
 
# plt.legend()
# plt.savefig('/home/development/divyankt/DT_MTP/cikm/data/divyank/kbest.png', bbox_inches='tight')


# #=========================================================================================================================================================

# barWidth = 0.12
# fig = plt.subplots(figsize =(8, 6))
 
# # set height of bar
# F1_score = [0.707, 0.596]
 
# # Set position of bar on X axis
# br3 = np.arange(len(F1_score))

 
# # Make the plot
# # plt.bar(br1, Precision, color ='r', width = barWidth,
# #         edgecolor ='grey', label ='Precision')
# # plt.bar(br2, Recall, color ='b', width = barWidth,
# #         edgecolor ='grey', label ='Recall')
# plt.bar(br3, F1_score, color ='y', width = barWidth,
#         edgecolor ='grey', label ='F1-Score')
 
# # Adding Xticks
# plt.xlabel('Number of features', fontweight ='bold', fontsize = 15)
# plt.ylabel('Score', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(F1_score))],
#         ['F1-score using V T A modality','F1-score using only Gaze features'])
 
# plt.legend()
# plt.savefig('/home/development/divyankt/DT_MTP/cikm/data/divyank/kbest2.png', bbox_inches='tight')
