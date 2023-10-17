import torch
import torch.nn as nn
import numpy as np
import random

'''Comment these to randomize the model training'''
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

"""This file contains all the models we have used for -
1. Multiclass Implicit Emotion Classification
2. Multiclass Explicit Emotion Classification
3. Multimodal Sarcasm Detection
    3.1 for MUStARD
    3.2 for MUStARD++


Class names and Variable names are self explanatory
Before setting the values to input embedding  we first sort the modality name in descending order. (VTA) in order to remove randomness in the model

Parameters:
      input_embedding_A:
            Takes the input dimension of first modality
      input_embedding_B:
            Takes the input dimension of second modality
      input_embedding_C:
            Takes the input dimension of third modality
      shared_embedding:
            This is the dimension size to which we have to project all modality, to have equal dimention input from each input modality
      projection_embedding:
            This is the intermidiate dimension size to which project our shared embedding to calsul;ate attention
      dropout: 
            Parameter to pass dropout (to be hyper-tuned)


we assign "num_classes" variable dependeing upon the task
for
    a. num_classes=5 (5) (Multiclass Implicit Emotion Classification)
    b. num_classes=9 (Multiclass Explicit Emotion Classification)
    c. num_classes=2 (Multimodal Sarcasm Detection)

We have used Softmax As Output layer
"""

audio_embedding_size = 291#314 #291, 319

######################################################################################################################################################################
# class Speaker_Dependent_4_Mode_with_Context(nn.Module):  #prunned_features
#     def __init__(self, n_speaker=24, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size,input_embedding_D=25, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=2):
#         super(Speaker_Dependent_4_Mode_with_Context, self).__init__()

#         self.n_speaker = n_speaker

#         self.input_embedding_A = input_embedding_A
#         self.input_embedding_B = input_embedding_B
#         self.input_embedding_C = input_embedding_C
#         self.input_embedding_D = input_embedding_D 

#         self.shared_embedding = shared_embedding
#         self.projection_embedding = projection_embedding
#         self.num_classes = num_classes
#         self.dropout = dropout

#         self.A_context_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)
#         self.A_utterance_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)

#         self.C_context_share = nn.Linear(
#             self.input_embedding_C, self.shared_embedding)
#         self.C_utterance_share = nn.Linear(
#             self.input_embedding_C, self.shared_embedding)

#         self.B_context_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)
#         self.B_utterance_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)
        
#         self.D_context_share = nn.Linear(
#             self.input_embedding_D, self.shared_embedding)
#         self.D_utterance_share = nn.Linear(
#             self.input_embedding_D, self.shared_embedding)

#         self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_D_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_D_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.collabrative_gate_1 = nn.Linear(
#             2*self.shared_embedding, self.projection_embedding)
#         self.collabrative_gate_2 = nn.Linear(
#             self.projection_embedding, self.shared_embedding)
#     #             w = list(l.parameters())   ##changed modified modify
#     # print(w)
#         self.pred_module = nn.Sequential(
#             l=nn.Linear(self.n_speaker+4*self.shared_embedding,    #modify
#                       3*self.shared_embedding),
#             nn.BatchNorm1d(3*self.shared_embedding),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
#             nn.BatchNorm1d(2*self.shared_embedding),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(2*self.shared_embedding,  512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512,  128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128,  self.num_classes)

#         )


#     def attention(self, featureA, featureB):
#         """ This method takes two features and caluate the attention """
#         input = torch.cat((featureA, featureB), dim=1)
#         return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

#     def attention_aggregator(self, feA, feB, feC, feD, feE, feF, feG, feH):   #modified
#         """ This method caluates the attention for feA with respect to others"""    
#         input = self.attention(feA, feB) + self.attention(feA, feC) + self.attention(
#             feA, feD) + self.attention(feA, feE) + self.attention(feA, feF) + self.attention(feA, feG)+ self.attention(feA, feH)
#         # here we call for pairwise attention
#         return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

#     def forward(self, uA, cA, uB, cB, uC, cC, speaker_embedding, uD, cD):
#         """Args:
#                 uA:
#                     Utternace Video
#                 uB:
#                     Utternace Text
#                 uC:
#                     Utternace Audio
#                 cA:
#                     Context Video
#                 cB:
#                     Context Text
#                 cC:
#                     Context Audio
#                 cD: Context gaze

#                 uD: utterance gaze

#             Returns:
#                 probability of emotion classe
#                 (
#                     Since we have used Crossentropy as loss function,
#                     Therefore we have not used softmax here because Crossentropy perform Softmax while calculating loss
#                     While evaluation we have to perform softmax explicitly
#                 )
#         """
#         """making Feature Projection in order to make all feature of same dimension"""

#         shared_A_context = self.norm_A_context(
#             nn.functional.relu(self.A_context_share(cA)))
#         shared_A_utterance = self.norm_A_utterance(
#             nn.functional.relu(self.A_utterance_share(uA)))

#         shared_C_context = self.norm_C_context(
#             nn.functional.relu(self.C_context_share(cC)))
#         shared_C_utterance = self.norm_C_utterance(
#             nn.functional.relu(self.C_utterance_share(uC)))

#         shared_B_context = self.norm_B_context(
#             nn.functional.relu(self.B_context_share(cB)))
#         shared_B_utterance = self.norm_B_utterance(
#             nn.functional.relu(self.B_utterance_share(uB)))
        
#         shared_D_context = self.norm_D_context(
#             nn.functional.relu(self.D_context_share(cD)))
#         shared_D_utterance = self.norm_D_utterance(
#             nn.functional.relu(self.D_utterance_share(uD)))

#         updated_shared_A = shared_A_utterance * self.attention_aggregator(
#             shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance,shared_D_context, shared_D_utterance)
#         updated_shared_C = shared_C_utterance * self.attention_aggregator(
#             shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance,shared_D_context, shared_D_utterance)
#         updated_shared_B = shared_B_utterance * self.attention_aggregator(
#             shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance,shared_D_context, shared_D_utterance)

#         updated_shared_D = shared_D_utterance * self.attention_aggregator(
#             shared_D_utterance, shared_D_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance,shared_B_context, shared_B_utterance)

#         temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
#         remp = torch.cat((temp, updated_shared_B), dim=1)
#         input = torch.cat((remp, updated_shared_D), dim=1)

#         input = torch.cat((input, speaker_embedding), dim=1)

#         return self.pred_module(input)
# ############################################################################################################################################################
# class Speaker_Independent_4_Mode_with_Context(nn.Module): #prunned_features
#     def __init__(self, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, input_embedding_D=31, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=2):
#         super(Speaker_Independent_4_Mode_with_Context, self).__init__()

#         self.input_embedding_A = input_embedding_A
#         self.input_embedding_B = input_embedding_B
#         self.input_embedding_C = input_embedding_C
#         self.input_embedding_D = input_embedding_D
#         self.shared_embedding = shared_embedding
#         self.projection_embedding = projection_embedding
#         self.num_classes = num_classes
#         self.dropout = dropout

#         self.A_context_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)
#         self.A_utterance_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)

#         self.C_context_share = nn.Linear(
#             self.input_embedding_C, self.shared_embedding)
#         self.C_utterance_share = nn.Linear(
#             self.input_embedding_C, self.shared_embedding)

#         self.B_context_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)
#         self.B_utterance_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)

#         self.D_context_share = nn.Linear(
#             self.input_embedding_D, self.shared_embedding)
#         self.D_utterance_share = nn.Linear(
#             self.input_embedding_D, self.shared_embedding)

#         self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_D_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_D_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.collabrative_gate_1 = nn.Linear(
#             2*self.shared_embedding, self.projection_embedding)
#         self.collabrative_gate_2 = nn.Linear(
#             self.projection_embedding, self.shared_embedding)

#         self.pred_module = nn.Sequential(
#             nn.Linear(4*self.shared_embedding, 3*self.shared_embedding),
#             nn.BatchNorm1d(3*self.shared_embedding),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
#             nn.BatchNorm1d(2*self.shared_embedding),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(2*self.shared_embedding,  512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512,  128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128,  self.num_classes)
#         )

#     def attention(self, featureA, featureB):
#         """ This method takes two features and caluate the attention """
#         input = torch.cat((featureA, featureB), dim=1)
#         return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

#     def attention_aggregator(self, feA, feB, feC, feD, feE, feF, feG, feH):
#         """ This method caluates the attention for feA with respect to others"""    
#         input = self.attention(feA, feB) + \
#             self.attention(feA, feC) + \
#             self.attention(feA, feD) + \
#             self.attention(feA, feE) + \
#             self.attention(feA, feF) + \
#             self.attention(feA, feG) + \
#             self.attention(feA, feH)
#         # here we call for pairwise attention
#         return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

#     def forward(self, uA, cA, uB, cB, uC, cC, uD, cD):
#         """Args:
#                 uA:
#                     Utternace Video
#                 uB:
#                     Utternace Text
#                 uC:
#                     Utternace Audio
#                 cA:
#                     Context Video
#                 cB:
#                     Context Text
#                 cC:
#                     Context Audio
                
#                 cD:
#                     Context gaze
#                 uD:
#                     utt gaze
#             Returns:
#                 probability of emotion classe
#                 (
#                     Since we have used Crossentropy as loss function,
#                     Therefore we have not used softmax here because Crossentropy perform Softmax while calculating loss
#                     While evaluation we have to perform softmax explicitly
#                 )
#         """
#         """making Feature Projection in order to make all feature of same dimension"""

#         shared_A_context = self.norm_A_context(
#             nn.functional.relu(self.A_context_share(cA)))
#         shared_A_utterance = self.norm_A_utterance(
#             nn.functional.relu(self.A_utterance_share(uA)))

#         shared_C_context = self.norm_C_context(
#             nn.functional.relu(self.C_context_share(cC)))
#         shared_C_utterance = self.norm_C_utterance(
#             nn.functional.relu(self.C_utterance_share(uC)))

#         shared_B_context = self.norm_B_context(
#             nn.functional.relu(self.B_context_share(cB)))
#         shared_B_utterance = self.norm_B_utterance(
#             nn.functional.relu(self.B_utterance_share(uB)))

#         shared_D_context = self.norm_D_context(
#             nn.functional.relu(self.D_context_share(cD)))
#         shared_D_utterance = self.norm_D_utterance(
#             nn.functional.relu(self.D_utterance_share(uD)))

#         # Feature Modulation

#         updated_shared_A = shared_A_utterance * self.attention_aggregator(
#             shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance,shared_D_context, shared_D_utterance)
#         updated_shared_C = shared_C_utterance * self.attention_aggregator(
#             shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance,shared_D_context, shared_D_utterance)
#         updated_shared_B = shared_B_utterance * self.attention_aggregator(
#             shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance,shared_D_context, shared_D_utterance)

#         updated_shared_D = shared_D_utterance * self.attention_aggregator(
#             shared_D_utterance, shared_D_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance,shared_B_context, shared_B_utterance)



#         temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
#         remp = torch.cat((temp, updated_shared_B), dim=1)
#         input = torch.cat((remp, updated_shared_D), dim=1)

       
#         return self.pred_module(input)

###############################################################################################################################################################

# class Speaker_Independent_Triple_Mode_with_Context(nn.Module):
#     def __init__(self, input_embedding_A=1024, input_embedding_B=20, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=2):
#         super(Speaker_Independent_Triple_Mode_with_Context, self).__init__()

#         self.input_embedding_A = input_embedding_A
#         self.input_embedding_B = input_embedding_B
#         self.input_embedding_C = input_embedding_C
#         self.shared_embedding = shared_embedding
#         self.projection_embedding = projection_embedding
#         self.num_classes = num_classes
#         self.dropout = dropout

#         self.A_context_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)
#         self.A_utterance_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)

#         self.C_context_share = nn.Linear(
#             self.input_embedding_C, self.shared_embedding)
#         self.C_utterance_share = nn.Linear(
#             self.input_embedding_C, self.shared_embedding)

#         self.B_context_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)
#         self.B_utterance_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)

#         self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.collabrative_gate_1 = nn.Linear(
#             2*self.shared_embedding, self.projection_embedding)
#         self.collabrative_gate_2 = nn.Linear(
#             self.projection_embedding, self.shared_embedding)

#         self.pred_module = nn.Sequential(
#             nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
#             nn.BatchNorm1d(2*self.shared_embedding),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(2*self.shared_embedding, self.shared_embedding),
#             nn.BatchNorm1d(self.shared_embedding),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.shared_embedding,  512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512,  128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128,  self.num_classes)
#         )

#     def attention(self, featureA, featureB):
#         """ This method takes two features and caluate the attention """
#         input = torch.cat((featureA, featureB), dim=1)
#         return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

#     def attention_aggregator(self, feA, feB, feC, feD, feE, feF):
#         """ This method caluates the attention for feA with respect to others"""    
#         input = self.attention(feA, feB) + \
#             self.attention(feA, feC) + \
#             self.attention(feA, feD) + \
#             self.attention(feA, feE) + \
#             self.attention(feA, feF)
#         # here we call for pairwise attention
#         return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

#     def forward(self, uA, cA, uB, cB, uC, cC):
#         """Args:
#                 uA:
#                     Utternace Video
#                 uB:
#                     Utternace Text
#                 uC:
#                     Utternace Audio
#                 cA:
#                     Context Video
#                 cB:
#                     Context Text
#                 cC:
#                     Context Audio

#             Returns:
#                 probability of emotion classe
#                 (
#                     Since we have used Crossentropy as loss function,
#                     Therefore we have not used softmax here because Crossentropy perform Softmax while calculating loss
#                     While evaluation we have to perform softmax explicitly
#                 )
#         """
#         """making Feature Projection in order to make all feature of same dimension"""
        
#         shared_A_context = self.norm_A_context(
#             nn.functional.relu(self.A_context_share(cA)))
        
#         shared_A_utterance = self.norm_A_utterance(
#             nn.functional.relu(self.A_utterance_share(uA)))

#         shared_C_context = self.norm_C_context(
#             nn.functional.relu(self.C_context_share(cC)))
#         shared_C_utterance = self.norm_C_utterance(
#             nn.functional.relu(self.C_utterance_share(uC)))

#         shared_B_context = self.norm_B_context(
#             nn.functional.relu(self.B_context_share(cB)))
#         shared_B_utterance = self.norm_B_utterance(
#             nn.functional.relu(self.B_utterance_share(uB)))

#         # Feature Modulation

#         updated_shared_A = shared_A_utterance * self.attention_aggregator(
#             shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance)
#         updated_shared_C = shared_C_utterance * self.attention_aggregator(
#             shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance)
#         updated_shared_B = shared_B_utterance * self.attention_aggregator(
#             shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance)

#         temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
#         input = torch.cat((temp, updated_shared_B), dim=1)

#         return self.pred_module(input)


class Speaker_Independent_Triple_Mode_with_Context(nn.Module):
    def __init__(self, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Triple_Mode_with_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_context_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)
        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD, feE, feF):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + \
            self.attention(feA, feC) + \
            self.attention(feA, feD) + \
            self.attention(feA, feE) + \
            self.attention(feA, feF)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, uC, cC):
        """Args:
                uA:
                    Utternace Video
                uB:
                    Utternace Text
                uC:
                    Utternace Audio
                cA:
                    Context Video
                cB:
                    Context Text
                cC:
                    Context Audio

            Returns:
                probability of emotion classe
                (
                    Since we have used Crossentropy as loss function,
                    Therefore we have not used softmax here because Crossentropy perform Softmax while calculating loss
                    While evaluation we have to perform softmax explicitly
                )
        """
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_context = self.norm_C_context(
            nn.functional.relu(self.C_context_share(cC)))
        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        # Feature Modulation

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Independent_Dual_Mode_with_Context(nn.Module):
    def __init__(self, input_embedding_A=2048, input_embedding_B=291, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Dual_Mode_with_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + \
            self.attention(feA, feC) + \
            self.attention(feA, feD)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_B_context, shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        return self.pred_module(input)

# class Speaker_Independent_Dual_Mode_with_Context(nn.Module):  #prunned_features
#     def __init__(self, input_embedding_A=25, input_embedding_B=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=2):
#         super(Speaker_Independent_Dual_Mode_with_Context, self).__init__()

#         self.input_embedding_A = input_embedding_A
#         self.input_embedding_B = input_embedding_B
#         self.shared_embedding = shared_embedding
#         self.projection_embedding = projection_embedding
#         self.num_classes = num_classes
#         self.dropout = dropout

#         self.A_context_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)
#         self.A_utterance_share = nn.Linear(
#             self.input_embedding_A, self.shared_embedding)

#         self.B_context_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)
#         self.B_utterance_share = nn.Linear(
#             self.input_embedding_B, self.shared_embedding)

#         self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
#         self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.collabrative_gate_1 = nn.Linear(
#             2*self.shared_embedding, self.projection_embedding)
#         self.collabrative_gate_2 = nn.Linear(
#             self.projection_embedding, self.shared_embedding)

#         self.pred_module = nn.Sequential(
#             nn.Linear(2*self.shared_embedding, self.shared_embedding),
#             nn.BatchNorm1d(self.shared_embedding),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.shared_embedding,  512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512,  128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128,  self.num_classes)
#         )

#     def attention(self, featureA, featureB):
#         """ This method takes two features and caluate the attention """
#         input = torch.cat((featureA, featureB), dim=1)
#         return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

#     def attention_aggregator(self, feA, feB, feC, feD):
#         """ This method caluates the attention for feA with respect to others"""    
#         input = self.attention(feA, feB) + \
#             self.attention(feA, feC) + \
#             self.attention(feA, feD)
#         # here we call for pairwise attention
#         return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

#     def forward(self, uA, cA, uB, cB):
#         """making Feature Projection in order to make all feature of same dimension"""

#         shared_A_context = self.norm_A_context(
#             nn.functional.relu(self.A_context_share(cA)))
#         shared_A_utterance = self.norm_A_utterance(
#             nn.functional.relu(self.A_utterance_share(uA)))

#         shared_B_context = self.norm_B_context(
#             nn.functional.relu(self.B_context_share(cB)))
#         shared_B_utterance = self.norm_B_utterance(
#             nn.functional.relu(self.B_utterance_share(uB)))

#         updated_shared_A = shared_A_utterance * self.attention_aggregator(
#             shared_A_utterance, shared_A_context, shared_B_context, shared_B_utterance)

#         updated_shared_B = shared_B_utterance * self.attention_aggregator(
#             shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance)

#         input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

#         return self.pred_module(input)
################################################################################################################################################################################################################


class Speaker_Independent_Single_Mode_with_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Single_Mode_with_Context, self).__init__()

        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.context_share = nn.Linear(
            self.input_embedding, self.shared_embedding)
        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_context = self.norm_context(
            nn.functional.relu(self.context_share(cA)))
        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_context)

        input = updated_shared

        return self.pred_module(updated_shared)

################################################################################################################################################################################################################


class Speaker_Independent_Triple_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Triple_Mode_without_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + self.attention(feA, feC)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, uB,  uC):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,   shared_C_utterance,  shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance,   shared_A_utterance,  shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,   shared_A_utterance,  shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Independent_Dual_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Dual_Mode_without_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA,  uB):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,  shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,  shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


# class Speaker_Independent_Single_Mode_without_Context(nn.Module):
#     def __init__(self, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.2, num_classes=5):
#         super(Speaker_Independent_Single_Mode_without_Context, self).__init__()
#         print("No. of classes:",num_classes)
#         self.input_embedding = input_embedding_A

#         self.shared_embedding = shared_embedding
#         self.projection_embedding = projection_embedding
#         self.num_classes = num_classes
#         self.dropout = dropout

#         self.utterance_share = nn.Linear(
#             self.input_embedding, self.shared_embedding)

#         self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

#         self.collabrative_gate_1 = nn.Linear(
#             2*self.shared_embedding, self.projection_embedding)
#         self.collabrative_gate_2 = nn.Linear(
#             self.projection_embedding, self.shared_embedding)

#         self.pred_module = nn.Sequential(
#             # nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
#             # nn.BatchNorm1d(2*self.shared_embedding),
#             # nn.ReLU(),
#             # nn.Linear(2*self.shared_embedding, self.shared_embedding),
#             # nn.BatchNorm1d(self.shared_embedding),
#             # nn.ReLU(),
#             nn.Linear(self.shared_embedding,  512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512,  128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128,  self.num_classes)
#         )

#     def attention(self, featureA, featureB):
#         """ This method takes two features and caluate the attention """
#         input = torch.cat((featureA, featureB), dim=1)
#         return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

#     def attention_aggregator(self, feA, feB):
#         """ This method caluates the attention for feA with respect to others"""    
#         input = self.attention(feA, feB)
#         # here we call for pairwise attention
#         return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

#     def forward(self, uA):
#         """making Feature Projection in order to make all feature of same dimension"""

#         shared_utterance = self.norm_utterance(
#             nn.functional.relu(self.utterance_share(uA)))

#         updated_shared = shared_utterance * self.attention_aggregator(
#             shared_utterance, shared_utterance)

#         input = updated_shared

#         return self.pred_module(updated_shared)


class Speaker_Independent_Single_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=25, shared_embedding=1024, projection_embedding=512, dropout=0.2, num_classes=5):
        super(Speaker_Independent_Single_Mode_without_Context, self).__init__()
        print("No. of classes:",num_classes)
        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            # nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            # nn.BatchNorm1d(2*self.shared_embedding),
            # nn.ReLU(),
            # nn.Linear(2*self.shared_embedding, self.shared_embedding),
            # nn.BatchNorm1d(self.shared_embedding),
            # nn.ReLU(),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_utterance)

        input = updated_shared

        return self.pred_module(updated_shared)
################################################################################################################################################################################################################


class Speaker_Dependent_Triple_Mode_with_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Triple_Mode_with_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_context_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)
        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+3*self.shared_embedding,
                      2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)

        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD, feE, feF):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + self.attention(feA, feC) + self.attention(
            feA, feD) + self.attention(feA, feE) + self.attention(feA, feF)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, uC, cC, speaker_embedding):
        """Args:
                uA:
                    Utternace Video
                uB:
                    Utternace Text
                uC:
                    Utternace Audio
                cA:
                    Context Video
                cB:
                    Context Text
                cC:
                    Context Audio

            Returns:
                probability of emotion classe
                (
                    Since we have used Crossentropy as loss function,
                    Therefore we have not used softmax here because Crossentropy perform Softmax while calculating loss
                    While evaluation we have to perform softmax explicitly
                )
        """
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_context = self.norm_C_context(
            nn.functional.relu(self.C_context_share(cC)))
        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Dual_Mode_with_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Dual_Mode_with_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+2*self.shared_embedding,
                      self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + self.attention(feA,
                                                          feC) + self.attention(feA, feD)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_B_context, shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Single_Mode_with_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Single_Mode_with_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.context_share = nn.Linear(
            self.input_embedding, self.shared_embedding)
        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_context = self.norm_context(
            nn.functional.relu(self.context_share(cA)))
        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_context)

        input = torch.cat((updated_shared, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Triple_Mode_without_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Triple_Mode_without_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+3*self.shared_embedding,
                      2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + self.attention(feA, feC)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, uB,  uC, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,   shared_C_utterance,  shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance,   shared_A_utterance,  shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,   shared_A_utterance,  shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Dual_Mode_without_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Dual_Mode_without_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+2*self.shared_embedding,
                      self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA,  uB, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,  shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,  shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Single_Mode_without_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Single_Mode_without_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+self.shared_embedding, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_utterance)

        input = torch.cat((updated_shared, speaker_embedding), dim=1)
        # print(input.shape)
        # print(speaker_embedding.shape)
        # print(self.n_speaker+self.shared_embedding)
        return self.pred_module(input)

################################################################################################################################################################################################################
