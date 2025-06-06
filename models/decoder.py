import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import sys

class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()

        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))

        self.classifier = nn.Sequential(*Layers)


    def aggregate(self, embeddings, mode):
        if mode == 'Maxmin':
            max_val, _ = torch.max(embeddings, dim=0)
            min_val, _ = torch.min(embeddings, dim=0)
            return max_val - min_val
        elif mode == 'Avg' :
            embedding = embeddings.mean(dim=0).squeeze()
            return embedding
            
                
    def classify(self, embedding):
        return F.sigmoid(self.classifier(embedding))
    
    def forward(self, v_feat, hedge_info, mode):
        preds =[]
        for hedge in hedge_info :
            embeddings = v_feat[hedge]
            embedding = self.aggregate(embeddings, mode)
            pred = self.classify(embedding)
            preds.append(pred)
        preds = torch.stack(preds)
        return preds 
    