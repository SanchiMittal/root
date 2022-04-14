import ROOT
import numpy as np
import torch
from torch import nn

def Batch_Generator(features, targets, batch_size=1):
    batch_count = 0
    features_batch = []
    targets_batch = []

    dataset_size = len(list(features.values())[0])

    for i in range(dataset_size):
        features_batch.append(torch.Tensor([features[x][i] for x in features.keys()]))
        targets_batch.append(torch.Tensor([targets[x][i] for x in targets.keys()]))
        if batch_count==batch_size-1:
            yield torch.stack(features_batch), torch.stack(targets_batch)
            features_batch = []
            targets_batch = []
            batch_count = 0
        else:
            batch_count += 1
        


df = ROOT.RDataFrame("sig_tree", "Higgs_data.root")
x_df = df.AsNumpy(columns=["jet1_phi","jet1_eta", "jet2_pt"])
y_df = df.AsNumpy(columns=["jet3_b-tag"])



batch_size = 32
# train_generator = Batch_Generator(x_df, y_df, batch_size)
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD
# data_loader = torch.utils.data.DataLoader(Batch_Generator(x_df, y_df, batch_size))


model = nn.Sequential()
model.add_module('linear_1', nn.Linear(in_features=3, out_features=64))
model.add_module('relu', nn.ReLU())
model.add_module('linear_2', nn.Linear(in_features=64, out_features=1))
model.add_module('softmax', nn.Softmax(dim=1))


def train(model, num_epochs, optimizer, criterion):
    optim = optimizer(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        print(epoch)
        # Training Loop
        # Set to train model
        model.train()
        running_train_loss = 0.0
        batches = Batch_Generator(x_df, y_df, batch_size)
        for i, (X, y) in enumerate(batches):
            optim.zero_grad()
            output = model(X)
            train_loss = criterion(output, y)
            train_loss.backward()
            optim.step()

            running_train_loss = running_train_loss + ((1 / (i + 1)) * (train_loss.data - running_train_loss))
        
            # print train statistics
            # running_train_loss += train_loss.item()
            if i % batch_size == batch_size-1 :   # print every 32 mini-batches
                print("[{}, {}] train loss: {:.3f}".format(epoch+1, i+1, running_train_loss / 32))
                # running_train_loss = 0.0
    
 

    print("Finished Training on {} Epochs!".format(epoch+1))
 
    return model
 

train(model, 20, optimizer, loss )

 