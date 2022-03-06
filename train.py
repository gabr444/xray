import torch
import numpy as np
from net import Net
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import tqdm

class dataset(Dataset):
    def __init__(self):
        self.samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.samples

if __name__ == '__main__':
    deviceType = ""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        deviceType = "gpu"
    else:
        device = torch.device("cpu")
        deviceType = "cpu"
    print("Training on", deviceType)
    net = Net().to(device)
    try:
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
    except:
        print("Failed to load training data files")
        sys.exit()

    batch_size = 32
    datas = dataset()
    loader = DataLoader(dataset=datas, batch_size=batch_size, shuffle=True)
    # Use cross entropy loss function (classification)
    criterion = nn.CrossEntropyLoss()
    # Use the ADAM optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # Set number of epochs. This is the amount of times that the model will process the whole dataset.
    epochs = 5
    print("Training with", epochs, "epochs")
    for epoch in range(epochs):
        totalLoss=0
        # Use tqdm to show a progress bar for each epoch.
        for (x_data, y_data) in tqdm(loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            x_data = x_data.view(-1, 1, 250, 200)
            output = net(x_data)
            loss = criterion(output, y_data)
            # Reset gradients (change to weights) from the last update
            optimizer.zero_grad()
            # Backpropagate loss. Used to determine the change that's needed to the weights
            loss.backward()
            # Add loss to totalLoss
            totalLoss+=loss.item()
            # Make changes to weights
            optimizer.step()
        print(epoch+1, "loss:", totalLoss)
    # Save model
    try:
        torch.save(net.state_dict(), "model.pt")
    except:
        print("Failed to save model")
        sys.exit()
    # Test the model
    try:
        x_val = np.load('x_val.npy')
        y_val = np.load('y_val.npy')
    except:
        print("Failed to load test data files")
        sys.exit()

    correct = 0
    total = 0
    # Set net in evaluation mode (for testing)
    net.eval()
    # Run with torch.no_grad() to avoid computing unecessary gradients.
    with torch.no_grad():
        for i in range(len(x_val)):
            x = torch.from_numpy(x_val[i]).to(device)
            x = x.view(-1, 1, 250, 200)
            output = net(x)
            predict = torch.argmax(output, dim=1)
            if predict.item() == y_val[i]:
                correct+=1
            total+=1
    print("Model accuracy:", correct/total)
