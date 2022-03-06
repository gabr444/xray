import torch
import numpy as np
from net import Net
import sys
from data import process

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        deviceType = "gpu"
    else:
        device = torch.device("cpu")
        deviceType = "cpu"

    net = Net().to(device)
    # Load model parameters
    net.load_state_dict(torch.load("model.pt"))
    # Set net in evaluation mode (for testing)
    net.eval()
    imgPath = "data/val/PNEUMONIA/person1946_bacteria_4874.jpeg"
    data = np.array(process(imgPath), dtype=np.float32)
    # Run with torch.no_grad() to avoid computing unecessary gradients.
    with torch.no_grad():
        x = torch.from_numpy(data).to(device).view(-1, 1, 250, 200)
        output = net(x)
        predict = torch.argmax(output, dim=1)
    if(predict.item() == 1):
        print("Model predicts pneumonia")
    else:
        print("Model couldn't find signs of pneumonia")
