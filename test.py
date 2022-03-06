import torch
import numpy as np
from net import Net
import sys
from data import process
import cv2

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
    data = process(imgPath)
    # Run with torch.no_grad() to avoid computing unecessary gradients.
    with torch.no_grad():
        # Convert image list to tensor
        x = torch.from_numpy(np.array(data, dtype=np.float32)).to(device).view(-1, 1, 250, 200)
        output = net(x)
        # Get prediction
        predict = torch.argmax(output, dim=1)
    if(predict.item() == 1):
        print("Model predicts pneumonia")
    else:
        print("Model could not find signs of pneumonia")
    # Show x-ray image
    cv2.imshow("x-ray", data)
    cv2.waitKey(0)
