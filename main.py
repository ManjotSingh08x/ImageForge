import torch 
import numpy
import math 
import flask

if torch.cuda.is_available():
    device = torch.device('cuda')

print(device)