# ImageForge

Project based on the following proposal: [Proposal PDF](https://drive.google.com/file/d/1ZMRwdBRHLBz8K4tykRPaMaqEdKj7pjNK/view?usp=drive_link)

Drive link for daily updates and revised timeline: [Drive Link](https://drive.google.com/drive/folders/1haHVB1JbDJXEDeHzCLc2UZ2R-1ytNdKW?usp=drive_link)

Kaggle Notebook: [Notebook](https://www.kaggle.com/code/manjotsingh08x/imageforgemain)


# Usage

Download Model and Network architecture from the drive link: [Model and Architecture](https://drive.google.com/drive/u/1/folders/1c-5tqHWRlHVjFq-suPen_SKiV5DF1vxR)

## For main model:
```python
import torch
from PartialConvArch256 import *
model = PartialConvUNet()
model = torch.load('ImageInpainting600k.pt', map_location=torch.device('cpu')) 
```
# Timeline 

Week 1: 15 – 21 December 

    Learn basics of pytorch and understand how to implement basic features 

    As a learning practice, follow a flower classification tutorial to learn how to implement CNN in pytorch 

    Learn math behind CNNs and Partial convolution networks 

    Learn basics of open Cv and how to create simple bitmasks. Learn how to process base64 bitmasks to communicate with the website and the model processor 

    Set up a basic website using bootstrap to allow users to upload their images 

Week 2: 22 – 28 December – until mid-evaluation 

    Start on data collection and organization. 

    Download and set up the ImageNet dataset. Learn about epochs in Pytorch 

    Use OpenCV to generate random bitmasks for the network 

    Start working on the main architecture of the model in Pytorch following the research paper. 

    Start training the model on a smaller scale to test for bugs and confirm main functionality 

Week 3: 29 Dec – 4 Jan 

    Start working on larger scale and testing  

    If the model is not returning satisfactory results, try diffusion models 

    Learn about diffusion models and how to implement them in Pytorch if PCN doesn’t work  

    Start working on more functionality in website. Such as more selection tools (rect, circle, polygon) 

    As a secondary goal, work on semantic segmentation models (SSM)for more selectivity in the image 

    Find and sanitize the dataset for SSMs 

    Start training on semantic segmentation model 

Week 4: 5 – 11 Jan 

    Work on integration of semantic segmentation model with the existing website so users can directly select unwanted objects and remove them completely 

    # todo: prupose of project, networks and dataset used. 

    Final touchups 

12 Jan – Final Evaluation 

    Work on documentation and cleanup 

    
