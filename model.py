"""
Matheus Espalaor
matheusespalaor14@gmail.com

In this project, we implemented a naive approach to face recognition as an initial step toward building a functional system. 
At this stage, the focus was not on optimizing the model, but rather on achieving a working version capable of performing inference. 
The goal was to validate the core functionality and establish a solid foundation for future improvements.
"""

"""
The reference for application is:
    Dive into Deep Learning, Chapter 14, Section 14.4, Anchor-boxes

"""


"""
@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}

We will use the WIDERFACE Dataset: http://shuoyang1213.me/WIDERFACE/ from Yang Shou
And for training and test, for first, we will use the .txt files provided by them, that select images for training and testing.
"""

import sys
import os
from loadmodel import get_dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorBox(nn.Module):
    pass




#path will be (train, test, val) val is optional
def main(path, type="train"):
    #Aqui, vamos tentar pegar só os 00 - parade
    x = 1

if __name__ == "__main__":
    #dir path that contains the dataset, with the real names
    path = sys.argv[1]
    main(path)



        


