from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfg_file):
  """
  Take a cfg txt file
  
  Return dicts for each block
  """
  file = open(cfg_file,'r')
  lines = file.read().split('\n')                #List
  lines = [x for x in lines if len(x) > 0]       #Remove empty lines
  lines = [x for x in lines if x[0] != '#']      #Remove comments
  lines = [x.rstrip().lstrip() for x in lines]   #remove white spaces

  block = {}
  blocks = []

  for line in lines:
     if line[0] == '[':  #A new block starts
        if len(block) != 0:
           blocks.append(block)
           block = {}
        block['type'] = line[1:-1].rstrip()
     else:
        key,value = line.split('=')
        block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return(blocks)  

def create_modules(blocks):
  """
  Takes in a list of blocks
  """
  net_info = blocks[0]  #Captures in
  module_list = nn.ModuleList()  #return this object
  prev_filters = 3  #Track filter depth - initialize to 3 (RGB)
  output_filters = []  #Tracks all previous output filter dimensions
  
  for index, x in enumerate(blocks[1:]):
     module = nn.Sequential()
     
     #CNN Layer
     if(x["type"] == "convolutional"):
        activation = x["activation"]
        try:
           batch_normalize = int(x["batch_normalize"])
           bias = False
        except:
           batch_normalize = 0
           bias = True

        filters = int(x["filters"])
        padding = int(x["pad"])
        kernel_size = int(x["size"])
        stride = int(x["stride"])
        
        if padding:
           pad = (kernel_size - 1) // 2
        else:
          pad = 0

        #Add cnn layer
        conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding=pad, bias=bias)
        module.add_module("conv_{0}".format(index), conv)

        #Batch Norm Layer
        if batch_normalize:
           bn = nn.BatchNorm2d(filters)
           module.add_module("batch_norm_{0}".format(index), bn)
        
        #Activation Layer
        if activation == "leaky":
           actn = nn.LeakyReLU(0.1, inplace = True)
           module.add_module("leaky_{0}".format(index), actn)

     #Upsampling Layer
     #Biilinear 2d upsample
     elif(x["type"] == "upsample"):
        stride =  int(x["stride"])
        upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
        module.add_module("upsample_{}".format(index), upsample)

     #residual layer
     elif(x["type"] == "route"):
        x["layers"] = x["layers"].split(',')
        #start a route
        start = int(x["layers"][0])
        #search for end, if exists
        try:
           end = int(x["layers"][1])
        except:
           end = 0
        if start > 0:
           start = start - index
        if end > 0:
           end = end - index
        route = EmptyLayer()
        module.add_module("route_{0}".format(index), route)
        if end < 0:
           filters = output_filters[index + start] + output_filters[index + end]
        else:
           filters = output_filters[index + start]
       
     #skip connections
     elif(x["type"] == "shortcut"):
        shortcut = EmptyLayer()
        module.add_module("shortcut_{}".format(index), shortcut)

     #Yolo detection layer
     elif(x["type"] == "yolo"):
        mask = (x["mask"]).split(',')
        mask = [int(x) for x in mask]
        
        anchors = x["anchors"].split(',')
        anchors = [int(x) for x in anchors]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in mask]

        detection = DetectionLayer(anchors)
        module.add_module("Detection_{}".format(index), detection)

     module_list.append(module)
     prev_filters = filters
     output_filters.append(filters)
  
  return(net_info, module_list)

class EmptyLayer(nn.Module):
  def __init__(self): 
     super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
  def __init__(self, anchors):
     super(DetectionLayer, self).__init__()
     self.anchors = anchors
        
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))     
