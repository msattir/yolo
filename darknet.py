from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
from torchsummary import summary
import numpy as np



def get_test_input():
  img = cv2.imread("dog-cycle-car.png")
  img = cv2.resize(img, (416,416))  
  img_ = img[:,:,::-1].transpose((2,0,1)) #BGR -> RGB | H X W X C -> C X H X W
  img_ = img_[np.newaxis,:,:,:]/255.0 #Channel at 0 and Normalise
  img_ = torch.from_numpy(img_).float()
  img_ = Variable(img_)
  return img_



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


class Darknet(nn.Module):
  def __init__(self, cfgfile):
     super(Darknet, self).__init__()
     self.blocks = parse_cfg(cfgfile)
     self.net_info, self.module_list = create_modules(self.blocks)
     
  def forward(self, x, CUDA):
     modules = self.blocks[1:]
     outputs = {} #Cache for shortcut layers that need outputs from previous layers
     
     write = 0
     for i, module in enumerate(modules):
        module_type = (module["type"])
        if module_type == "convolutional" or module_type == "upsample":
           x = self.module_list[i](x)
        elif module_type == "route":
           layers = module["layers"] 
           layers = [int(a) for a in layers]

           if (layers[0]) > 0:
              layers[0] = layers[0] - i

           if len(layers) == 1:  #No concatinate
              x = outputs[i + (layers[0])]

           else:
              if (layers[1] > 0):
                 layers[1] = layers[1] - i

              map1 = outputs[i + layers[0]]
              map2 = outputs[i + layers[1]] 
  
              x = torch.cat((map1, map2), 1) #Depthwise concat

        elif module_type == "shortcut":
           from_ = int(module["from"])
           x = outputs[i-1] + outputs[i+from_]
    
        elif module_type == "yolo":
           anchors = self.module_list[i][0].anchors
           #Get input dimensions
           inp_dim = int(self.net_info["height"])
           num_classes = int(module["classes"])

           #Transform
           x = x.data
           x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
           if not write: #First time
              detections = x
              write = 1

           else:
              detections = torch.cat((detections, x), 1)
        outputs[i] = x

     
     return detections     

  def load_weights(self, weightfile):
      
      #Open the weights file
      fp = open(weightfile, "rb")

      #The first 4 values are header information 
      # 1. Major version number
      # 2. Minor Version Number
      # 3. Subversion number 
      # 4. IMages seen 
      header = np.fromfile(fp, dtype = np.int32, count = 5)
      self.header = torch.from_numpy(header)
      self.seen = self.header[3]
      
      #The rest of the values are the weights
      # Let's load them up
      weights = np.fromfile(fp, dtype = np.float32)
      
      ptr = 0
      for i in range(len(self.module_list)):
          module_type = self.blocks[i + 1]["type"]
          
          if module_type == "convolutional":
              model = self.module_list[i]
              try:
                  batch_normalize = int(self.blocks[i+1]["batch_normalize"])
              except:
                  batch_normalize = 0
              
              conv = model[0]
              
              if (batch_normalize):
                  bn = model[1]
                  
                  #Get the number of weights of Batch Norm Layer
                  num_bn_biases = bn.bias.numel()
                  
                  #Load the weights
                  bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                  ptr += num_bn_biases
                  
                  bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                  ptr  += num_bn_biases
                  
                  bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                  ptr  += num_bn_biases
                  
                  bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                  ptr  += num_bn_biases
                  
                  #Cast the loaded weights into dims of model weights. 
                  bn_biases = bn_biases.view_as(bn.bias.data)
                  bn_weights = bn_weights.view_as(bn.weight.data)
                  bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                  bn_running_var = bn_running_var.view_as(bn.running_var)

                  #Copy the data to model
                  bn.bias.data.copy_(bn_biases)
                  bn.weight.data.copy_(bn_weights)
                  bn.running_mean.copy_(bn_running_mean)
                  bn.running_var.copy_(bn_running_var)
              
              else:
                  #Number of biases
                  num_biases = conv.bias.numel()
              
                  #Load the weights
                  conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                  ptr = ptr + num_biases
                  
                  #reshape the loaded weights according to the dims of the model weights
                  conv_biases = conv_biases.view_as(conv.bias.data)
                  
                  #Finally copy the data
                  conv.bias.data.copy_(conv_biases)
                  
                  
              #Let us load the weights for the Convolutional layers
              num_weights = conv.weight.numel()
              
              #Do the same as above for weights
              conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
              ptr = ptr + num_weights

              conv_weights = conv_weights.view_as(conv.weight.data)
              conv.weight.data.copy_(conv_weights)


mod = Darknet("cfg/yolov3.cfg")           
#model = Darknet("cfg/yolov3.cfg")
#model.load_weights("yolov3.weights")
#inp = get_test_input()
#pred = model(inp, 0)
#print (pred.shape)
#blocks = parse_cfg("cfg/yolov3.cfg")
#print(create_modules(blocks))     
