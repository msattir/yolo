from __future__ import division
import cv2
import numpy as np
from utils import letterbox_image2

def y_pred(filter_size, det, max_size, num_classes=1):
  x_dir, y_dir = filter_size
  jump_cell = np.zeros(2)
  jump_cell[0] = max_size[0]/filter_size[0]
  jump_cell[1] = max_size[1]/filter_size[1]
  predictions = np.zeros([filter_size[0]*filter_size[1], 5+num_classes])
  print (predictions.shape)
  p = 0
  for i in range(1,filter_size[0]):
     for j in range(1,filter_size[1]):
        a1 = det[:,0]-jump_cell[0]*i >= 0
        a2 = det[:,0]-jump_cell[0]*i < jump_cell[0]
        b1 = det[:,1]-jump_cell[1]*j >= 0
        b2 = det[:,1]-jump_cell[1]*j < jump_cell[1]
        if any((a1*a2)*(b1*b2)):
           loc_x = np.where(a1*a2)[0]
           loc_y = np.where(b1*b2)[0]
           loc = list(set(loc_x) & set(loc_y))
           print (i,j,'---', (det[loc[0],0]-(jump_cell[0]*i))/jump_cell[0], (det[loc[0],1]-(jump_cell[1]*j))/jump_cell[1],det[loc[0],2], det[loc[0],3], int(det[loc[0],4]==1), int(det[loc[0],4]==2), det[loc[0],:]) #Doesn't handle 2 boxes in same cell
           cx = (det[loc[0],0]-(jump_cell[0]*i))/jump_cell[0]
           cy = (det[loc[0],1]-(jump_cell[1]*j))/jump_cell[1]
           class_prob = np.eye(num_classes)[det[loc[0],4]-1]
           predictions[p,:] = np.concatenate((np.array([1, cx, cy, det[loc[0],2], det[loc[0],3]]), class_prob))
           #print (predictions[p,:])
     p=p+1
  return predictions 

def gt_pred(img, filt):
  img = cv2.imread('datasets/I1_2009_09_08_drive_0004_000951.png')
  det = np.genfromtxt('datasets/my_csv.csv', delimiter=',')
  
  canvas_shape = []
  img4, canvas_shape  = letterbox_image2(img,[416, 416])
  
  img2 = img4.astype(np.uint8)
  
  x_fact = canvas_shape[0]/img.shape[0]
  y_fact = canvas_shape[1]/img.shape[1]
  
  det[:,0] = det[:,0]*y_fact
  det[:,2] = det[:,2]*y_fact
  det[:,1] = det[:,1]*x_fact+int((416-canvas_shape[0])/2)
  det[:,3] = det[:,3]*x_fact
  
  det2 = det.copy()
  
  det[:,2] = det[:,0]+det[:,2]
  det[:,3] = det[:,1]+det[:,3]
  
  
  det = det.astype(int)
  for i in range(0,det.shape[0]):
    cv2.rectangle(img2, (det[i,0],det[i,1]), (det[i,2],det[i,3]), (0, 255, 0), 2)
  
  
  det2[:,0] = det2[:,0]+det2[:,2]/2
  det2[:,1] = det2[:,1]+det2[:,3]/2
  
  det2 = det2.astype(int)
  for i in range(0,det.shape[0]):
    cv2.circle(img2, (det2[i,0],det2[i,1]), 1, (255, 0, 0), 2)
  
  pred=y_pred(filt, det2, [416, 416], 2)
  return (pred)
#cv2.imshow('image', img2)
#cv2.waitKey(0)
#cv2.imshow('image',img)


