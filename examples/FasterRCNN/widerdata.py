# -*- coding: utf-8 -*-
# File: widerdata.py

import os
import cv2
import numpy as np
import json
MINSIZE = 9
class WiderData(object):

    def __init__(self, basedir):
        self.basedir = basedir
        self.anndir = os.path.join(basedir,'wider_face_split')



    def loadImgs(self,data='train',fromfile=False):
        print('load WIDER data info')
        labeldir = os.path.join(self.anndir,'wider_face_{}_bbx_gt.txt'.format(data))
        baseimgdir = os.path.join(self.basedir,'WIDER_{}/images'.format(data))
        labelfile = open(labeldir)
        lines = labelfile.readlines()
        lindex = 0
        imgs = []
        hwarray=[]
        while(lindex<len(lines)):
            img = {}
            name = lines[lindex][:-1]
            imgname = os.path.join(baseimgdir,name)
            img['file_name'] = imgname
            lindex += 1
            objnum = int(lines[lindex][:-1])
            lindex += 1
            boxes = []
            delnum = 0
            for i in range(objnum):
                labels = lines[lindex+i][:-1].split(' ')
                box = [float(labels[0]), float(labels[1]),float(labels[2])+float(labels[0]),float(labels[3])  + float(labels[1])]
                if float(labels[2])*float(labels[3])>MINSIZE:
                    boxes.append(box)
                else:
                    delnum+=1
            lindex += objnum
            
            objnum = objnum - delnum
            img['boxes'] = np.float32(np.asarray(boxes))
            img['is_crowd'] = np.asarray([0]*objnum)
            img['class'] = np.asarray([1]*objnum)
            if fromfile:
                hws = np.load('wider_wh_{}.npy'.format(data))
                
                
                img['height'] = hws[len(imgs)][0]
                img['width'] = hws[len(imgs)][1]
            else:
                imgdata = cv2.imread(imgname)
                img['height'] = imgdata.shape[0]
                img['width'] = imgdata.shape[1]
                hwarray.append([imgdata.shape[0],imgdata.shape[1]])
            imgs.append(img)
        if fromfile==False:
            np.save('wider_wh_{}.npy'.format(data),hwarray)
            #res.write(str(hwdict))
            #res.close()
        return imgs

    def loadImgsfromfile(self,data='train'):
        labelfile = open('wider_label_{}.json'.format(data),'r')
        return json.load(labelfile)  
            
        
 
