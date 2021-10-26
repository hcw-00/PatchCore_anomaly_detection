
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
from PIL import Image

# for debugging
from time import time

class PatchCoreDataset(Dataset):

    def __init__(self, data_path_root, gt_path_root, transform, gt_transform, phase, M=1, N=1, recursive=False):
        """Dataset for PatchCore.

        Parameters:
        -----------
         - ``data_path`` : (str) image data path for training or testing.
         - ``gt_path`` : (str/None) ground-truth image data path for testing.
         - ``transform`` : (torchvision.transform) transform object for input
            data. Currently, we use resize, totensor, centercrop and normalize.
         - ``gt_transform`` : (torchvision.transform) transform object for 
            ground-truth data. Currently, we use resize, totensor, centorcrop.
         - ``phase`` : (str) phase of PatchCore. It can be `train` or `test`.
         - ``recursive`` : (bool) If the dataset has a folder structure, we need
            recursive search.
        """
        self.phase = phase
        self.img_path_root = data_path_root
        self.gt_path_root = gt_path_root
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.rois = \
            self.load_dataset(recursive) # self.labels => good : 0, anomaly : 1
        
        self.divide_by_grid( M, N )



    def load_dataset(self, recursive) -> tuple[list[str], list[str], list[int], list[tuple[int,int,int,int]]]:
        """Load data file list from data_path and gt_path.
        If ``recursive`` is True, it will full-search in all directories from
        data path. else, search all image files in current data path.
        
        Parameters:
        -----------
         - ``recursive`` : (bool) If the dataset has a folder structure, we need
            recursive search.

        Return:
        -------
         - ``img_paths`` : (list(str))
         - ``gt_paths`` : (list(str)) ground-truth path
         - ``labels`` : (list(int)) 0 is normal, 1 is abnormal.
         - ``rois`` : (list((int,int))) ROI of image.
        """

        img_tot_paths = []
        gt_tot_paths = []
        gt_tmp_paths = []

        if recursive:
            filelist = []
            for (root, _, files) in os.walk(self.img_path_root):
                filelist.extend([os.path.join(root,file) for file in files
                                if file.endswith((".jpg", ".bmp", ".png"))])
        else:
            filelist = [os.path.join(self.img_path_root, file)
                            for file in os.listdir(self.img_path_root)
                            if file.endswith((".jpg", ".bmp", ".png"))]
        img_tot_paths.extend(filelist)

        if self.phase == "test" and self.gt_path_root is not None:
            if recursive:
                filelist = []
                for (root, _, files) in os.walk(self.gt_path_root):
                    filelist.extend([os.path.join(root,file) for file in files
                                    if file.endswith((".jpg", ".bmp", ".png"))])
            else:
                filelist = [os.path.join(self.gt_path_root, file)
                                for file in os.listdir(self.gt_path_root)
                                if os.path.isfile(file) and 
                                    file.endswith((".jpg", ".bmp", ".png"))]
            gt_tmp_paths.extend(filelist)
        
        tot_labels = [0]*len(img_tot_paths)
        gt_tot_paths = [0]*len(img_tot_paths)
        tot_rois = [None]*len(img_tot_paths)
        # gt file name can be "[filename].[ext]" or "[filename]_label.[ext]".
        # What happen when we have a file which named "test1.jpg" and "test11
        # .jpg"? "test1" in "test11.jpg" is True.
        if gt_tmp_paths:
            for i, filepath in enumerate( img_tot_paths ):
                filename = os.path.splitext(os.path.basename( filepath ))[0]
                for gt_filepath in gt_tmp_paths:
                    if filename in gt_filepath:
                        tot_labels[i] = 1
                        gt_tot_paths[i] = gt_filepath
                        break
        
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_rois
    

    def divide_by_grid(self, m, n):
        """Input data will be divided by ``m`` and ``n``.

        Example :
            If input Image size is 4096x2160 and (m,n) is (4,2).
            Image will be divided to 8 pieces.
             Img(0,0) = (0, 0, 1024, 1080), Img(0,1) = (1024, 0, 1024, 1080),
             ..., Img(1,0) = (0, 1080, 1024, 1080), ...,
             Img(1,3) = (3072, 1080, 1024, 1080).
        """
        tmp_img_paths = []
        tmp_gt_paths = []
        tmp_labels = []
        tmp_rois = []
        for i in range( len(self.img_paths) ):
            filepath = self.img_paths[i]
            gt_filepath = self.gt_paths[i]
            label = self.labels[i]
            img = Image.open(filepath)
            step_y = img.height / n
            step_x = img.width / m
            for r in range(n):
                y = int(r * step_y)
                y_next = int( (r+1)*step_y )
                for c in range(m):
                    x = int(c*step_x)
                    x_next = int( (c+1)*step_x )
                    tmp_img_paths.append( filepath )
                    tmp_gt_paths.append( gt_filepath )
                    tmp_labels.append( label )
                    tmp_rois.append( (x, y, x_next, y_next ) )
        
        self.img_paths = tmp_img_paths
        self.gt_paths = tmp_gt_paths
        self.labels = tmp_labels
        self.rois = tmp_rois

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        #start = time()
        img_path, gt, label, roi = \
            self.img_paths[idx], self.gt_paths[idx], \
            self.labels[idx], self.rois[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]

        dx = (roi[2] - roi[0]) / 12
        dy = (roi[3] - roi[1]) / 12
        newroi = (roi[0] - dx, roi[1] - dy, roi[2] + dx, roi[3] + dy)

        img = Image.open(img_path).convert('RGB')
        img = img.crop(newroi) #if roi is not None else img
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt).convert("L")
            gt = gt.crop(roi) #if roi is not None else gt
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        #end = time()
        #print( f"getitem : {(end-start)*1000} ms" )

        return img, gt, label, filename, roi if roi is not None else 0


# class GloTecDataset(Dataset):
#     """Glotec dataset.
#     It is saved with ``bmp`` format. and There is no any ground-truth."""
#     def __init__(self, root, transform, gt_transform, phase):
#         self.img_path = root
#         self.phase = phase
#         self.transform = transform
#         # self.gt_transform = gt_transform
#         # load dataset
#         # self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1
#         self.img_paths, self.labels = self.load_dataset() # self.labels => good : 0, anomaly : 1

#     def load_dataset(self):
#         img_tot_paths = []

#         img_paths = glob.glob( self.img_path + "/*.bmp" )
#         img_tot_paths.extend(img_paths)
#         if self.phase=="train":
#             labels = [0] * len( img_tot_paths )
#         elif self.phase=="test":
#             labels = [1] * len( img_tot_paths )
#         else:
#             print(f"phase error : phase is {self.phase}!")

#         return img_tot_paths, labels

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         # img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
#         img_path, label = self.img_paths[idx], self.labels[idx]
#         img = Image.open(img_path).convert('RGB')
#         img = self.transform(img)
#         # if gt == 0:
#         #     gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
#         # else:
#         #     gt = Image.open(gt)
#         #     gt = self.gt_transform(gt)
        
#         # assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

#         # return img, gt, label, os.path.basename(img_path[:-4]), img_type
#         return img, None, label, os.path.basename(img_path[:-4]), self.phase

