import argparse
import torch
from torch.nn import functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
import shutil
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import pickle


from werkzeug.sansio.multipart import File
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import InterpolationMode

from PatchCoreDataset import PatchCoreDataset

from time import time

from pytorch_lightning.callbacks.progress import ProgressBar
from tqdm import tqdm
import sys

def distance_matrix(x:torch.Tensor, y:torch.Tensor=None, p=2)->torch.Tensor:  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)

    # dist = torch.pow(x - y, p).sum(2)

    dist = x.unsqueeze(1).expand(n, m, d) - y.unsqueeze(0).expand(n, m, d)
    # dist = torch.pow(dist, p).sum(2)
    # dist = dist.pow(p).sum(2)
    torch.pow(dist, p, out=dist)
    dist = dist.sum(2)
    torch.pow(dist, 1/p, out=dist)

    return dist

"""Image write function from opencv-api.
It have a error with korean characters.
url=https://jangjy.tistory.com/337"""
def imwrite( filename, img, params=None):
    try:
        ext = os.path.splitext( filename )[1]
        res, binary_code = cv2.imencode( ext, img, params )

        if res:
            with open( filename, mode="w+b" ) as f:
                binary_code.tofile( f )
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        # dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        dist = distance_matrix(x, self.train_pts, self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):


        # dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        dist = distance_matrix(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)


        return knn

class LitProgressBar(ProgressBar):

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,  # This two lines are only for pycharm
            ncols=100,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout
        )
        return bar

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2, device=x.device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt).convert("L")
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type



def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)
    

def merge_with_rois( data : list, rois : list ) -> np.ndarray:
    """merge data with rois.
    Region can be overlapped or non-overlapped.

    Args:
    ------
     - `data` : data patch from roi. (list or ndarray)
     - `rois` : roi table. (list or ndarray)

    Return :
     - `merged data` : merged data.
    """
    if not isinstance(rois, list):
        print( "rois is not list!" )
        rois = list( rois )
    if not isinstance(data, list):
        print( "data is not list!" )
        data = list( data )

    assert len( data ) > 0, "length of data is zero!"
    assert len( data ) == len(rois), "length of data and rois are different!"
    # assert data.shape[0] == rois.shape[0], \
    #     "length of data and rois are different!"

    rois_max = np.stack( rois, axis=0 ).max(axis=0)

    fullimage_width = rois_max[2]
    fullimage_height = rois_max[3]

    # resize roi 
    H, W = data[0].shape[-2], data[0].shape[-1]
    rsx = W * args.M / fullimage_width
    rsy = H * args.N / fullimage_height
    resized_rois = [(int(roi[0]*rsx), int(roi[1]*rsy),
                     int(roi[2]*rsx), int(roi[3]*rsy)) for roi in rois]
    rois_max = np.stack( resized_rois, axis=0 ).max(axis=0)
    resized_width = rois_max[2]
    resized_height = rois_max[3]

    merged_data = []
    masks = []
    for datum, roi in zip( data, resized_rois):
        l, t, r, b = roi
        extend_data = cv2.resize( np.zeros_like(datum), (resized_width, resized_height) )
        mask = np.zeros( (resized_height, resized_width), dtype=np.uint32 )
        extend_data[ t:b, l:r ] = cv2.resize( datum, (r-l, b-t) )
        mask[ t:b, l:r ] = 1
        merged_data.append( extend_data )
        masks.append( mask )

    merged_data = np.stack( merged_data, axis=0 ).sum(axis=0)
    masks = np.stack( masks, axis=0 ).sum(axis=0)
    if merged_data.ndim == 3 and masks.ndim == 2:
        masks = np.reshape( masks, (masks.shape[0], masks.shape[1], 1) )
    merged_data = np.true_divide( merged_data, masks, where=masks!=0 )

    merged_data = cv2.resize( merged_data,
                              (args.input_size*args.M, args.input_size*args.N) )

    return merged_data


def concatenate_with_rois( data, rois ):
    """merge data with rois.
    Region must be non-overlapped.
    Just concatenate data with rois.

    Args:
    ------
     - `data` : data patch from roi. (list or ndarray)
     - `rois` : roi table. (list or ndarray)

    Return :
     - `merged data` : merged data."""
    if isinstance(rois, list):
        rois = np.asarray( rois )
    if isinstance(data, list):
        data = np.asarray( data )

    assert data.shape[0] > 0, "length of data is zero!"
    # assert len( data ) == len(rois), "length of data and rois are different!"
    assert data.shape[0] == rois.shape[0], \
        "length of data and rois are different!"

    x = -1
    cols = []
    while np.any( rois[:,0] > x ):
        cols.append(rois[:,0].min())

class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        # self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape[0] != input_img.shape[0] and \
           anomaly_map.shape[1] != input_img.shape[1]:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        if gt_img is not None:
            imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        # image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        # image_datasets = GloTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        image_datasets = PatchCoreDataset(
            data_path_root=args.dataset_path, gt_path_root=args.gt_path,
            transform=self.data_transforms, gt_transform=self.gt_transforms,
            phase='train', M=args.M, N=args.N)
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        # test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        # test_datasets = GloTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_datasets = PatchCoreDataset(
            data_path_root=args.dataset_path, gt_path_root=args.gt_path,
            transform=self.data_transforms, gt_transform=self.gt_transforms,
            phase='test', M=args.M, N=args.N, recursive=args.recursive_search)
        test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        self.img_fullpath_list = test_datasets.img_paths
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []
        self.x_list = []
        self.roi_list = []
        self.filename_list = []
    
    def on_test_start(self):
        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.prev_filename = ""
        self.inputs, self.exmaps, self.rois = [], [], []
        self.times = { "detection":[],
                       "cnn":[],
                       "embedding":[],
                       "knn":[],
                       "merge":[],
                       "reshape":[],
                       "score":[],
                       "gaussian":[] }
                       
        self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
        self.knn = KNN( torch.from_numpy(self.embedding_coreset).cuda(), 
                        k=args.n_neighbors )
        self.initialize_buffer()
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, file_name, rois = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        # embedding = embedding_concat(embeddings[1], embeddings[2])
        # embedding = embedding_concat(embeddings[0], embedding)
        self.embedding_list.extend(reshape_embedding(np.array(embedding.cpu().detach())))
        
        # Save roi and filename list.
        # Roi and filename list will use for display selected coreset samples.
        # embedding = (B, D, H, W), filename = (B, 1), rois = (B, 4)
        # self.gathering_roi_and_filename( rois, file_name )
        
    def gathering_roi_and_filename(self, rois, filenames):
        """Gather roi and filename and save it.
        Roi and filename will used for display selected coreset samples.
        
        Args :
        ------
        rois : list of Tensor( shape=(B,4) )
            data patch roi list. It may be tensor because we use dataloader.
        filename : list( len=B )
            data patch filename list.

        Targets :
        --------
        embedding = (B,D,H,W) --> embedding_list = (BxHxW,D)
        H, W = 28, 28
        """
        rois = [ list(i.cpu().detach().numpy()) for i in rois ]
        rois = np.asarray( rois ).transpose()

        roilist = []
        B = rois.shape[0]
        for i, roi in enumerate( rois ):
            l, t, r, b = roi
            L, T = np.meshgrid( np.linspace(l, r, 28, endpoint=False),
                                np.linspace(t, b, 28, endpoint=False) )
            R, B = np.meshgrid( np.sort(np.linspace(r, l, 28, endpoint=False)),
                                np.sort(np.linspace(b, t, 28, endpoint=False)) )
            XY = np.stack((L, T, R, B)).transpose(1,2,0)
            roilist.append( XY )

        roilist = reshape_embedding( np.array(roilist).transpose(0,3,1,2) )
        self.roi_list.extend( roilist )

        filename_list = []
        for filename in filenames:
            filename_list.extend( [filename]*(28*28) )
        
        self.filename_list.extend( filename_list )

    
    def save_coreset_list( self, indexes ):
        """Save coreset list by filename_list and roi_list
        I don't have file path, so I couldn't save by image."""
        filepath = os.path.join(self.sample_path, "coreset_samples.csv")
        with open( filepath, "w" ) as f:
            for idx in indexes:
                filename = self.filename_list[idx]
                roi = self.roi_list[idx]
                f.write(f"{filename},{roi[0]},{roi[1]},{roi[2]},{roi[3]}\n")



    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        if args.coreset_sampling_ratio > 1:
            N = int(args.coreset_sampling_ratio)
        else:
            N = int(total_embeddings.shape[0]*args.coreset_sampling_ratio)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=N)
        self.embedding_coreset = total_embeddings[selected_idx]

        # self.save_coreset_list( selected_idx )
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
            pickle.dump(self.embedding_coreset, f)

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x_, gt_, label_, file_name_, roi_ = batch
        roi_ = [ list(i.cpu().detach().numpy()) for i in roi_ ]
        roi_ = np.asarray( roi_ ).transpose()

        start = time()
        # extract embedding
        features = self(x_)
        duration = (time() - start) * 1000
        self.tmp_cnn_times.append(duration)
        
        start = time()
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        # embedding = embedding_concat(embeddings[1], embeddings[2])
        # embedding = embedding_concat(embeddings[0], embedding)
        duration = (time() - start) * 1000
        self.tmp_embed_times.append(duration)

        B, D, H, W = embedding.shape
        x_ = x_[:,:,2:H-2,2:W-2]
        embedding = embedding[:,:,2:H-2,2:W-2]

        start = time()
        B, D, H, W = embedding.shape
        embedding_test = np.array(
            reshape_embedding(np.array(embedding.cpu().detach())))
        # embedding_test = embedding.reshape( -1, D )
        duration = (time() - start) * 1000
        self.tmp_reshape_times.append(duration)

        # NN
        #nbrs = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
        #score_patches, _ = nbrs.kneighbors(embedding_test)
        #
        #Approximately 60x performance improvement
        #tack time 1.9070019721984863 -> 0.03699636459350586
        start = time()
        K = self.knn.k
        score_patches = self.knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy().reshape((B,-1,K))
        # score_patches = self.knn(embedding_test)[0].reshape(B,-1,K).cpu().detach().numpy()
        # record times. (unit:micro seconds)
        duration = (time() - start) * 1000 
        self.tmp_knn_times.append( duration )


        for score_map, label, file_name, roi in zip( score_patches, label_, file_name_, roi_):
            # Because previous image patch is finished and new image incomes.
            if len( self.prev_filename ) != 0 and \
               file_name != self.prev_filename:
               self.end_of_one_image()
               self.initialize_buffer(next_filename=file_name, next_label=label)
            elif file_name != self.prev_filename:
               self.initialize_buffer(next_filename=file_name, next_label=label)

            start = time()
            anomaly_map = score_map[:,0]
            N_b = score_map[np.argmax(anomaly_map)]
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            score = w*max(anomaly_map) # Image-level score
            if self.input_max_score < score:
                self.input_max_score = score
            duration = (time() - start) * 1000
            self.tmp_score_times.append( duration )

            # save to buffer.
            self.exmaps.append( w * anomaly_map.reshape(H,W) )
            self.rois.append( roi )
            # input_x = self.inv_normalize(x).cpu().detach().numpy()
            # input_x = cv2.cvtColor( input_x.transpose(1,2,0)*255,
            #                         cv2.COLOR_RGB2BGR )
            # self.inputs.append( input_x )
        
        # gt_np = gt.cpu().numpy()[0,0].astype(int)
        # anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        # anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        
        # self.gt_list_px_lvl.extend(gt_np.ravel())
        # self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        # self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        # self.pred_list_img_lvl.append(score)
        # self.img_path_list.extend(file_name)
        # # save images
        # x = self.inv_normalize(x)
        # input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        # self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def end_of_one_image(self):
        """When we use grid detection, we need merging process when one image
        end."""
        start = time()
        anomaly_map_resized = merge_with_rois( self.exmaps, self.rois )
        # input_img = merge_with_rois( self.inputs, self.rois )
        duration = (time() - start) * 1000
        self.times["merge"].append( duration )

        # save result.
        # gt_np = gt.cpu().numpy()[0].astype(int)
        start = time()
        #anomaly_map_resized = cv2.resize(exmaps, (img_h, img_w))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        duration = (time() - start) * 1000
        self.times["gaussian"].append( duration )

        # record times. (unit:micro seconds)
        duration = (time() - self.prev_time) * 1000
        self.times["detection"].append( duration )
        mean_time = sum(self.tmp_cnn_times) #/ \
        #    (args.batch_size*len(self.tmp_cnn_times))
        self.times["cnn"].append( mean_time )
        mean_time = sum(self.tmp_embed_times) #/ \
        #    (args.batch_size*len(self.tmp_embed_times))
        self.times["embedding"].append( mean_time )
        mean_time = sum(self.tmp_knn_times) #/ \
        #    (args.batch_size*len(self.tmp_knn_times))
        self.times["knn"].append( mean_time )
        mean_time = sum(self.tmp_reshape_times) #/ \
        #    (args.batch_size*len(self.tmp_reshape_times))
        self.times["reshape"].append( mean_time )
        mean_time = sum(self.tmp_score_times) #/ \
        #    (args.batch_size*len(self.tmp_score_times))
        self.times["score"].append( mean_time )

        filename_match_list = [ fullpath for fullpath in self.img_fullpath_list
                                            if self.prev_filename in fullpath ]
        # if len( filename_match_list ) == 1:
        #     input_img = np.asarray( Image.open( filename_match_list[0] ) )
        # elif len( filename_match_list ) > 1:
        #     print( "too many matched name! : ", len(filename_match_list) )
        #     print( filename_match_list )
        input_img = np.asarray( Image.open( filename_match_list[0] ) )
        input_img = cv2.resize( input_img, (args.input_size*args.M,
                                            args.input_size*args.N) )

        
        # self.gt_list_px_lvl.extend(gt_np.ravel())
        # self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        # self.gt_list_img_lvl.append(self.input_label)
        # self.pred_list_img_lvl.append(self.input_max_score)
        # self.img_path_list.extend(self.prev_filename)
        # save images
        self.save_anomaly_map(anomaly_map_resized_blur, input_img,
                              np.zeros_like(input_img), self.prev_filename,
                              args.category)


    def initialize_buffer(self, next_filename="", next_label=0):
        """We need buffer for collecting results of each grid."""
        # initialize buffer
        self.prev_filename = next_filename
        self.inputs, self.exmaps, self.rois = [], [], []
        if isinstance(next_label, torch.Tensor):
            self.input_label = int(next_label.cpu().detach().numpy()) 
        else:
            self.input_label = next_label
        self.input_max_score = 0
        self.tmp_knn_times = []
        self.tmp_cnn_times = []
        self.tmp_reshape_times = []
        self.tmp_score_times = []
        self.tmp_embed_times = []
        self.prev_time = time()


    def test_epoch_end(self, outputs):
        if len( self.exmaps ) != 0 :
            self.end_of_one_image()
            self.initialize_buffer()

        # print computation times
        for key, value in self.times.items():
            print( f"{key} times(average) : {sum( value ) / len( value )}")

        return

        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)
        # anomaly_list = []
        # normal_list = []
        # for i in range(len(self.gt_list_img_lvl)):
        #     if self.gt_list_img_lvl[i] == 1:
        #         anomaly_list.append(self.pred_list_img_lvl[i])
        #     else:
        #         normal_list.append(self.pred_list_img_lvl[i])

        # # thresholding
        # # cal_confusion_matrix(self.gt_list_img_lvl, self.pred_list_img_lvl, img_path_list = self.img_path_list, thresh = 0.00097)
        # # print()
        # with open(args.project_root_path + r'/results.txt', 'a') as f:
        #     f.write(args.category + ' : ' + str(values) + '\n')

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'/home/changwoo/hdd/datasets/mvtec_anomaly_detection') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--gt_path', default=None )
    parser.add_argument('--category', default='carpet')
    parser.add_argument('--recursive_search', type=bool, default=False )
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_size', type=int, default=256) # 256
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--grid_size', type=str, default="1x1")
    parser.add_argument('--coreset_sampling_ratio', type=float, default=0.001)
    parser.add_argument('--project_root_path', default=r'/home/changwoo/hdd/project_results/patchcore/test') # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #
    parser.add_argument('--save_src_code', type=bool, default=True)
    parser.add_argument('--save_anomaly_map', type=bool, default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    tmp_list = args.grid_size.split('x')
    if len(tmp_list) != 2:
        print("grid_size has wrong format")
        args.M = 1
        args.N = 1
    else:
        args.M = int(tmp_list[0])
        args.N = int(tmp_list[1])
    return args

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    
    bar = LitProgressBar()
    
    trainer = pl.Trainer.from_argparse_args(args, 
        default_root_dir=os.path.join(args.project_root_path, args.category),
        max_epochs=args.num_epochs, gpus=1, callbacks=[bar]) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = STPM(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        # trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)
