import tifffile, h5py
import numpy as np
from .preprocess import check_volume, affinitize, affinitize_mask
from .dataset import default_dst, DataSetLoader, DataSet

def open_dataset(fov=(32, 160, 160), root="/home/neuro/workspace/data/Igakubu/extracted/", split="A1", filt=None, select=None):
    """
    """
    assert not (filt and select)
    anno = h5py.File(os.path.join(root, "{}/anno.h5".format(split)))["data"]
    em   = h5py.File(os.path.join(root, "{}/em.h5".format(split)))["data"]
    coord= np.load(os.path.join(root, "{}/coords.npy".format(split))).T
    idx = np.load(os.path.join(root, "{}/idx.npy".format(split)))
    
    if filt:
        msk = ~np.isin(idx, filt)
        coord = coord[msk]
    if select:
        msk = np.isin(idx, select)
        coord = coord[msk]
        
    return DataSet(fov, em, anno, coord)

def get_multi_dataset(fov=(32, 160, 160), splits=["A1", "A2", "A3"], filt=None, select=None):
    """
    """
    ds = [open_dataset(fov, split, filt=filt, select=select) for split in splits]
    return  MultiDataset(ds)

def get_Igakubu_dataset(fov=(32, 160, 160), splits=["A1", "A2", "A3"], test=[1] ):
    """
    """
    ds_train = get_multi_dataset(fov, splits, filt=test)
    ds_test  = get_multi_dataset(fov, splits, select=test)
    return ds_train, ds_test

def get_snemid_half_dataset(root, fov):
    """
    """
    img  = tifffile.imread(root+"train-input.tif")
    lbl  = tifffile.imread(root+"train-labels.tif")
    val_img, train_img = img[:,:,:200], img[:,:,200:]
    val_lbl, train_lbl = lbl[:,:,:200], lbl[:,:,200:]
    
    train_img_1, train_img_2 = train_img[::2], train_img[1::2]
    train_lbl_1, train_lbl_2 = train_lbl[::2], train_lbl[1::2]
    train_dataset_1 = DataSet(fov, train_img_1, train_lbl_1)
    train_dataset_2 = DataSet(fov, train_img_2, train_lbl_2)
    train_dataset = MultiDataset([train_dataset_1, train_dataset_2], mode="train")
    
    val_img_1, val_img_2 = val_img[::2], val_img[1::2]
    val_lbl_1, val_lbl_2 = val_lbl[::2], val_lbl[1::2]
    
    val_dataset_1 = DataSet(fov, val_img_1, val_lbl_1, mode="test")
    val_dataset_2 = DataSet(fov, val_img_2, val_lbl_2, mode="test")
    val_dataset = MultiDataset([val_dataset_1, val_dataset_2], mode="test")
    
    return train_dataset, val_dataset

def get_snemid_dataset(root, fov, num_workers, sparse=False):
    """
    """
    img  = tifffile.imread(root+"train-input.tif")
    lbl  = tifffile.imread(root+"train-labels.tif")
    val_img, train_img = img[:,:200,:200], img[:,:,200:]
    val_lbl, train_lbl = lbl[:,:200,:200], lbl[:,:,200:]
    
    train_dataset = DataSetLoader(DataSet(fov, train_img, train_lbl, sparse=sparse),num_workers)
    val_dataset   = DataSetLoader(DataSet(fov, val_img, val_lbl, sparse=sparse, mode="test"),num_workers)

    return train_dataset, val_dataset


def get_sparse_snemid_dataset(root, fov, sparse_rate, num_workers):
    """
    """
    img  = tifffile.imread(root+"train-input.tif")
    #lbl  = tifffile.imread(root+"train-labels.tif")
    val_img, train_img = img[:,:200,:200], img[:,:,200:]
    #val_lbl, train_lbl = lbl[:,:200,:200], lbl[:,:,200:]
    val_lbl = tifffile.imread(root+"sparse/train-labels-sparse_" + str(sparse_rate) + "_test.tif")
    train_lbl = tifffile.imread(root+"sparse/train-labels-sparse_" + str(sparse_rate) + "_train.tif")
    
    train_dataset = DataSetLoader(DataSet(fov, train_img, train_lbl),num_workers)
    val_dataset   = DataSetLoader(DataSet(fov, val_img, val_lbl, mode="test"),num_workers)

    return train_dataset, val_dataset


def get_igaku_dendrite_dataset(root,fov,num_workers,dst,sparse=False):
    img  = tifffile.imread(root+"train-allImages.tif")
    lbl  = tifffile.imread(root+"train-allLabels.tif")
    val_img, train_img = img[:,:,:150], img[:,:,150:]
    val_lbl, train_lbl = lbl[:,:,:150], lbl[:,:,150:]
    
    train_dataset = DataSetLoader(DataSet(fov, train_img, train_lbl, dst=dst, sparse=sparse),num_workers)
    val_dataset   = DataSetLoader(DataSet(fov, val_img, val_lbl, dst=dst, sparse=sparse, mode="test"),num_workers)

    return train_dataset, val_dataset


def get_igaku_background_dataset(root,fov,num_workers,dst,sparse=False):
    img  = tifffile.imread(root+"train-allImages.tif")
    lbl  = tifffile.imread(root+"train-allBackgrounds.tif")
    val_img, train_img = img[:,:,:150], img[:,:,150:]
    val_lbl, train_lbl = lbl[:,:,:150], lbl[:,:,150:]
    
    train_dataset = DataSetLoader(DataSet(fov, train_img, train_lbl, dst=dst, sparse=sparse),num_workers)
    val_dataset   = DataSetLoader(DataSet(fov, val_img, val_lbl, dst=dst, sparse=sparse, mode="test"),num_workers)

    return train_dataset, val_dataset


def get_snemi3d_adaptedIgaku_dataset(root,fov,num_workers,dst,sparse=False):
    img  = tifffile.imread(root+"train-input.tif")
    lbl  = tifffile.imread(root+"train-labels.tif")
    val_img, train_img = img[:,:,:150], img[:,:,150:]
    val_lbl, train_lbl = lbl[:,:,:150], lbl[:,:,150:]
    
    train_dataset = DataSetLoader(DataSet(fov, train_img, train_lbl, dst=dst, sparse=sparse),num_workers)
    val_dataset   = DataSetLoader(DataSet(fov, val_img, val_lbl, dst=dst, sparse=sparse, mode="test"),num_workers)

    return train_dataset, val_dataset
