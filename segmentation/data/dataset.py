from torch.utils.data import DataLoader
from itertools import product
import torch
import numpy as np
from .preprocess import Warp, Flip, Greyscale, Affinity
from.superaug import BlurAugment, MisalignAugment, MissingAugment
from tqdm import tqdm

default_dst = list()
default_dst.append((0,0,1))
default_dst.append((0,1,0))
default_dst.append((1,0,0))
default_dst.append((0,0,3))
default_dst.append((0,3,0))
default_dst.append((2,0,0))
default_dst.append((0,0,9))
default_dst.append((0,9,0))
default_dst.append((3,0,0))
default_dst.append((0,0,27))
default_dst.append((0,27,0))
default_dst.append((4,0,0))

def make_gaussian(z, y, x, sigma=0.5):
    y, z, x = np.meshgrid(np.linspace(-1,1,y), np.linspace(-1,1,z), np.linspace(-1,1,x))
    d = np.sqrt(z*z+y*y+x*x)
    return np.exp(-(d**2 / (2.0 * sigma**2)))

class DataSet(object):
    def __init__(self, fov, img, lbl, coords=None, sparse=True, mode="train", 
                 w=True, g=True, f=True, ma=False, mis=False, blr=False, 
                 dst=default_dst, nsamp=None):
        self.img = img
        self.lbl = np.ones_like(img) if lbl is None else lbl
        
        self.fov   = fov
        self.dst   = dst
        self.size  = self.img.shape
        self.range = np.max(dst, 0)        
        self._w = w
        self._f = f
        self._g = g
        self._ma = ma
        self._mis = mis
        self._blr = blr
        self.mode = mode
        self.nsamp = nsamp if nsamp else int(10**10)
        
        self.w = Warp()
        self.f = Flip()
        self.g = Greyscale()
        self.blr = BlurAugment()
        self.mis = MissingAugment()
        self.ma = MisalignAugment()
        self.a = Affinity(dst=self.dst, recompute=False, sparse=sparse)
        
        
    def __getitem__(self, _):
        return self.sample()
    
    def __len__(self):
        return self.nsamp
        
    def sample(self):
        img, lbl = self._sample()
        img, lbl = self.aug(img, lbl)
        aff, msk = self.a(lbl)
        return img, aff, msk
        
    def aug(self, img, lbl):
        if self.mode=="train":
            #print("First", img.shape)
            img, lbl = self.ma.augment(img,lbl) if self._ma else (img[None,:], lbl[None,:])
            #print("Miss alignement", img.shape)
            img, lbl = self.w(img,lbl) if self._w else (img[None,:], lbl[None,:])
            #print("Wrap", img.shape)
            img, lbl = self.f(img,lbl) if self._f else (img, lbl)
            #print("Flip", img.shape)
            img = self.blr.augment(img) if self._blr else img
            #print("Blur", img.shape)
            img = self.g(img) if self._g else img
            #print("Gray", img.shape)
            img = self.mis.augment(img) if self._mis else img
            #print("Missing section", img.shape)
            #print(img.shape)
        else:
            img, lbl = img[None,:], lbl[None,:]
        return img, lbl
        
    def _sample(self, fov_=None):
        fov_  = self._prepare_aug() if fov_ is None else fov_
        delta = (fov_ // 2) 
        coord = self.gen_coord(delta)
        return self._slice(coord, delta)
    
    def gen_coord(self, delta):
        mins  = delta
        maxs  = np.array(self.size) - np.array(delta) - 1
        coord = *map(lambda x: np.random.randint(*x), zip(mins, maxs)),
        return np.array(coord)
        
    def _slice(self, coord, delta):
        bbox = self._bbox(coord, delta)
        a = self.img[bbox]
        b = self.lbl[bbox]
        return a, b
    
    def _bbox(self, coord, delta):
        mins = coord - delta
        maxs = coord + delta
        return tuple(map(lambda x: slice(*x), zip(mins, maxs)))
    
    def _prepare_aug(self):
        if self.mode=="train":
            fov_ = self.w.prepare(self.fov) if self._w else fov_
            fov_ = self.ma.prepare(fov_) if self._ma else fov_
            return np.array(fov_)
        else:
            return np.array(self.fov)

class MultiDataset():
    def __init__(self, datasets, weight=None, nsamp=None, mode="train"):
        self.datasets = datasets
        self.nsamp = nsamp if nsamp else int(10**10)
        self.weight = weight
        self.mode=mode
    
    def __getitem__(self, _):
        i = np.random.randint(len(self.datasets))
        ds = self.datasets[i]
        return ds[_]
    
    def __len__(self):
        return self.nsamp 
    
    def sample(self):
        i = np.random.randint(len(self.datasets))
        ds = self.datasets[i]
        return ds.sample()

class DataSetLoader():
    def __init__(self, ds, num_workers=2):
        self.mode = ds.mode
        self.ds   = ds
        self.dl = iter(DataLoader(self.ds, num_workers=num_workers, pin_memory=True))
        
    def sample(self, mode):
        if mode == self.mode:
            return next(self.dl)
        else:
            mode_ = self.mode
            self.ds.mode=mode
            out = self.ds.sample()
            self.ds.mode = mode_
            return map(lambda x:torch.from_numpy(x[None,:]), out)
        
    
    #============ background computation ================================
    def sample_back(self, mode):
        if mode == self.mode:
            return next(self.dl)
        else:
            mode_ = self.mode
            self.ds.mode=mode
            out = self.ds.sample_back()
            self.ds.mode = mode_
            return map(lambda x:torch.from_numpy(x[None,:]), out)
    #=====================================================================
        
        

class Inference():
    def __init__(self, fov, img, out_idx=[0,1,2], stride_ratio=(2,2,2), sigma=.5,device=0):
        self.ds = DataSet(fov=fov, img=img, lbl=None, mode="test")
        fov = np.array(fov)
        self.delta = fov // 2
        self.device = device
        self.stride = np.array(fov//stride_ratio, dtype=int)
        self.out_idx = out_idx
        self.kernel = make_gaussian(*fov, sigma=sigma)
        self.outputs = np.zeros((len(out_idx),)+self.ds.size)
        self.masks = np.zeros((len(out_idx),)+self.ds.size)
        self.locs = self._init_locs()
    
    def segment(self, model):
        model.eval()
        with torch.no_grad():
            for loc in tqdm(self.locs):
                inp = self._get(loc)
                out = self._tonp(torch.sigmoid(model(inp)))
                out = out[self.out_idx]
                self._put(loc, out)
            return self._get_output()
    
    def _init_locs(self):
        mins = self.delta
        maxs = self.ds.size - self.delta
        stride = self.stride
        return list(product(*[list(range(i,j,k))+[j] for i,j,k in zip(mins, maxs, stride)]))
    
    def _get(self, loc):
        img,_ = self.ds._slice(loc, self.delta)
        return self._topy(img)
    
    def _put(self, loc, out):
        self.outputs[self._bbox(loc)]+=out*self.kernel
        self.masks[self._bbox(loc)]+=self.kernel
    
    def _get_output(self):
        return self.outputs/self.masks
    
    def _topy(self, x):
        return torch.from_numpy(x[None,None,:]).cuda(self.device)
    
    def _tonp(self, x):
        return x.squeeze().cpu().detach().numpy()
    
    def _bbox(self, coord):
        mins = coord - self.delta
        maxs = coord + self.delta
        return (slice(None,None,None),) + tuple(map(lambda x: slice(*x), zip(mins, maxs)))