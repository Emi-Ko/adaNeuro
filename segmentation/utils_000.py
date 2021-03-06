from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from .data.dataset import Inference
from .cython_wathed import segment
from gala import agglo

flatten = lambda x: [y for z in x for y in z]

def pad_size(ks, mode):
    assert mode in ['valid', 'same', 'full']
    if mode == 'valid':
        pad = (0,0,0)
    elif mode == 'same':
        assert all([x %  2 for x in ks])
        pad = tuple(x // 2 for x in ks)
    elif mode == 'full':
        pad = tuple(x - 1 for x in ks)
    return pad

def layers_mod(model):
    return list(set(map(lambda x:type(x).__name__, model.modules())))#__name__

def params_mod(model):
    return list(model.parameters())

def params_size_mod(model):
    return np.sum(list(map(lambda x:np.prod(x.size()), model.parameters())))
    
def select_mod(model, modules):
    modules = list(map(lambda x: x if isinstance(x, str) else x.__name__, modules))
    return list(filter(lambda x:type(x).__name__ in modules, model.modules()))#
    
def distrib_params_per_mod(model, rel=False):
    res = {}
    for mod in layers_mod(model):
        layers = select_mod(model, [mod])
        count = np.sum(list(map(lambda x:params_size_mod(x), layers)))
        res[mod]=count
    if rel:
        nparams=params_size_mod(model)
        res = {k:v/nparams for k,v in res.items()}
    return res

def distrib_mod(model):
    res = {}
    for mod in layers_mod(model):
        #print(mod)
        layers = select_mod(model, [mod])
        res[mod]=len(layers)
    return res

class Monitor():
    def __init__(self, logdir, msk=True, acc=True, loss=True, vi=True, rand=True):
        """
        """
        self.writer = SummaryWriter(logdir)
        self.loss = loss
        self.acc = acc
        self.msk = msk
        self.vi = vi
        self.rand = rand
    
    def log_train(self, i, loss, msk_stat):
        """
        """
        self.writer.add_scalar("training_loss", loss, i)
        mean_msk = 0
        if self.msk:
            for i in range(msk_stat.shape[0]):
                raise NotImplementedError()
    
    def log_val(self, i, key, loss_stats, acc_stats, msk_stat):
        """
        """
        mean_acc  = acc_stats.mean()
        self.writer.add_scalar("valid_accuracy_{}".format(key), mean_acc, i)
        mean_loss = loss_stats.mean()
        self.writer.add_scalar("valid_loss_{}".format(key), mean_loss, i)

        if self.msk:
            for i in range(msk_stat.shape[0]):
                raise NotImplementedError()
        if self.acc:
            for i in range(acc_stats.shape[0]):
                raise NotImplementedError()
        if self.loss:
            for i in range(loss_stats.shape[0]):
                raise NotImplementedError()
                
def valid(model, dataset, loss_fn, device=0, nrun=20):
    """
    """
    model.eval()
    with torch.no_grad():
        losses, accs, msks = [],[],[]
        for i in range(nrun):
            x, y, msk = dataset.sample("valid")
            x, y, msk = map(lambda x:x.cuda(device), [x, y, msk])
            out = model(x)
            msks.append(msks_stats(msk).unsqueeze(0))
            losses.append(loss_stats(out, y, msk, loss_fn).unsqueeze(0))
            accs.append(class_stats(out, y, msk).unsqueeze(0))
    accs   = torch.cat(accs).mean(0)
    losses = torch.cat(losses).mean(0)
    msks   = torch.cat(msks).mean(0)
    return losses, accs, msks

def train(model, optimizer, dataset, loss_fn, device=0):
    """
    """
    model.train()
    x, y, msk = dataset.sample("train")
    x, y, msk = map(lambda x:x.cuda(device), [x, y, msk])
    optimizer.zero_grad()
    out  = model(x)
    loss = loss_fn(out, y, msk)
    loss.backward()
    optimizer.step()
    msks = msks_stats(msk)
    return loss.item(), msks

def valid_koyama(model, dataset, loss_fn, device=0, ws_low=0.2, ws_high=0.5, ws_relative=True, agg_th=0.5):
    model.eval()
    with torch.no_grad():
        losses, accs, msks = [],[],[]
        
        inf = Inference(dataset.ds.fov, dataset.ds.img[:,:200,:200], device=device)
        aff = inf.segment(model)
        y,msk = dataset.ds.a(dataset.ds.lbl[None,:,:200,:200])
        y_tensor = torch.from_numpy(y[inf.out_idx]).float()
        msk_tensor = torch.from_numpy(msk[inf.out_idx])
        aff_tensor = torch.from_numpy(aff).float()
        accs   = class_stats(aff_tensor, y_tensor, msk_tensor).unsqueeze(0)
        losses = loss_stats(aff_tensor, y_tensor, msk_tensor, loss_fn).unsqueeze(0)
        msks   = msks_stats(msk_tensor).unsqueeze(0)

        #if accs.mean() > 0.87:
        #    #watershed
        #    pr  = 1-aff.mean(0)
        #    ws  = segment(aff.astype(np.float32), 0.2, 0.5, True)
        #    #agglomeration
        #    rag = agglo.Rag(ws, pr, gt_vol=dataset.ds.lbl[:,:200,:200], merge_priority_function=agglo.boundary_mean)
        #    out = rag.agglomerate(0.5, save_history=True)

        #    l1 = [out[2][i][1][0] for i in range(len(out[2]))]
        #    l2 = [out[2][i][1][1] for i in range(len(out[2]))]
        #    min_vi = [min(l1),min(l2)]
        #    min_rand = min(out[3])
        #else:
        min_vi = [0,0]
        min_rand = [0,0]
        
        return losses, accs, msks, min_vi, min_rand

def experiment(model, opt, train_ds, test_ds, monitor, val_freq=300, nstart=0, niter=30000, loss_fn=None, nvalrun=20, device=0):
    loss_fn = F.binary_cross_entropy_with_logits if loss_fn is None else loss_fn
    for i in tqdm(list(range(nstart, niter))):
        a,b = train(model, opt, train_ds, loss_fn, device)
        monitor.log_train(i, a, b)  
        if i % val_freq == 0:
            val_stats = valid(model, test_ds, loss_fn, device=device)
            trn_stats = valid(model, train_ds, loss_fn, device=device)
            monitor.log_val(i, "valid", *val_stats)  
            monitor.log_val(i, "train", *trn_stats)  
            
def loss_stats(out, y, msk, loss_fn):
    loss = loss_fn(out, y, msk, reduction="none").squeeze()
    msk  = (msk > 0).squeeze()
    loss = [loss[i][msk[i]].mean() for i in range(loss.size(0))]
    loss =  torch.FloatTensor(loss)
    return loss

def class_stats(out, lbl, msk, thr=.5):
    msk = (msk > 0).squeeze()
    acc = ((torch.sigmoid(out) > thr) == lbl.byte()).squeeze().float()
    acc = [acc[i][msk[i]].mean() for i in range(acc.size(0))]
    acc = torch.FloatTensor(acc)
    return acc

def msks_stats(msk):
    zeros = reduce(msk==0)
    pos   = reduce(msk>0.5)
    neg   = reduce((0<msk)&(msk<0.5))
    return torch.cat(list(map(lambda x:x.unsqueeze(0), [zeros, pos, neg])))

def reduce(x):
    x = x.squeeze().float()
    x = x.mean(1).mean(1).mean(1)
    return x

def moto_margin_loss(out, lbl, msk, thr=.2, reduction='mean'):
    out  = torch.sigmoid(out)
    msk2 = (torch.abs(out-lbl) > thr).float()
    msk  = msk * msk2
    return F.binary_cross_entropy(out, lbl, msk, reduction=reduction)

def margin_loss(out, lbl, msk, thr=.2, reduction='mean'):
    out  = torch.sigmoid(out)
    msk2 = (torch.abs(out-lbl) > thr).float()
    msk  = msk * msk2
    return F.binary_cross_entropy(out, lbl, msk, reduction=reduction)







#============================= background computation ==================================


def experiment_back(model, opt, train_ds, test_ds, monitor, val_freq=300, nstart=0, niter=30000, loss_fn=None, nvalrun=20, device=0):
    loss_fn = F.binary_cross_entropy_with_logits if loss_fn is None else loss_fn
    for i in tqdm(list(range(nstart, niter))):
        a,b = train_back(model, opt, train_ds, loss_fn, device)
        monitor.log_train(i, a, b)  
        if i % val_freq == 0:
            val_stats = valid_back(model, test_ds, loss_fn, device=device)
            trn_stats = valid_back(model, train_ds, loss_fn, device=device)
            monitor.log_val(i, "valid", *val_stats)  
            monitor.log_val(i, "train", *trn_stats)  


def valid_back(model, dataset, loss_fn, device=0, nrun=20):
    """
    """
    model.eval()
    with torch.no_grad():
        losses, accs, msks = [],[],[]
        for i in range(nrun):
            x, y, msk = dataset.sample_back("valid")
            x, y, msk = map(lambda x:x.cuda(device), [x, y, msk])
            out = model(x)
            msks.append(msks_stats_back(msk).unsqueeze(0))
            losses.append(loss_stats_back(out, y, msk, loss_fn).unsqueeze(0))
            accs.append(class_stats_back(out, y, msk).unsqueeze(0))
    accs   = torch.cat(accs).mean(0)
    losses = torch.cat(losses).mean(0)
    msks   = torch.cat(msks).mean(0)
    return losses, accs, msks

def train_back(model, optimizer, dataset, loss_fn, device=0):
    """
    """
    model.train()
    x, y, msk = dataset.sample_back("train")
    x, y, msk = map(lambda x:x.cuda(device), [x, y, msk])
    optimizer.zero_grad()
    out  = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    msks = msks_stats_back(msk)
    return loss.item(), msks

def loss_stats_back(out, y, msk, loss_fn):
    loss = loss_fn(out, y, reduction="none").squeeze()
    msk  = (msk > 0).squeeze()
    loss = [loss[i].mean() for i in range(loss.size(0))]
    loss =  torch.FloatTensor(loss)
    return loss

def class_stats_back(out, lbl, msk, thr=.5):
    msk = (msk > 0).squeeze()
    acc = ((torch.sigmoid(out) > thr) == lbl.byte()).squeeze().float()
    acc = [acc[i].mean() for i in range(acc.size(0))]
    acc = torch.FloatTensor(acc)
    return acc

def msks_stats_back(msk):
    zeros = reduce(msk==0)
    pos   = reduce(msk>0.5)
    neg   = reduce((0<msk)&(msk<0.5))
    return torch.cat(list(map(lambda x:x.unsqueeze(0), [zeros, pos, neg])))

def reduce(x):
    x = x.squeeze().float()
    x = x.mean(1).mean(1)#.mean(1)
    return x