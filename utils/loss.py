# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh, check_version, xywh2xyxy)
import cv2
import numpy as np


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def compute_probs(data, n=10): 
    h, e = np.histogram(data, n, range=(-200,200))
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))

def add_one_smoothing(p):
    p = np.round(p*1000)
    p += 1
    p /= np.sum(p)
    return p

def calc_postreg_loss(train_sample, test_sample, loss_type='kl', n_bins=10): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=n_bins)

    if loss_type == 'l1':
        return np.abs(np.subtract(p,q)).mean()
    elif loss_type == 'l2':
        return np.square(np.subtract(p,q)).mean()
    else:
        p = add_one_smoothing(p)
        q = add_one_smoothing(q)

        list_of_tuples = support_intersection(p, q)
        p, q = get_probs(list_of_tuples)
        
        return kl_divergence(p, q)

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    X,y = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

from sklearn import mixture

def calc_postreg_loss_gmm(train_sample, test_sample):
  g1 = mixture.GaussianMixture(n_components=2,random_state=0).fit(train_sample)
  g2 = mixture.GaussianMixture(n_components=2,random_state=0).fit(test_sample)

  return gmm_kl(g1,g2)


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride', 'no':
            setattr(self, k, getattr(det, k))

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid

    def __call__(self, p, targets,paths=None,loss_type="kl"):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lreg = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
  
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        if paths is not None:
          imgs = []
          for j in range(len(paths)):
            img = cv2.imread(paths[j])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            imgs.append(img)

          out = []

          for i in range(self.nl):
            bs, _, ny, nx, _ = p[i].shape
            if self.grid[i].shape[2:4] != p[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = p[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            
            out.append(y.view(bs, -1, self.no))

          out = non_max_suppression(torch.cat(out,1), 0.25, 0.6, labels=[], multi_label=True, agnostic=False)
          
          # initialise intensity lists
          target_intensity = []
          pred_intensity = []

          # initialise size lists
          target_size = []
          pred_size = []

          for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            predn = pred.clone()

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                # labels -> [class,x1,y1,x2,y2] , predn -> [x1,y1,x2,y2,conf,class]

                pixel_iel = 0
                size_iel = 0
                num_iel = 0
                pixel_epith = 0
                size_epith = 0
                num_epith = 0

                for i in range(labelsn.shape[0]):
                  x1 , y1 , x2, y2 = int(labelsn[i][1]*imgs[si].shape[1]) , int(labelsn[i][2]*imgs[si].shape[0]) , int(labelsn[i][3]*imgs[si].shape[1]) , int(labelsn[i][4]*imgs[si].shape[0])
                  pixel_sum = np.sum(imgs[si][y1:y2,x1:x2,2])
                  pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))

                  if int(labelsn[i][0]) == 0:
                    pixel_epith += pixel_sum
                    size_epith += (x2 - x1 + 1)*(y2 - y1 + 1)
                    num_epith += 1
                  else:
                    pixel_iel += pixel_sum
                    size_iel += (x2 - x1 + 1)*(y2 - y1 + 1)
                    num_iel += 1

                if num_iel != 0:
                    pixel_iel /= num_iel
                    size_iel /= num_iel
                if num_epith != 0:
                    pixel_epith /= num_epith
                    size_epith /= num_epith

                pixel_epith -= pixel_iel
                size_epith -= size_iel
                    
                if num_iel > 0 and num_epith > 0:
                    target_intensity.append(pixel_epith)
                    target_size.append(size_epith)

                pixel_iel = 0
                size_iel = 0
                num_iel = 0
                pixel_epith = 0
                size_epith = 0
                num_epith = 0

                for i in range(predn.shape[0]):
                  x1 , y1 , x2, y2 = int(predn[i][0]) , int(predn[i][1]) , int(predn[i][2]) , int(predn[i][3])
                  pixel_sum = np.sum(imgs[si][y1:y2,x1:x2,2])
                  pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))

                  if int(predn[i][5]) == 0:
                    pixel_epith += pixel_sum
                    size_epith += (x2 - x1 + 1)*(y2 - y1 + 1)
                    num_epith += 1
                  else:
                    pixel_iel += pixel_sum
                    size_iel += (x2 - x1 + 1)*(y2 - y1 + 1)
                    num_iel += 1

                if num_iel != 0:
                    pixel_iel /= num_iel
                    size_iel /= num_iel
                if num_epith != 0:
                    pixel_epith /= num_epith
                    size_epith /= num_epith

                pixel_epith -= pixel_iel
                size_epith -= size_iel
                    
                if num_iel > 0 and num_epith > 0:
                    pred_intensity.append(pixel_epith)
                    pred_size.append(size_epith)

          if len(target_intensity) > 0 and len(pred_intensity) > 0:
            if loss_type == "gmm":
                target_intensity = np.array(target_intensity)
                target_intensity = target_intensity.reshape((target_intensity.shape[0],1))
                target_size = np.array(target_size)
                target_size = target_size.reshape((target_size.shape[0],1))
                pred_intensity = np.array(pred_intensity)
                pred_intensity = pred_intensity.reshape((pred_intensity.shape[0],1))
                pred_size = np.array(pred_size)
                pred_size = pred_size.reshape((pred_size.shape[0],1))
                lreg += calc_postreg_loss_gmm(np.concatenate((target_intensity,target_size)) , np.concatenate((pred_intensity,pred_size)))

            else:
                lreg += calc_postreg_loss(np.array(target_intensity), np.array(pred_intensity))
            
            lreg *= (0.1)
            print('Regularisation Loss : ', lreg)
            return (lbox + lobj + lcls + lreg) * bs, torch.cat((lbox, lobj, lcls)).detach()
          else:
            print('No defined regularisation loss')

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
