
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

def to_torch(x):
    return torch.from_numpy(x).float()

def to_cuda(x, use_cuda):
    if use_cuda:
        try:
            return x.cuda()
        except (RuntimeError, AssertionError) as e:
            print(f"Warning: CUDA error: {e}. Falling back to CPU.")
            return x
    else:
        return x

def to_numpy(x):
    return x.detach().cpu().numpy()

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, demo_dim, use_cuda=True):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.demo_dim = demo_dim
        self.use_cuda = use_cuda
        self.enc1 = to_cuda(nn.Linear(input_dim, 1000).float(), use_cuda)
        self.enc2 = to_cuda(nn.Linear(1000, latent_dim).float(), use_cuda)
        self.dec1 = to_cuda(nn.Linear(latent_dim+demo_dim, 1000).float(), use_cuda)
        self.dec2 = to_cuda(nn.Linear(1000, input_dim).float(), use_cuda)

    def enc(self, x):
        x = F.relu(self.enc1(x))
        z = self.enc2(x)
        return z

    def gen(self, n):
        return to_cuda(torch.randn(n, self.latent_dim).float(), self.use_cuda)

    def dec(self, z, demo):
        z = to_cuda(torch.cat([z, demo], dim=1), self.use_cuda)
        x = F.relu(self.dec1(z))
        x = self.dec2(x)
        #x = x.reshape(len(z), 264, 5)
        #x = torch.einsum('nac,nbc->nab', x, x)
        #a,b = np.triu_indices(264, 1)
        #x = x[:,a,b]
        return x

def rmse(a, b, mean=torch.mean):
    return mean((a-b)**2)**0.5

def latent_loss(z, use_cuda=True):
    C = z.T@z
    mu = torch.mean(z, dim=0)
    tgt1 = to_cuda(torch.eye(z.shape[-1]).float(), use_cuda)*len(z)
    tgt2 = to_cuda(torch.zeros(z.shape[-1]).float(), use_cuda)
    loss_C = rmse(C, tgt1)
    loss_mu = rmse(mu, tgt2)
    return loss_C, loss_mu, C, mu

def decor_loss(z, demo, use_cuda=True):
    ps = []
    losses = []
    for di in range(demo.shape[1]):
        d = demo[:,di]
        d = d - torch.mean(d)
        p = torch.einsum('n,nz->z', d, z)
        p = p/torch.std(d)
        p = p/torch.einsum('nz,nz->z', z, z)
        tgt = to_cuda(torch.zeros(z.shape[-1]).float(), use_cuda)
        loss = rmse(p, tgt)
        losses.append(loss)
        ps.append(p)
    losses = torch.stack(losses)
    return losses, ps

def pretty(x):
    return f'{round(float(x), 4)}'

def demo_to_torch(demo, demo_types, pred_stats, use_cuda):
    demo_t = []
    demo_idx = 0
    for d,t,s in zip(demo, demo_types, pred_stats):
        if t == 'continuous':
            demo_t.append(to_cuda(to_torch(d), use_cuda))
        elif t == 'categorical':
            for dd in d:
                if dd not in s:
                    print(f'Model not trained with value {dd} for categorical demographic {demo_idx}')
                    raise Exception('Bad demographic')
            for ss in s:
                idx = (d == ss).astype('bool')
                zeros = torch.zeros(len(d))
                zeros[idx] = 1
                demo_t.append(to_cuda(zeros, use_cuda))
        demo_idx += 1
    demo_t = torch.stack(demo_t).permute(1,0)
    return demo_t

def train_vae(vae, x, demo, demo_types, nepochs, pperiod, bsize, loss_C_mult, loss_mu_mult, loss_rec_mult, loss_decor_mult, loss_pred_mult, lr, weight_decay, alpha, LR_C, ret_obj):
    # Get linear predictors for demographics
    pred_w = []
    pred_i = []
    # Pred stats are mean and std for continuous, and a list of all values for categorical
    pred_stats = []
    for i,d,t in zip(range(len(demo)), demo, demo_types):
        print(f'Fitting auxilliary guidance model for demographic {i} {t}...', end='')
        if t == 'continuous':
            pred_stats.append([np.mean(d), np.std(d)])
            reg = Ridge(alpha=alpha).fit(x, d)
            reg_w = to_cuda(to_torch(reg.coef_), vae.use_cuda)
            reg_i = reg.intercept_
            pred_w.append(reg_w)
            pred_i.append(reg_i)
        elif t == 'categorical':
            pred_stats.append(sorted(list(set(list(d)))))
            reg = LogisticRegression(C=LR_C).fit(x, d)
            # Binary
            if len(reg.coef_) == 1:
                reg_w = to_cuda(to_torch(reg.coef_[0]), vae.use_cuda)
                reg_i = reg.intercept_[0]
                pred_w.append(-reg_w)
                pred_i.append(-reg_i)
                pred_w.append(reg_w)
                pred_i.append(reg_i)
            # Categorical
            else:
                for i in range(len(reg.coef_)):
                    reg_w = to_cuda(to_torch(reg.coef_[i]), vae.use_cuda)
                    reg_i = reg.intercept_[i]
                    pred_w.append(reg_w)
                    pred_i.append(reg_i)
        else:
            print(f'demographic type "{t}" not "continuous" or "categorical"')
            raise Exception('Bad demographic type')
        print(' done')
    ret_obj.pred_stats = pred_stats
    # Convert input to pytorch
    print('Converting input to pytorch')
    x = to_cuda(to_torch(x), vae.use_cuda)
    # Convert demographics to pytorch
    print('Converting demographics to pytorch')
    demo_t = demo_to_torch(demo, demo_types, pred_stats, vae.use_cuda)
    # Training loop
    print('Beginning VAE training')
    ce = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    for e in range(nepochs):
        perm = torch.randperm(len(x), device=x.device)
        for bs in range(0,len(x),bsize):
            idx = perm[bs:(bs+bsize)]
            # A batch no larger than latent_dim can't satisfy the covariance loss
            if len(idx) <= vae.latent_dim:
                continue
            xb = x[idx]
            db = demo_t[idx]
            optim.zero_grad()
            # Reconstruct
            z = vae.enc(xb)
            y = vae.dec(z, db)
            loss_C, loss_mu, _, _ = latent_loss(z, vae.use_cuda)
            loss_decor, _ = decor_loss(z, db, vae.use_cuda)
            loss_decor = sum(loss_decor)
            loss_rec = rmse(xb, y)
            # Sample demographics
            demo_gen = []
            for s,t in zip(pred_stats, demo_types):
                if t == 'continuous':
                    mu = s[0]
                    std = s[1]
                    dd = torch.randn(100).float()
                    dd = dd*std+mu
                    dd = to_cuda(dd, vae.use_cuda)
                    demo_gen.append(dd)
                elif t == 'categorical':
                    idx = random.randint(0, len(s)-1)
                    for i in range(len(s)):
                        if idx == i:
                            dd = torch.ones(100).float()
                        else:
                            dd = torch.zeros(100).float()
                        dd = to_cuda(dd, vae.use_cuda)
                        demo_gen.append(dd)
            demo_gen = torch.stack(demo_gen).permute(1,0)
            # Generate
            z = vae.gen(100)
            y = vae.dec(z, demo_gen)
            # Regressor/classifier guidance loss
            losses_pred = []
            idcs = []
            dg_idx = 0
            for s,t in zip(pred_stats, demo_types):
                if t == 'continuous':
                    yy = y@pred_w[dg_idx]+pred_i[dg_idx]
                    loss = rmse(demo_gen[:,dg_idx], yy)
                    losses_pred.append(loss)
                    idcs.append(float(demo_gen[0,dg_idx]))
                    dg_idx += 1
                elif t == 'categorical':
                    loss = 0
                    for i in range(len(s)):
                        yy = y@pred_w[dg_idx]+pred_i[dg_idx]
                        loss += ce(torch.stack([-yy, yy], dim=1), demo_gen[:,dg_idx].long())
                        idcs.append(int(demo_gen[0,dg_idx]))
                        dg_idx += 1
                        losses_pred.append(loss)
            total_loss = loss_C_mult*loss_C + loss_mu_mult*loss_mu + loss_rec_mult*loss_rec + loss_decor_mult*loss_decor + loss_pred_mult*sum(losses_pred)
            total_loss.backward()
            optim.step()
            if e%pperiod == 0 or e == nepochs-1:
                print(f'Epoch {e} ', end='')
                print(f'ReconLoss {pretty(loss_rec)} ', end='')
                print(f'CovarianceLoss {pretty(loss_C)} ', end='')
                print(f'MeanLoss {pretty(loss_mu)} ', end='')
                print(f'DecorLoss {pretty(loss_decor)} ', end='')
                losses_pred = [pretty(loss) for loss in losses_pred]
                print(f'GuidanceTargets {idcs} GuidanceLosses {losses_pred} ', end='')
                print()
    print('Training complete.')
                

