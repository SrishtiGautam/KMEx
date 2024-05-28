
import torch
import torch.nn as nn

from utils.prp import *
from utils.model import *
from utils.dataloader_nrm import NRM

from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

import umap

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


##########################################################################

NLLLOSS = nn.NLLLoss(reduction=('mean')).to(device)
CELOSS  = nn.CrossEntropyLoss(reduction=('mean')).to(device)
MSELOSS = nn.MSELoss(reduction=('mean')).to(device)

train_loader, push_loader, test_loader = None, None, None



##########################################################################

from scipy.optimize import linear_sum_assignment

def accuracy(y_true, y_pred):
    assert y_pred.shape[0] == y_true.shape[0]

    D = int( max(y_pred.max(), y_true.max()) + 1 )
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.shape[0]):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    acc =  sum([w[i, j] for i, j in ind]) * 1.0 / np.prod(y_pred.shape)
    return acc


##########################################################################


def PROTOVAE_fwd(title, model, img, lbl, wgt, no_loss, lrp):

    # Project
    ebd = model.encoder(img)

    mu     = ebd[:,:model.LATENT]
    logVar = ebd[:,model.LATENT:].clamp(np.log(1e-8), -np.log(1e-8))
    cov    = torch.exp( logVar/2)
    if no_loss or lrp:
        z  = mu
    else:
        z  = torch.randn(mu.shape).to(device) * cov + mu

    if 'kmeans' in title:
        prt = model.kmc_prt
        classes = model.kmc_prt_classes
        dist = torch.cdist(z, prt, p=2)
        sim_scores  = torch.log( dist + 1 ) - torch.log( dist + 1e-8 )
        sss = nn.Softmax(dim=-1)(sim_scores)
        proba = torch.zeros(img.shape[0],model.K)
        for k in range(model.K):
            proba[:,k] = sss[:, classes == k ].sum(-1)

    else:
        dist = torch.cdist(z, model.prt, p=2)
        sim_scores  = torch.log(dist + 1) - torch.log(dist + 1e-8)
        logit = model.clf(sim_scores)
        proba = nn.Softmax(dim=-1)(logit)

        if lrp:
            return proba, sim_scores

    if no_loss:
        return proba,sim_scores
    if lrp:
        sim_scores  = torch.log(dist + 1) - torch.log(dist + 1e-8)
        return proba,sim_scores

    # ## ###
    N = mu.shape[0]

    # CE
    loss_ce = CELOSS(proba, lbl)
    if np.isnan(float(loss_ce)):
        print('loss_ce is nan')
        raise ValueError

    # REC
    rec = model.decoder(z)
    loss_rec = MSELOSS(rec,img*.5+.5)
    if np.isnan(float(loss_rec)):
        print('loss_rec is nan')
        raise ValueError

    # DKL
    w = torch.zeros((model.H,N)).to(device)
    loss_kl = torch.zeros((model.H,N)).to(device)
    p = torch.distributions.Normal(mu, cov)
    for h in range(model.H):
        q_v = model.prt[ lbl*model.H+h ,:].to(device)
        q = torch.distributions.Normal(q_v, torch.ones(q_v.shape).to(device))
        dkl = torch.distributions.kl.kl_divergence(p, q).mean(-1).to(device)
        loss_kl[h] = dkl

        w[h] = (sim_scores * torch.nn.functional.one_hot(lbl*model.H+h,model.NB_PRT).to(device)).sum(-1)
    w = w / w.sum(0)[None,:]
    loss_kl = torch.mean(loss_kl * w)
    if np.isnan(float(loss_kl)):
        print('loss_kl is nan')
        raise ValueError

    # ORTH
    EYE_H = torch.eye(model.H).to(device).detach()
    loss_orth = 0
    for k in range(model.K):
        p = model.prt[k*model.H:(k+1)*model.H,:]
        pm = p.mean(0).detach()
        p = p - pm
        pp = p @ p.T
        ppe = (pp - EYE_H)
        loss_orth += torch.norm(ppe, p=2) / model.K
    if np.isnan(float(loss_orth)):
        print('loss_orth is nan')
        raise ValueError

    print(' '*18+'CE {:.4f} REC {:.4f} KL {:.4f} ORTH {:.4f}'.format(
        loss_ce.item(), loss_rec.item(), loss_kl.item(), loss_orth.item()
        ), end='\r')
    loss = loss_ce*wgt[0] + loss_rec*wgt[1] + loss_kl*wgt[2] + loss_orth*wgt[3]

    return proba, loss


##########################################################################

def PPNET_fwd(title, model, img, lbl, wgt, no_loss, lrp):
    logit,dist = model(img)
    proba = nn.Softmax(dim=-1)(logit)

    if no_loss:
        return proba,logit

    if lrp:
        if 'kmeans' in title:

            prt = model.kmc_prt
            classes = model.kmc_prt_classes.long()

            ebd = model.encoder(img)
            if len(ebd.shape) == 1:
               ebd = ebd[None,:]
            dist = torch.cdist(ebd, prt, p=2)

            sim_scores = torch.log(dist + 1) - torch.log(dist + 1e-4)
            sss = nn.Softmax(dim=-1)(sim_scores)
            proba = torch.zeros(img.shape[0],model.K)
            for k in range(model.K):
               proba[:,k] = sss[:, classes == k ].sum(-1)

        else:
            dist = torch.sqrt(dist)

        sim_scores  = torch.log(dist + 1) - torch.log(dist + 1e-4)
        return proba, sim_scores

    return proba


##########################################################################

def FLINT_fwd(title, model, img, lbl, no_loss, lrp):

    g_ebd = model.encoder(img)
    sim_scores = model.s(g_ebd)
    logit = sim_scores
    proba = nn.Softmax(dim=-1)(logit)

    if no_loss:
        return proba,logit

    if lrp:
        if 'kmeans' in title:
            prt = model.kmc_prt
            classes = model.kmc_prt_classes.long()

            dist = torch.cdist(g_ebd, prt, p=2)

            proba = torch.nn.functional.one_hot( classes[dist.argmin(-1)].long(), num_classes=model.K)

        else:
            dist = torch.cdist(g_ebd, model.prt, p=2)
        sim_scores  = torch.log(dist + 1) - torch.log(dist + 1e-4)

        return proba,sim_scores

    return proba, sim_scores


##########################################################################


def BBOX_fwd(title, model, img, lbl, wgt, no_loss, lrp):

    ebd = model.encoder(img)

    if 'kmeans' in title:
        if 'kmeans_all' in title:
            prt = model.kma_prt
            classes = model.kma_prt_classes.long()
        else:
            prt = model.kmc_prt
            classes = model.kmc_prt_classes.long()

        dist = torch.cdist(ebd, prt, p=2)
        logit  = torch.log(dist + 1) - torch.log(dist + 1e-4)


        sim_scores = torch.log(dist + 1) - torch.log(dist + 1e-4)
        sss = nn.Softmax(dim=-1)(sim_scores)
        proba = torch.zeros(img.shape[0],model.K)
        for k in range(model.K):
            proba[:,k] = sss[:, classes == k ].sum(-1)

    else:
        logit = model.clf(ebd)
        proba = nn.Softmax(dim=-1)(logit)

    if no_loss:
        return proba,logit

    if lrp:
        return proba,sim_scores

    loss = CELOSS(logit, lbl)

    return proba, loss


##########################################################################


def model_fwd(title, model, img, lbl, wgt=None, no_loss=False, lrp=False):
    if 'PROTOVAE' in title:
        return PROTOVAE_fwd( title, model, img, lbl, wgt, no_loss, lrp )
    elif 'PPNET' in title:
        return PPNET_fwd( title, model, img, lbl, wgt, no_loss, lrp )
    elif 'FLINT' in title:
        return FLINT_fwd( title, model, img, lbl, wgt, no_loss, lrp )
    else:
        return BBOX_fwd( title, model, img, lbl, wgt, no_loss, lrp )


##########################################################################

def test(title, model):
    global train_loader, test_loader
    model.eval()

    e_pred,e_lbl = [],[]
    for i, (img, lbl) in enumerate(test_loader):
        with torch.no_grad():
            img = img.to(device)
            lbl = lbl.to(device).squeeze()

            if 'kmeans_class' in title:
                ebd = model.encoder(img)[:,:model.LATENT].detach().cpu().numpy()
                pred = model.kmc.predict(ebd)
                pred = model.kmc_prt_classes.detach().cpu().numpy()[pred]
            else:
                proba,_ = model_fwd(title, model, img, lbl, no_loss=True)
                pred = proba.argmax(-1).detach().cpu().numpy()

            e_pred += pred.tolist()
            e_lbl  += lbl.detach().cpu().numpy().tolist()

    e_pred = np.array(e_pred)
    e_lbl = np.array(e_lbl)

    acc = accuracy( e_lbl, e_pred )*100
    return acc


##########################################################################

def train(
    title,
    model,
    optim,
    nb_epoch,
    wgt,
    ):
    print('\n# Train the model')

    global train_loader, test_loader

    params = []
    if 'PROTOVAE' in title:
        params.append({'params': model.encoder.parameters()})#, 'weight_decay':1e-3})
        params.append({'params': model.decoder.parameters()})#, 'weight_decay':1e-3})
        params.append({'params': model.clf.parameters()})
        params.append({'params': model.prt})
    else:
        params.append({'params': model.encoder.parameters()})
        params.append({'params': model.clf.parameters()})

    if model.DATASET == 'tiny':
        params.pop(0)

    if 'SGD' in optim:
        name,lr,mom = optim.split('|')[:3]
        optimizer = torch.optim.SGD(
                            params,
                            lr=float(lr),
                            momentum=float(mom)
                            )
    elif 'Adam' in optim:
        name,lr = optim.split('|')[:2]
        optimizer = torch.optim.Adam(
                            params,
                            lr=float(lr),
                            )
    elif 'Adamax' in optim:
        name,lr = optim.split('|')[:2]
        optimizer = torch.optim.Adamax(
                            params,
                            lr=float(lr),
                            )
    elif 'AdamW' in optim:
        name,lr = optim.split('|')[:2]
        optimizer = torch.optim.AdamW(
                            params,
                            lr=float(lr),
                            )
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        len(train_loader)*nb_epoch,
        eta_min=1e-4,
        last_epoch=-1,
        verbose=False
    )

    n_batch = len(train_loader)
    m_loss = 0
    epoch = 0
    for epoch in range(nb_epoch):
        model.train()
        m_loss = 0
        for i, (img, lbl) in enumerate(train_loader):
            with torch.no_grad():
                img = img.to(device)
                lbl = lbl.to(device).squeeze()

            proba,loss = model_fwd(title, model, img, lbl, wgt)

            # compute gradient and do GD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            m_loss += float(loss.detach().cpu().numpy())

            pred = proba.argmax(-1).detach().cpu().numpy()

            acc = accuracy( lbl.detach().cpu().numpy(), pred )*100

            if 'PROTOVAE' not in model.title:
                print('Epoch {:d} Iter {:d}/{:d} Loss {:.6f} Train Acc {:.1f}'.format(epoch+1, i+1, n_batch, float(loss), acc)+' '*20, end='\r')
            else:
                print('Epoch {:d} Acc {:.1f}'.format(epoch+1,acc), end='\r')

        m_loss /= (i+1)

        if (epoch+1) % 5 == 0:
            model.eval()
            acc = test(title,model)

            print('\nEpoch {:d} Loss {:.6f} Test Acc {:.1f}'.format(epoch+1, m_loss, acc)+' '*20, end='\n')

    acc = test(title,model)
    print('\nEpoch {:d} Loss {:.6f} Test Acc {:.1f}'.format(epoch+1, m_loss, acc)+' '*20, end='\n')

    return acc


##########################################################################

def get_embedding(model):
    model.eval()

    print('>> Project')
    nb = len(push_loader.dataset)
    bs = push_loader.batch_size

    embed = torch.empty((nb,model.LATENT)).to(device)
    if hasattr( push_loader.dataset, 'labels'):
        label = push_loader.dataset.labels
    elif hasattr( push_loader.dataset, 'attr'):
        label = push_loader.dataset.attr.squeeze()
    elif hasattr( push_loader.dataset, 'targets'):
        label = push_loader.dataset.targets
    else:
        label = []

    for i, (img, lbl) in tqdm(enumerate(push_loader)):
        with torch.no_grad():
            img = img.to(device)

            try:
                ebd = model.encoder(img.to(device))[:,:model.LATENT]
            except:
                ebd = model.conv_features(img.to(device)).squeeze()
            embed[i*bs:(i+1)*bs] = ebd
            if len(label) < nb:
                label += lbl
    label = torch.LongTensor(label).to(device)
    return embed, label


##########################################################################

def get_kmeans_class(model,embed=None,label=None, knn=1):

    with torch.no_grad():
        print('\n# Get Kmeans per class and closest images')

        if 'PROTOVAE' in model.title:
            mrn,srn = .5*torch.ones((1,3,1,1)).to(device), .5*torch.ones((1,3,1,1)).to(device)
        else:
            mrn,srn = [torch.tensor(v,dtype=torch.float32)[None,:,None,None].to(device) for v in NRM[model.DATASET]]

        model.eval()

        if embed is None or label is None:
            print('>> Project')
            embed = np.empty((0,model.LATENT))
            label = np.empty((0,))
            for i, (img, lbl) in tqdm(enumerate(push_loader)):
                with torch.no_grad():
                    img = img.cuda(0, non_blocking=True)
                    lbl = lbl.squeeze().detach().cpu().numpy()

                    if 'PPNET' in model.title:
                        ebd = model.conv_features(img.to(device)).squeeze()
                    else:
                        ebd = model.encoder(img.to(device))[:,:model.LATENT]

                    embed = np.concatenate( (embed, ebd.detach().cpu().numpy()), axis=0)
                    label = np.concatenate( (label, lbl) )
        elif type(embed) != np.ndarray:
            embed = embed.detach().cpu().numpy()
            label = label.detach().cpu().numpy().astype(int)

        print('>> Fit kmeans')
        model.kmc_prt = torch.zeros((model.NB_PRT,model.LATENT), requires_grad=True).to(device)
        model.kmc_prt_classes = torch.repeat_interleave(torch.arange(model.K),model.H,0).to(device)
        model.kmc_prt_closest_images = torch.empty((model.NB_PRT,model.CHANNEL,model.WIDTH,model.HEIGHT), requires_grad=False).to(device)
        model.kmc_prt_images = torch.empty((model.NB_PRT,model.CHANNEL,model.WIDTH,model.HEIGHT), requires_grad=False).to(device)
        model.kmc_prt_train_ipt = torch.zeros((model.NB_PRT,), requires_grad=False).to(device)

        model.kmc = KMeans(n_clusters=model.NB_PRT, init='random', n_init=1, max_iter=1,).fit(embed)

        for k in tqdm(range(model.K)):
            k_embed = embed[label==k]
            nnz  = np.nonzero(label==k)[0]

            kmeans = KMeans(n_clusters=model.H).fit(k_embed)
            model.kmc_prt[k*model.H:(k+1)*model.H] += torch.FloatTensor(kmeans.cluster_centers_).to(device)
            model.kmc.cluster_centers_[k*model.H:(k+1)*model.H] = kmeans.cluster_centers_

            dist = np.linalg.norm(k_embed[:,None,:] - kmeans.cluster_centers_[None,:,:],axis=-1)
            model.kmc_prt_train_ipt[k*model.H:(k+1)*model.H] += torch.FloatTensor(np.histogram(dist.argmin(-1),range(model.H+1),density=False)[0]).to(device)

            idc = nnz[dist.argmin(0)].tolist()
            for h,ik in enumerate(idc):
                kh = k*model.H+h
                model.kmc_prt_images[kh] = (push_loader.dataset[ik][0]).to(device)*srn[0]+mrn[0]

            dist = np.linalg.norm(embed[:,None,:] - kmeans.cluster_centers_[None,:,:],axis=-1)

            idx = dist.argmin(0)
            for h,ix in enumerate(idx):
                kh = k*model.H+h
                model.kmc_prt_closest_images[kh] = (push_loader.dataset[ix][0]).to(device)*srn[0]+mrn[0]

        print('>> KM per Class Accuracy', end=' ')
        model.kmc_acc = test(model.kmc_title, model)
        print(np.round(float(model.kmc_acc),1))



##########################################################################

def plot_prt_img(model,folder,per_class=True):
    print('\n# Plot the closest images to the prototypes', 'within class' if per_class else 'in training set')

    is_tiny = (model.DATASET == 'tiny')

    is_protovae = 'PROTOVAE' in model.title
    clf_h = model.H if is_protovae else 1

    if per_class:
        prt_images = model.prt_images.detach().cpu().numpy()
        km_prt_images = model.kmc_prt_images.detach().cpu().numpy()
    else:
        prt_images = model.prt_closest_images.detach().cpu().numpy()
        km_prt_images = model.kmc_prt_closest_images.detach().cpu().numpy()

    if is_tiny:
        fig_clf = plt.figure('clf',(20,40))
        fig_km  = plt.figure('km', (20,40))
    else:
        fig_clf = plt.figure('clf',(2*clf_h,2*model.K))
        fig_km  = plt.figure('km', (2*model.H,2*model.K))


    for k in tqdm(range(model.K)):
        if not is_protovae:
            if is_tiny:
                sb = fig_clf.add_subplot(20,10,k+1)
            else:
                sb = fig_clf.add_subplot(model.K,clf_h,k+1)

            ims = prt_images[k]
            ims = np.moveaxis(ims, 0, -1)
            ims = np.clip(ims,0,1)
            sb.imshow( ims )

            sb.set_xticks([]);sb.set_yticks([])
            sb.set_title( '{:.2f}'.format(model.prt_train_ipt[k]) )

        if is_tiny:
            sb = fig_km.add_subplot(20,10,k+1)

            ims = km_prt_images[k*model.H]
            ims = np.moveaxis(ims, 0, -1)
            ims = np.clip(ims,0,1)
            sb.imshow( ims )

            sb.set_xticks([]);plt.yticks([])
            sb.set_title( '({:.0f}) {:.2f}'.format(model.kmc_prt_classes[k*model.H], model.kmc_prt_train_ipt[k*model.H]) )

            if is_protovae:
                sb = fig_clf.add_subplot(20,10,k+1)

                ims = prt_images[k*model.H]
                ims = np.moveaxis(ims, 0, -1)
                ims = np.clip(ims,0,1)
                sb.imshow( ims )

                sb.set_xticks([]);plt.yticks([])
                sb.set_title( '{:.2f}'.format(model.prt_train_ipt[k*model.H]) )

        else:
            for h in range(model.H):
                kh = k*model.H+h

                sb = fig_km.add_subplot(model.K,model.H,kh+1)

                ims = km_prt_images[kh]
                ims = np.moveaxis(ims, 0, -1)
                ims = np.clip(ims,0,1)
                sb.imshow( ims )

                sb.set_xticks([]);plt.yticks([])
                sb.set_title( '({:.0f}) {:.2f}'.format(model.kmc_prt_classes[kh], model.kmc_prt_train_ipt[kh]) )

                if is_protovae:
                    sb = fig_clf.add_subplot(model.K,model.H,kh+1)

                    ims = prt_images[kh]
                    ims = np.moveaxis(ims, 0, -1)
                    ims = np.clip(ims,0,1)
                    sb.imshow( ims )

                    sb.set_xticks([]);plt.yticks([])
                    sb.set_title( '{:.2f}'.format(model.prt_train_ipt[kh]) )

    if per_class:
        head = 'PRT'
    else:
        head = 'CLOSEST'
    fig_clf.tight_layout()
    fig_clf.savefig(folder+'plot/['+head+'] '+model.title )
    plt.close(fig_clf)

    fig_km.tight_layout()
    fig_km.savefig(folder+'plot/['+head+'] '+model.kmc_title )
    plt.close(fig_km)

    print("Saved plots to "+folder+'plot/['+head+']')


##########################################################################

def plot_umap(
        model,
        folder,
        _nb_data=5000,
        ):
    print('\n# Plot UMAP')
    global train_loader, push_loader, test_loader


    mrn,srn = [torch.tensor(v,dtype=torch.float32)[None,:,None,None].to(device) for v in NRM[model.DATASET]]

    model.eval()

    viz_dataset = torch.empty((0,model.LATENT))
    viz_lbl = []
    for (img,lbl) in test_loader:
        if viz_dataset.shape[0] < _nb_data:
            with torch.no_grad():
                ebd = model.encoder(img.to(device)).detach().cpu()[:,:model.LATENT]
                viz_dataset = torch.cat( (viz_dataset, ebd ), dim=0)
                viz_lbl += list(lbl.squeeze())

    viz_dataset = viz_dataset.detach().cpu().numpy()

    print('>> Fit UMAP')
    viz_umap = umap.UMAP()
    viz_ebd  = viz_umap.fit_transform(viz_dataset)

    viz_prt  = viz_umap.transform(model.prt.detach().cpu().numpy())
    viz_prt_lbl = model.prt_classes.detach().cpu().numpy()
    viz_ipt = np.clip(model.prt_train_ipt.detach().cpu().numpy(),1,1)

    viz_km_prt  = viz_umap.transform(model.km_prt.detach().cpu().numpy())
    viz_km_prt_lbl = model.km_prt_classes.detach().cpu().numpy()
    viz_km_ipt = np.clip(model.km_prt_train_ipt.detach().cpu().numpy(),1,1)

    print('>> Plot')
    plt.figure('umap',(18,4.5))
    plt.suptitle( model.title )

    plt.subplot(1,2,1)
    plt.scatter( viz_ebd[:,0], viz_ebd[:,1], c=viz_lbl, s=16, marker='.' , alpha=.5)
    plt.scatter( viz_prt[:,0], viz_prt[:,1], c='w', s=48*viz_ipt, marker='s' )
    plt.scatter( viz_prt[:,0], viz_prt[:,1], c='k', s=32*viz_ipt, marker='s' )
    plt.scatter( viz_prt[:,0], viz_prt[:,1], c=viz_prt_lbl, s=16*viz_ipt, marker='s')
    plt.title('Orig   acc={:.1f}'.format(model.acc))
    plt.xticks([]);plt.yticks([])
    plt.tight_layout()

    plt.subplot(1,2,2)
    plt.scatter( viz_ebd[:,0], viz_ebd[:,1], c=viz_lbl, s=16, marker='.' , alpha=.5)
    plt.scatter( viz_km_prt[:,0], viz_km_prt[:,1], c='w', s=48*viz_km_ipt, marker='s' )
    plt.scatter( viz_km_prt[:,0], viz_km_prt[:,1], c='k', s=32*viz_km_ipt, marker='s' )
    plt.scatter( viz_km_prt[:,0], viz_km_prt[:,1], c=viz_km_prt_lbl, s=16*viz_km_ipt, marker='s' )
    plt.title('KMEx   acc={:.1f}'.format(model.km_acc))
    plt.xticks([]);plt.yticks([])
    plt.tight_layout()

    plt.savefig(folder+'plot/[UMAP] '+model.title)
    plt.savefig(folder+'/../[UMAP] '+model.title)
    plt.close()


