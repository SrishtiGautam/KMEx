import os
import torch
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from utils.prp import *

from tqdm import tqdm

from utils.common import model_fwd
from utils.dataloader_nrm import NRM

import torchvision.transforms as transforms

from utils.lrp_general6 import *

ToPIL = transforms.ToPILImage()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 8 if torch.cuda.is_available() else 0


##########################################################################

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import homogeneity_score as homog

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
def AI_AD(
    title,
    model,
    folder,
    loader,
    selec_images,
    ):

    print('\n# Compute AD/AI for '+('BBOX' if 'kmeans' not in title else ('KMEx ALL' if 'kmeans_all' in title else 'KMEx CLASS')) )

    nb_prt = model.kmc_prt.shape[0]//model.K if 'kmeans' in title else model.prt.shape[0]//model.K
    WIDTH  = model.WIDTH
    HEIGHT = model.HEIGHT

    ad_sim = np.zeros((len(selec_images),nb_prt))
    ai_sim = np.zeros((len(selec_images),nb_prt))

    dataset = loader.dataset

    m_nrm,s_nrm = [np.array(v)[:,None,None] for v in NRM[model.DATASET]]

    r_img_path = '/'.join(folder.split("/")[:2]) + "/" + "random_img_" + str(WIDTH) + ".npy"
    if (os.path.isfile(r_img_path)):
        random_img = np.load(r_img_path)
    else:
        random_img = np.random.rand(1,3,WIDTH,HEIGHT)
        np.save(r_img_path, random_img)

    random_img = torch.FloatTensor((random_img - m_nrm) / s_nrm).to(device)

    ## GET WRAPPED MODEL
    wrp_model = copy.deepcopy(model)
    wrapper = model_canonized()


    if 'PPNET' in title:
        wrp_model = PRPCanonizedModel(wrp_model)
    elif 'RESNET' in title:
        wrapper.copyfromresnet(
            wrp_model,
            wrp_model,
            lrp_params_def1,
            lrp_layer2method,
            model.DATASET
            )
    else:
        wrapper.copyfrommodel(
            wrp_model,
            wrp_model,
            lrp_params_def1,
            lrp_layer2method,
            model.DATASET
            )

    wrp_model.eval()


    for cnt,si in enumerate(tqdm(selec_images)):
        # preprocessing
        img,lbl = dataset[si]
        img = img[None, :].to(device)
        lbl = int(lbl)

        proba, sim_scores = model_fwd(title, model, img, lbl, lrp=True)

        kh = proba.argmax(-1)
        pred_orig = proba[0, kh].item() + 1e-8
        true_orig = proba[0, lbl].item() +1e-8


        ## get explanations

        prp_img = get_pixel_exp_prototype(title, wrp_model, img, range(lbl*nb_prt,(lbl+1)*nb_prt))

        for k in range(nb_prt):
            sim_orig = sim_scores[0, k].item() + 1e-8

            mask = np.zeros((WIDTH*HEIGHT))
            flat = np.abs(prp_img[k]).flatten()
            morf = flat.argsort()

            r = .5

            # True Image
            mask *= 0
            mask[morf[:int((WIDTH * HEIGHT)*r)]] = 1
            i_mask = torch.tensor(mask.reshape((1,WIDTH,HEIGHT)),dtype=torch.float32).to(device)

            masked_img = random_img*(1-i_mask) + img*i_mask

            proba_mask, sim_scores_mask = model_fwd(title, model, masked_img, lbl, lrp=True)

            sim_mask = sim_scores_mask[0, k].item()

            ad_sim[cnt,k] = max(0, sim_orig - sim_mask) / sim_orig
            ai_sim[cnt,k] = int(sim_orig < sim_mask)


    if 'kmeans_class' in title:
        model.kmc_ad_sim = torch.tensor(ad_sim).to(device)
        model.kmc_ai_sim = torch.tensor(ai_sim).to(device)
    else:
        model.ad_sim = torch.tensor(ad_sim).to(device)
        model.ai_sim = torch.tensor(ai_sim).to(device)


    print( 'sim ad', ad_sim.mean(), 'ai',ai_sim.mean() )

    return



def get_pixel_exp_prototype(
    title,
    wrp_model,
    img,
    prt_list,
    ):

    if len(img.shape) != 4:
        img = img[None,:]

    img.requires_grad = True

    sim_rule = sim.apply

    prp_images = np.zeros( [len(prt_list), *list(img.shape)[-2:]] )

    for i, k in enumerate(prt_list):
        with torch.enable_grad():
            if ('kmeans' in title):
                prt = wrp_model.kmc_prt
            else:
                prt = wrp_model.prt
            if ('PPNET' in title):
                prt = wrp_model.prt
                if ('kmeans' in title):
                    prt = wrp_model.kmc_prt
                ebd = wrp_model.conv_features(img).squeeze(2).squeeze(2)
            elif ('PROTOVAE' in title):
                ebd = wrp_model.encoder(img)
                ebd = ebd[:, :wrp_model.LATENT]
            else:
                ebd = wrp_model.encoder(img)

            similarity_scores = sim_rule(ebd, prt[prt_list[i],:])


        with torch.enable_grad():
            similarity_scores[0,:].backward(torch.ones_like(similarity_scores[0,:]))

        with torch.no_grad():
            rel = img.grad.data

            hm = rel.squeeze().detach().cpu().numpy().sum(0)
            hm = hm / np.abs(hm).max()

            prp_images[i] = hm

    return prp_images


##########################################################################

from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def diversity(
        title,
        model,
        loader,
        is_train=True):
    print('\n# Compute Diversity')

    RATIO = np.zeros((7, 4, 5, 2))
    for d, DATASET in enumerate([model.DATASET]):
        print('')

        for a, ARCHITECTURE in enumerate(['RESNET34'][:]):
            for r in range(1):
                RUN = str(r)
                if ARCHITECTURE == 'RESNET34':
                    RUN = 'r' + RUN

                print(DATASET.upper() + ' ' * (9 - len(DATASET)), ARCHITECTURE.upper() + ' ' * (18 - len(ARCHITECTURE)),
                      RUN.upper(), sep='\t', end='\t')

                ROOT_FLD = 'saved_models/' + DATASET.upper() + '/' + ARCHITECTURE.upper() + '/' + RUN + '/'

                model = torch.load(ROOT_FLD + 'model/' + ARCHITECTURE + ':' + RUN + '.model').to(device)
                model.eval()

                if ARCHITECTURE == 'RESNET34':
                    sim = model.clf(model.prt)
                    sim = torch.repeat_interleave(sim, model.H, 1)
                elif 'FLINT' in ARCHITECTURE:
                    sim = model.s(model.s.weight)
                elif 'PPNET' in ARCHITECTURE:
                    sim = torch.cdist(model.prt, model.prt) ** 2
                    sim = torch.log(sim + 1) - torch.log(sim + 1e-4)
                elif 'PROTOVAE' in ARCHITECTURE:
                    sim = torch.cdist(model.prt, model.prt)
                    sim = torch.log(sim + 1) - torch.log(sim + 1e-4)

                kim = torch.cdist(model.kmc_prt, model.kmc_prt)
                kim = torch.log(kim + 1) - torch.log(kim + 1e-4)

                sx = torch.softmax(sim, -1)
                e = -(torch.log(sx) * sx).sum(-1) / np.log(model.NB_PRT)
                e = torch.nan_to_num(e, 1).mean()

                ksx = torch.softmax(kim, -1)
                ke = -(torch.log(ksx) * ksx).sum(-1) / np.log(model.NB_PRT)
                ke = torch.nan_to_num(ke, 1).mean()

                RATIO[d, a, r, 0] = e
                RATIO[d, a, r, 1] = ke

                print(*["{:.3f}".format(x) for x in RATIO[d, a, r, :].T.flatten()[:]], sep='\t')
