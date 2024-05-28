import os,sys,time,copy

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

import utils.common as cf
import utils.evaluation as ev
import utils.builders as bd

from datetime import datetime
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 8 if torch.cuda.is_available() else 0


# book keeping namings and code
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      type=str)
parser.add_argument('--model',        type=str,   choices=['RESNET34', 'DENSENET121', 'VGG16', 'PROTOVAE','PROTOVAE_RESNET34', 'PPNET_RESNET34','FLINT_RESNET34'])
parser.add_argument('--run',          type=str,   default='x')
parser.add_argument('--load',        action='store_true')

args = parser.parse_args()


##########################################################################

ROOT_FLD = 'saved_models/'
if not os.path.exists(ROOT_FLD):
    os.mkdir(ROOT_FLD)


DATASET = args.dataset
print("dd",DATASET.upper(),"bb")
if not os.path.exists(ROOT_FLD+DATASET.upper()):
    os.mkdir(ROOT_FLD+DATASET.upper())
ROOT_FLD = ROOT_FLD+DATASET.upper()+'/'


ARCHITECTURE = args.model
print("mm",ARCHITECTURE.upper(),"mm")
if not os.path.exists(ROOT_FLD+ARCHITECTURE.upper()):
    os.mkdir(ROOT_FLD+ARCHITECTURE.upper())
ROOT_FLD = ROOT_FLD+ARCHITECTURE.upper()+'/'


RUN = args.run
print("rr",RUN,"rr")
if not os.path.exists(ROOT_FLD+RUN):
    os.mkdir(ROOT_FLD+RUN)
ROOT_FLD = ROOT_FLD+RUN+'/'

for subfolder in ['model','plot','umap']:
    if not os.path.exists(ROOT_FLD + subfolder):
        os.mkdir(ROOT_FLD + subfolder)


TITLE = ARCHITECTURE+":"+RUN
KM_TITLE = ARCHITECTURE+"_kmeans:"+RUN


EPOCH = 10
if DATASET in ['cifar10','stl10','quickdraw']:
    EPOCH = 30
elif DATASET == 'tiny':
    EPOCH = 100
if 'PROTOVAE' in ARCHITECTURE:
    EPOCH = EPOCH*2

##########################################################################
model = nn.Module().to(device)
model.DATASET = DATASET
model.title = TITLE

if DATASET == 'tiny':
    model.WIDTH, model.HEIGHT, model.CHANNEL = 224, 224, 3
    model.K, model.H, model.NB_PRT = 200, 5, 1000
if DATASET == "cub200":
    model.WIDTH, model.HEIGHT, model.CHANNEL = 224, 224, 3
    model.K, model.H, model.NB_PRT = 200, 10, 2000
elif DATASET == 'celeba':
    model.WIDTH, model.HEIGHT, model.CHANNEL = 64, 64, 3
    model.K, model.H, model.NB_PRT = 2, 20, 40
elif DATASET == 'stl10':
    model.WIDTH, model.HEIGHT, model.CHANNEL = 96, 96, 3
    model.K, model.H, model.NB_PRT = 10, 5, 50
else:
    model.WIDTH, model.HEIGHT, model.CHANNEL = 32, 32, 3
    model.K, model.H, model.NB_PRT = 10, 5, 50
bd.build_model(TITLE,model)
cf.train_loader,cf.push_loader,cf.test_loader = bd.build_loader(DATASET, 128)

if args.load:
    print('\n# Load the model')
    try:
        print(ROOT_FLD+'/model/'+TITLE+'.model')
        model = torch.load(ROOT_FLD+'/model/'+TITLE+'.model',map_location=torch.device(device))
    except:
        print('!! Trained model not existing !!')
        sys.exit(1)
else:

    if 'PROTOVAE' in ARCHITECTURE:
        OPTIM = 'Adam|.01|.5'
        WGT = [1,.1,1,.1]
    else:
        OPTIM='SGD|.01|.5'
        if DATASET == 'celeba':
            OPTIM='SGD|.001|.9'
        WGT = None

    print('oo',OPTIM,'oo')
    print('ww',WGT,'ww')

    model.acc = cf.train(
        TITLE,
        model,
        OPTIM,
        EPOCH,
        WGT
        )

    torch.save( obj=model, f=ROOT_FLD+'/model/'+TITLE+'.model' )

print("Accuracy of black-box: ", model.acc)
model.eval()

embed,label = cf.get_embedding(model)

## Get Kmeans-class
model.kmc_title = ARCHITECTURE+"_kmeans_class:"+RUN
cf.get_kmeans_class(model, embed, label)

torch.save( obj=model, f=ROOT_FLD+'/model/'+TITLE+'.model' )

cf.plot_prt_img(model,ROOT_FLD,True)


## Computing diversity
ev.diversity(TITLE,model,cf.test_loader,False)


## Computing robustness of explanations
# ev.AI_AD(
#    TITLE,
#    model,
#    ROOT_FLD,
#    cf.test_loader,
#    selec_images
#    )
#
# model.kmc_title = ARCHITECTURE+"_kmeans_class:"+RUN
# ev.AI_AD(
#    model.kmc_title,
#    model,
#    ROOT_FLD,
#    cf.test_loader,
#    selec_images
#    )
