import numpy as np
import matplotlib.pyplot as plt
from utils.lrp_general6 import *
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0



class  Modulenotfounderror(Exception):
  pass

class model_canonized():

    def __init__(self):
        super(model_canonized, self).__init__()
    # runs in your current module to find the object layer3.1.conv2, and replaces it by the obkect stored in value (see         success=iteratset(self,components,value) as initializer, can be modified to run in another class when replacing that self)
    def setbyname(self, model, name, value):

        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                # print('found!!', components[0])
                # exit()
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split('.')
        success = iteratset(model, components, value)
        return success

    def copyfrommodel(self, title, model, net, lrp_params, lrp_layer2method):
        if 'RESNET' in title:
            return self.copyfromresnet(model, net, lrp_params, lrp_layer2method)
        elif 'VGG' in title:
            return self.copyfromvgg(model, net, lrp_params, lrp_layer2method)
        elif 'DENSENET' in title:
            return self.copyfromdensenet(model, net, lrp_params, lrp_layer2method)
        else:
            return self.copyfromcnn(model, net, lrp_params, lrp_layer2method)

    def copyfromresnet(self, model, net, lrp_params, lrp_layer2method):
        # assert( isinstance(net,ResNet))

        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation dependent

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():

            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                # m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                # store conv2d layers
                last_src_module_name = src_module_name
                last_src_module = src_module
            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # conv-bn chain
                foundsth = True

                if (True == lrp_params['use_zbeta']) and (last_src_module_name == 'encoder.conv1'):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(m, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta, dataset=model.DATASET)


                if False == self.setbyname(model,last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + last_src_module_name + " in target net to copy")

                updated_layers_names.append(last_src_module_name)

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)

                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)


        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in model.named_modules():

            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + target_module_name + " in target net , impossible!")
                updated_layers_names.append(target_module_name)




    def copyfromvgg(self, model, net, lrp_params, lrp_layer2method):
        # assert( isinstance(net,ResNet))

        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation dependent

        updated_layers_names = []

        for src_module_name, src_module in net.named_modules():

            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                # m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                if (True == lrp_params['use_zbeta']) and (src_module_name == 'encoder.features.0'):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                # m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(src_module, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta,dataset=model.DATASET)

                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy")

                updated_layers_names.append(src_module_name)

            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # bn chain

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(src_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)


        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in model.named_modules():

            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + target_module_name + " in target net , impossible!")
                updated_layers_names.append(target_module_name)



    def copyfromcnn(self, model, net, lrp_params, lrp_layer2method):
        # assert( isinstance(net,ResNet))

        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation dependent

        updated_layers_names = []

        for src_module_name, src_module in net.named_modules():

            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                # m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                if (True == lrp_params['use_zbeta']) and (src_module_name == 'encoder.0'):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                # m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(src_module, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta,dataset=model.DATASET)

                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy")

                updated_layers_names.append(src_module_name)

            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # bn chain

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(src_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)


        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in model.named_modules():

            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + target_module_name + " in target net , impossible!")
                updated_layers_names.append(target_module_name)




    def copyfromdensenet(self, model, net, lrp_params, lrp_layer2method):

        # use_zbeta=lrp_params['use_zbeta']
        # linear_eps=lrp_params['linear_eps']
        # pooling_eps=lrp_params['pooling_eps']
        # conv2d_ignorebias= lrp_params['conv2d_ignorebias']

        name_prev2 = None
        mod_prev2 = None

        name_prev1 = None
        mod_prev1 = None

        updated_layers_names = []
        for name, mod in net.named_modules():

            # print('curchain:', name_prev2, name_prev1, name)

            # treat the first conv in the NN and its subsequent BN layer
            if name == 'encoder.features.norm0':  # fuse first conv with subsequent BatchNorm layer

                # print('trying to update ', 'features.norm0', 'features.conv0')
                if name_prev1 != 'encoder.features.conv0':
                    raise Modulenotfounderror('name_prev1 expected to be features.conv0, but found:' + name_prev1)
                #
                if (True == lrp_params[
                    'use_zbeta']):  # condition for whether its the first conv, actually already detected
                    # by the if name=='features.norm0': clause above
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                conv = bnafterconv_overwrite_intoconv(conv=copy.deepcopy(mod_prev1), bn=mod)
                # wrap conv
                wrapped = get_lrpwrapperformodule(conv, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta,dataset=model.DATASET)

                success = self.setbyname(model, name='encoder.features.conv0', value=wrapped)  # was value = conv
                if False == success:
                    raise Modulenotfounderror(' could not find ', 'features.conv0')

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(mod), lrp_params, lrp_layer2method)

                success = self.setbyname(model, name='encoder.features.norm0', value=wrapped)  # was value = resetbn(mod)
                if False == success:
                    raise Modulenotfounderror(' could not find ', 'features.norm0')

                updated_layers_names.append('encoder.features.conv0')
                updated_layers_names.append('encoder.features.norm0')

            elif name == 'encoder.classifier':  # fuse densenet head, which has a structure
            #     # BN(norm5)-relu(toprelu)-adaptiveAvgPool(toppool)-linear
            #     print('trying to update ', 'classifier', 'features.norm5', 'toprelu')
            #
            #     if name_prev1 != 'encoder.features.norm5':
            #         # if that fails, run an inner loop to get 'features.norm5'
            #         raise Modulenotfounderror('name_prev1 expected to be features.norm5, but found:' + name_prev1)
            #
            #     # approach:
            #     #    BN(norm5)-relu(toprelu)-adaptiveAvgPool(toppool)-linear('classifier')
            #     # = threshrelu - BN - adaptiveAvgPool(toppool)-linear
            #     # = threshrelu - adaptiveAvgPool(toppool) - BN  -linear # yes this should commute bcs of no zero padding!
            #     # = threshrelu - adaptiveAvgPool(toppool) - fusedlinear with tensorbias
            #     # = resetbn(BN) -  threshrelu/clamplayer(toprelu) -  adaptiveAvgPool(toppool) - fusedlinear with tensorbias
            #
            #     # get the right threshrelu/clamplayer
            #     threshrelu = getclamplayer(mod_prev1)
            #     wrapped = get_lrpwrapperformodule(threshrelu, lrp_params, lrp_layer2method)
            #
            #     # success = self.setbyname(model, name='encoder.toprelu', value=wrapped)
            #     # if False == success:
            #     #     raise Modulenotfounderror(' could not find ', 'toprelu')
            #
            #     # get the right linearlayer with tensor bias
            #     linearlayer_with_biastensor = linearafterbn_returntensorbiasedlinearlayer(linearlayer=mod, bn=mod_prev1)
            #     wrapped = get_lrpwrapperformodule(linearlayer_with_biastensor, lrp_params, lrp_layer2method)
            #
            #     success = self.setbyname(model, name='encoder.classifier', value=wrapped)
            #     if False == success:
            #         raise Modulenotfounderror(' could not find ', 'features.classifier')
            #
            #     # resetbn(BN)
                wrapped = get_lrpwrapperformodule(resetbn(mod_prev1), lrp_params, lrp_layer2method)
                success = self.setbyname(model, name='encoder.features.norm5', value=wrapped)
                if False == success:
                    raise Modulenotfounderror(' could not find ', 'features.norm5')

                # no need to touch the pooling
                updated_layers_names.append('encoder.classifier')
                updated_layers_names.append('encoder.features.norm5')
                # updated_layers_names.append('encoder.toprelu')



            elif 'conv' in name:

                if name == 'encoder.features.conv0':
                    name_prev2 = name_prev1
                    mod_prev2 = mod_prev1

                    name_prev1 = name
                    mod_prev1 = mod

                    continue

                # print('trying to update ', name_prev2, name_prev1, name)

                # bn-relu-conv chain

                # print('shapes?', mod_prev2.weight.shape, mod.weight.shape)

                if not isinstance(mod_prev2, nn.BatchNorm2d):
                    print('error: no bn at the start, ', name_prev2, name_prev1, name)
                    exit()
                if not isinstance(mod_prev1, nn.ReLU):
                    print('error: no relu in the middle, ', name_prev2, name_prev1, name)
                    exit()

                # approach:
                #    BN-relu-conv
                # =  threshrelu/clamplayer-BN-conv
                # =  threshrelu/clamplayer-(fused conv with tensorbias) # the bias is tensorshaped
                #         with difference in spatial dimensions, whenever zero padding is used!!
                # = resetbn(BN)- threshrelu/clamplayer- (fused conv with tensorbias)

                # print('trying to update BN-relu-conv chain: ', name_prev2, name_prev1, name)

                # bn-relu-conv chain

                if not isinstance(mod_prev2, nn.BatchNorm2d):
                    print('error: no bn at the start, ', name_prev2, name_prev1, name)
                    exit()
                if not isinstance(mod_prev1, nn.ReLU):
                    print('error: no relu in the middle, ', name_prev2, name_prev1, name)
                    exit()

                # get the right threshrelu/clamplayer
                threshrelu = getclamplayer(bn=mod_prev2)
                wrapped = get_lrpwrapperformodule(threshrelu, lrp_params, lrp_layer2method)
                # success = self.setbyname(name= name_prev2 ,value =  zeroparam_wrapper_class( clampl2 , relu_wrapper_fct() ) )
                success = self.setbyname(model, name=name_prev2, value=wrapped)
                if False == success:
                    raise Modulenotfounderror(' could not find ', name_prev2)

                # get the right convolution, likely with tensorbias
                convm2 = convafterbn_returntensorbiasedconv(conv=mod, bn=mod_prev2)
                '''
                if isinstance(convm2, tensorbiased_convlayer ):
                  wrappedconv = oneparam_wrapper_class( convm2 ,  tensorbiasedconv2d_beta0_wrapper_fct(), parameter1 =  conv2d_ignorebias  )
                else:
                  assert(  isinstance(convm2, torch.nn.Conv2d ) )
                  wrappedconv =  oneparam_wrapper_class(convm2, conv2d_beta0_wrapper_fct() , parameter1 =  conv2d_ignorebias  )
                '''
                wrapped = get_lrpwrapperformodule(convm2, lrp_params, lrp_layer2method)
                success = self.setbyname(model, name=name, value=wrapped)
                if False == success:
                    raise Modulenotfounderror(' could not find ', name)

                # reset batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(mod_prev2), lrp_params, lrp_layer2method)
                success = self.setbyname(model, name=name_prev1, value=wrapped)
                if False == success:
                    raise Modulenotfounderror(' could not find ', name_prev1)

                updated_layers_names.append(name)
                updated_layers_names.append(name_prev1)
                updated_layers_names.append(name_prev2)

            else:
                pass

            # read
            name_prev2 = name_prev1
            mod_prev2 = mod_prev1

            name_prev1 = name
            mod_prev1 = mod

        '''
        if isinstance(mod, nn.ReLU):
          if False== self.setbyname(name, zeroparam_wrapper_class(nn.ReLU(),relu_wrapper_fct()) ):
            raise Modulenotfounderror("could not find module "+name+ " in target net to copy" )            
          updated_layers_names.append(name)
        '''
        for target_module_name, target_module in model.named_modules():

            # wrap other layers: relu, adaptiveavgpool2d, avgpool2d
            if isinstance(target_module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.ReLU)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                success = self.setbyname(model, target_module_name, value=wrapped)
                if False == success:
                    raise Modulenotfounderror(' could not find ', target_module_name)
                updated_layers_names.append(target_module_name)

            '''
            if isinstance(target_module, nn.AdaptiveAvgPool2d):
                foundsth=True
                print('is nn.AdaptiveAvgPool2d')
                if False== self.setbyname(target_module_name,  oneparam_wrapper_class(target_module, adaptiveavgpool2d_wrapper_fct(), parameter1 = pooling_eps ) ):
                  raise Modulenotfounderror("could not find module "+target_module_name+ " in target net to copy" )            
                updated_layers_names.append(target_module_name) 
  
            elif isinstance(target_module, nn.AvgPool2d):
                foundsth=True
                print('is nn.AvgPool2d')
                if False== self.setbyname(target_module_name,  oneparam_wrapper_class(mod, avgpool2d_wrapper_fct(), parameter1 = pooling_eps ) ):
                  raise Modulenotfounderror("could not find module "+target_module_name+ " in target net to copy" )            
                updated_layers_names.append(target_module_name)  
            # wrap relu not overwritten in targetclass
            if isinstance(target_module, nn.ReLU ):
              if False== self.setbyname(target_module_name, zeroparam_wrapper_class(nn.ReLU(),relu_wrapper_fct()) ):
                raise Modulenotfounderror("could not find module "+target_module_name+ " in target net , impossible!" )            
              updated_layers_names.append(target_module_name)
            '''

        # print('not updated ones:')
        # for target_module_name, target_module in model.named_modules():
        #     if target_module_name not in updated_layers_names:
        #         print('not updated:', target_module_name)



### Save heatmaps overlayed on original images
def imshow_im(hm,imgtensor,q=100,folder="folder", folder_orig="orig", name="name"):

    def invert_normalize(ten, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
      # print(ten.shape)
      s=torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
      m=torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

      res=ten*s+m
      return res

    def showimgfromtensor(inpdata):

      ts=invert_normalize(inpdata)
      a=ts.data.squeeze(0).numpy()
      saveimg=(a*255.0).astype(np.uint8)

    hm = hm.squeeze().detach().numpy()
    clim = np.percentile(np.abs(hm), q)
    hm = hm / clim


    makedir(folder+"/")
    plt.imsave(folder + name, hm, cmap="seismic", vmin=-1, vmax=+1)

    ### OVERLAY FINAL
    heatmap = np.array(Image.open(folder+name).convert('RGB'))
    heatmap = np.float32(heatmap) / 255
    ts = invert_normalize(imgtensor.squeeze())
    a = ts.data.numpy().transpose((1, 2, 0))
    makedir(folder_orig + "/")
    plt.imsave(folder_orig + name,
               a,
               vmin=0,
               vmax=+1.0)
    overlayed_original_img_j = 0.2 * a + 0.6 * heatmap
    plt.imsave(folder+name,
               overlayed_original_img_j,
               vmin=-1,
               vmax=+1.0)


## Generating protoypical explanations for each prototypes for 100 test images.
def generate_explanations(test_loader,model,prototypes,n_prototypes,write_path, write_path_orig,epsilon):
    model.eval()

    def x_prp(test_loader,write_path, write_path_orig,epsilon):
        im = 0
        for data in test_loader:
        # for data in itertools.islice(test_loader, stop=100):
            # get the inputs
            inputs = data[0]
            labels = data[1]
            # d = torch.cdist(zx_mean, model.module.prototype_vectors, p=2)


            inputs = inputs.to(device)
            inputs.requires_grad = True

            with torch.enable_grad():
                zx_mean = model(inputs)
                zx_mean = zx_mean[:, :latent]
                p_vector = prototypes[pno,:]
                d = (zx_mean-p_vector)**2
                R_zx = 1/(d+epsilon)
                R_zx.backward(torch.ones_like(R_zx))
                rel = inputs.grad.data

                # print(write_path+'/prototype'+str(pno)+'/'+str(labels.item())+"-"+str(im)+"-PRP.png")
                imshow_im(rel.to('cpu'), imgtensor=inputs.to('cpu'), folder=write_path+'/prototype'+str(pno)+'/', folder_orig = write_path_orig,name=str(labels.item())+"-"+str(im)+"-PRP.png")
                im += 1
                if(im==100):
                    return


    for pno in range(n_prototypes):
        print("Protoype: ", pno)
        print("Saving LRP maps for 100 test images in ", write_path+'/prototype'+str(pno)+'/...')
        x_prp(test_loader,write_path, write_path_orig,epsilon)




class addon_canonized(nn.Module):

    def __init__(self):
        super(addon_canonized, self).__init__()
        self.addon = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.Sigmoid()
        )


def _addon_canonized(pretrained=False, progress=True, **kwargs):
    model = addon_canonized()
    # if pretrained:
    #     raise Cannotloadmodelweightserror("explainable nn model wrapper was never meant to load dictionary weights, load into standard model first, then instatiate this class from the standard model")
    return model


### Save heatmaps overlayed on original images
def imshow_im(hm,imgtensor,q=100,folder="folder", folder_orig="orig", name="name"):

    def invert_normalize(ten, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
      # print(ten.shape)
      s=torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
      m=torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

      res=ten*s+m
      return res

    def showimgfromtensor(inpdata):

      ts=invert_normalize(inpdata)
      a=ts.data.squeeze(0).numpy()
      saveimg=(a*255.0).astype(np.uint8)

    hm = hm.squeeze().detach().numpy()
    clim = np.percentile(np.abs(hm), q)
    hm = hm / clim


    makedir(folder+"/")
    plt.imsave(folder + name, hm, cmap="seismic", vmin=-1, vmax=+1)

    ### OVERLAY FINAL
    heatmap = np.array(Image.open(folder+name).convert('RGB'))
    heatmap = np.float32(heatmap) / 255
    ts = invert_normalize(imgtensor.squeeze())
    a = ts.data.numpy().transpose((1, 2, 0))
    makedir(folder_orig + "/")
    plt.imsave(folder_orig + name,
               a,
               vmin=0,
               vmax=+1.0)
    overlayed_original_img_j = 0.2 * a + 0.6 * heatmap
    plt.imsave(folder+name,
               overlayed_original_img_j,
               vmin=-1,
               vmax=+1.0)


## Generating protoypical explanations for each prototypes for 100 test images.
def generate_explanations(test_loader,model,prototypes,n_prototypes,write_path, write_path_orig,epsilon):
    model.eval()

    def x_prp(test_loader,write_path, write_path_orig,epsilon):
        im = 0
        for data in test_loader:
        # for data in itertools.islice(test_loader, stop=100):
            # get the inputs
            inputs = data[0]
            labels = data[1]
            # d = torch.cdist(zx_mean, model.module.prototype_vectors, p=2)


            inputs = inputs.to(device)
            inputs.requires_grad = True

            with torch.enable_grad():
                zx_mean = model(inputs)
                zx_mean = zx_mean[:, :latent]
                p_vector = prototypes[pno,:]
                d = (zx_mean-p_vector)**2
                R_zx = 1/(d+epsilon)
                R_zx.backward(torch.ones_like(R_zx))
                rel = inputs.grad.data

                # print(write_path+'/prototype'+str(pno)+'/'+str(labels.item())+"-"+str(im)+"-PRP.png")
                imshow_im(rel.to('cpu'), imgtensor=inputs.to('cpu'), folder=write_path+'/prototype'+str(pno)+'/', folder_orig = write_path_orig,name=str(labels.item())+"-"+str(im)+"-PRP.png")
                im += 1
                if(im==100):
                    return


    for pno in range(n_prototypes):
        print("Protoype: ", pno)
        print("Saving LRP maps for 100 test images in ", write_path+'/prototype'+str(pno)+'/...')
        x_prp(test_loader,write_path, write_path_orig,epsilon)




def setbyname(obj, name, value):

    def iteratset(obj, components, value):

        if not hasattr(obj, components[0]):
            print(components[0])
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            return True
        else:
            nextobj = getattr(obj, components[0])
            return iteratset(nextobj, components[1:], value)

    components = name.split('.')
    success = iteratset(obj, components, value)
    return


def PRPCanonizedModel(ppnet):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wrapper = model_canonized()

    wrapper.copyfromresnet(ppnet.features, ppnet.features, lrp_params=lrp_params_def1, lrp_layer2method=lrp_layer2method, dataset=ppnet.DATASET)

    conv_layer1 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(device)
    conv_layer1.weight.data = ppnet.ones

    wrapped = get_lrpwrapperformodule(copy.deepcopy(conv_layer1), lrp_params_def1, lrp_layer2method)
    conv_layer1 = wrapped

    conv_layer2 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(device)
    conv_layer2.weigh



### LRP parameters
lrp_params_def1={
    'conv2d_ignorebias': True,
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': True ,
  }

lrp_layer2method={
'nn.ReLU':          relu_wrapper_fct,
'nn.Sigmoid':          sigmoid_wrapper_fct,
'nn.BatchNorm2d':   relu_wrapper_fct,
'nn.Conv2d':        conv2d_beta0_wrapper_fct,
'nn.Linear':        linearlayer_eps_wrapper_fct,
'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
'nn.MaxPool2d': maxpool2d_wrapper_fct,
'nn.AvgPool2d': avgpool2d_wrapper_fct,
'nn.Identity': identity_wrapper_fct,
'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
'clamplayer': relu_wrapper_fct,
'tensorbiased_linearlayer': tensorbiased_linearlayer_eps_wrapper_fct,
'tensorbiased_convlayer': tensorbiasedconv2d_beta0_wrapper_fct,
}
