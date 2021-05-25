import copy
import torch
import torch.nn as nn
import torchvision.models as models

from image_loader import image_loader
from loss import ContentLoss, StyleLoss

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1 ,1, 1)
    
    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_style_model_and_losses(model, style_img, content_img, device,
                               normalization_mean = [0.485, 0.456, 0.406],
                               normalization_std = [0.229, 0.224, 0.225],
                               content_layers = ['conv_4'],
                               style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    """Create a new model from pretrained model by adding content loss and style loss layers.
    Parameters
    ----------
    model(torchvision model): pretrained model. In NST paper VGG19 is used.
    style_img (tensor): style image
    content_img (tensor): content image
    device (str): device to run model 
    normalization_mean (list): default mean of VGG networks
    normalization_std (list): default standard deviation of VGG networks
    content_layers (list): add content loss after the convolutional layers are detected
    style_layers (list):  add style loss after the convolutional layers are detected
    """
    cnn = model.features.to(device).eval()
    cnn = copy.deepcopy(cnn) # for more information, refer https://www.programiz.com/python-programming/shallow-deep-copy

    # normalization module
    normalization_mean = torch.tensor(normalization_mean).to(device)
    normalization_std = torch.tensor(normalization_std).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.sequential to put in 
    # modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0 # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss 
            # and StyleLoss we insert below. So we replace with out-of-place 
            # ones here. (Not really understanding this...)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            # add style loss
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    model = model[:(i+1)]    

    return model, style_losses, content_losses

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    model = models.vgg19(pretrained=True)
    style_img = image_loader('images\picasso.jpg', device)
    content_img = image_loader('images\dancing.jpg', device)
    get_style_model_and_losses(model, style_img, content_img, device)