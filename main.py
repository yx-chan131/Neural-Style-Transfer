import argparse
import torch
import torchvision.models as models

from image_loader import image_loader, show_image
from animate import animate
from optimizer import get_input_optimizer
from model import get_style_model_and_losses

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Neural Style Transfer")

    parser.add_argument('--content_img', default='images\dancing.jpg',
                        help='Path to content image')
    parser.add_argument('--style_img', default='images\picasso.jpg',
                        help='Path to style image')
    parser.add_argument('--use_noise', default=False,
                        help='Use white noise as input image')
    parser.add_argument('--num_steps', default=300, type=int, help='Number of iterations')
    parser.add_argument('--style_weight', default=1000000, type=int,
                        help='Weighting factor for style reconstruction')
    parser.add_argument('--content_weight', default=1, type=int,
                        help='Weighting factor for content reconstruction')
    parser.add_argument('--output_path', default='images\output_img.png')
    parser.add_argument('--save_anim', default=False, help='Save training process as animation')
    args = parser.parse_args()
    return args

def get_input_img(content_img, device, use_noise=False):
    """Select the input image. Can use a copy of the content image or white noise.
    content_img (tensor): content image
    use_noise (bool): choose whether use content image or white noise as input image
    """
    if use_noise:
        input_img = torch.randn(content_img.data.size(), device=device)        
    else:
        input_img = content_img.clone() 
    return input_img

def run_style_transfer(args):
    """ Run the style transfer """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Building the style transfer model..')
    model = models.vgg19(pretrained=True)

    style_img = image_loader(args.style_img, device)
    content_img = image_loader(args.content_img, device)
    input_img = get_input_img(content_img, device, use_noise=args.use_noise)
    anim_lst = []

    model, style_losses, content_losses = get_style_model_and_losses(model, 
                                                                    style_img, 
                                                                    content_img, device)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = 0    
    
    while run < args.num_steps:
        def closure():
            """
            Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate
            the function multiple times, so you have to pass in a closure that allows them to 
            recompute your model. The closure should clear the gradients, compute the loss, and
            return it. (Refer to https://pytorch.org/docs/stable/optim.html)

            Update the parameters by optimizer.step(closure)
            """   
            input_img.data.clamp_(0, 1) # correct the values of updated input image

            nonlocal run # enable rebinding of a nonlocal name
                         # https://stackoverflow.com/questions/2609518/unboundlocalerror-with-nested-function-scopes
            if args.save_anim and run%10==0:
                anim_lst.append(input_img.cpu().clone()) 

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            style_score *= args.style_weight
            content_score *= args.content_weight

            total_loss = style_score + content_score     
            total_loss.backward()
            
            run += 1
            if run % 25 == 0:
                print("run {}".format(run))
                print("Style Loss: {:4f} Content Loss: {:4f}".format(
                    style_score.item(), content_score.item()
                ))
                print()

            return total_loss
        
        optimizer.step(closure)

    input_img.data.clamp_(0, 1) 
        
    if args.save_anim:   
        anim_lst.append(input_img.cpu().clone())        
        animate(anim_lst, 'images\output.gif')
    show_image(input_img, title='Output Image', save=True, save_path=args.output_path)



if __name__ == "__main__":
    args = get_args()
    run_style_transfer(args)
    
