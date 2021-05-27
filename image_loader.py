import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

def image_loader(img, device, imsize=(512, 512)):
    img_transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    img = Image.open(img)
    tensor = img_transform(img).unsqueeze(0)
    return tensor.to(device, torch.float)
    
def show_image(tensor, title=None, save=False, save_path=None):    
    to_img = transforms.ToPILImage()
    img = to_img(tensor.cpu().clone().squeeze(0))    
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    if save:
        plt.savefig(save_path, bbox_inches='tight')    
    plt.show()

if __name__ == "__main__":
    tensor1 = image_loader(".\images\dancing.jpg")
    tensor2 = image_loader(".\images\picasso.jpg")
    show_image(tensor1, "dancing.jpg")
    show_image(tensor2, "picasso.jpg")