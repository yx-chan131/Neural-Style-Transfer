import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate(anim_lst, output_path):
    to_img = transforms.ToPILImage()
    fig = plt.figure()
    ims = []    
    print('Making Animation ...')
    for img in anim_lst:
        img = to_img(img.squeeze(0)) 
        ims.append([plt.imshow(img, animated=True)])    
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000)
    print('Animation Produced!')
    plt.show()
    ani.save(output_path, writer=animation.PillowWriter(2))
    print('Animation saved to {}'.format(output_path))