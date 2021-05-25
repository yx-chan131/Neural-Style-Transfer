import torch.optim as optim

def get_input_optimizer(input_img):
    # input is a parameter that requires a gradient
    optimizer = optim.LBFGS(input_img.requires_grad())
    return optimizer