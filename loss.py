import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    """ Content loss is the squared-error loss between two feature representations
    target (tensor): feature representation of original image in layer l
    input (tensor): feature representation of generated image in layer l

    Backpropagation is performed on Content Loss with respect to the input feature representation.
    Thus we can change the initially random image ("input") until it generates the same response 
    in a certain layer of the CNN as the original image ("target")
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(x):
    """ Gram matrix 
    Gram matrix of a set of vectors v_1, ..., v_n is an inner product space, whose entries are 
    given by G_ij = <v_i, v_j>. If the vectors v_1, ..., v_n are real and the row of matrix
    X, then the Gram matrix is XX' 
    """
    b, c, h, w = x.size()   # b = batch size(=1)
                            # c = number of feature maps
                            # (h, w) = dimensions of a f. map  
    x = x.view(b*c, -1) # size (b*c, h*w)
    G = torch.mm(x, x.t())
    # The gram matrix must be normalized by dividing each element by the total number of
    # elements in the matrix. This normalization is to counteract the fact that "x" matrices
    # with a large h*w dimension yield larger values in the Gram matrix. These larger values
    # will cause the first layers to have a larger impact during the gradient descent. Style
    # features tend to be in the deeper layers of the network so this normalization step is 
    # crucial.
    return G.div(b*c*h*w)

class StyleLoss(nn.Module):
    """ 
    Style loss is the mean square error between gram matrix of target feature representation
    and input feature representation.
    """
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()
    
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



