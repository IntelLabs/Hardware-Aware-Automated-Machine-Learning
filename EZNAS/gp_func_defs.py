import numpy as np
import torch
import torchvision.transforms as transforms
import yaml

'''
For analyzing any layer, we have the following data shapes probably available
                    ForwardPass          BackwardPass
InpActivation      (b, c, ha, wa)        (c, ha, wa)
Weight             (c, d, hw, ww)       (c, d, hw, ww)
PreActivation      (b, d, hp, wp)        (d, hp, wp)
Activation         (b, d, hp, wp)        (d, hp, wp)

     InpAct       --> Layer_Weight       -->  PreAct -->     ActFunc -->     Act
 (b, c, ha, wa)      (c, d, hw, ww)         (b, d, hp, wp)              (b,d, hp, wp)

Thus, there are 8 statistics to capture for the following scenarios:
Data, Noise, DataPerturbation --> 7*3 + 1 (weights same for all cases) --> 22 static addresses
Batch size == B, to be decided later.

There are going to be 3 tensor shapes and one scalar shape.
    --> weight   shape : Weight themselves, and weight gradients
    --> inpact   shape : Input activation and corresponding gradient
    --> pre/act  shape : Pre activation, activation and corresponding gradient
    --> all scalars

Keep memory addressing as:
[1 empty] [22 static]  [20 wtaddr] [20 inpaddr] [20 preactaddr] [20 actaddr] [20 scalar] : 123 addresses

static addr: [wt, wtgraddata, wtgradnoise, wtgraddataperturb,
            inpactdata, inpactnoise, inpactdataperturb, inpactgraddata, inpactgradnoise, inpactgraddataperturb,
            preactdata, preactnoise, preactdataperturb, preactgraddata, preactgradnoise, preactgraddataperturb,
            actdata, actnoise, actdataperturb, actgraddata, actgradnoise, actgraddataperturb]
'''

with open("evol_config.yaml", "r") as f:
    config = yaml.load(f)

rangemix=config['rangemix']
LEN_IND_ADDR = config['LEN_IND_ADDR']
wt_addr_range = [x+1 for x in (list(range(4)) + list(range(22, 22+LEN_IND_ADDR, 1)))]
inpact_addr_range = [x+1 for x in (list(range(4, 10, 1)) + list(range(22+LEN_IND_ADDR, 22+2*LEN_IND_ADDR, 1)))]
preact_addr_range = [x+1 for x in (list(range(10, 16, 1)) + list(range(22+2*LEN_IND_ADDR, 22+3*LEN_IND_ADDR, 1)))]
act_addr_range = [x+1 for x in (list(range(16, 22, 1)) + list(range(22+3*LEN_IND_ADDR, 22+4*LEN_IND_ADDR, 1)))]
activation_superrange = inpact_addr_range[:6]+preact_addr_range[:6]+act_addr_range[:6] + inpact_addr_range[6:]+preact_addr_range[6:]+act_addr_range[6:] 
if (rangemix):
    inpact_addr_range = activation_superrange
    preact_addr_range = activation_superrange
    act_addr_range    = activation_superrange
scalar_addr_range = [x+1 for x in (list(range(22+4*LEN_IND_ADDR, 22+5*LEN_IND_ADDR, 1)))]


'''
Generate shape preservation list of all operators
0 : shape not preserved (scalar output)
1 : shape preserved
Arg dim matching checker
0: Irrelevant (Single arg)
1: Read spaces must match
'''
# In case of @ operator, since read spaces must match, it is assumed that shape is preserved
# in case of transpose, since all tensors have square submatrices, we assume shape is preserved
op_preserve_dim = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 
                        7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1,
                        14:1, 15:1, 16:1, 17:1, 18:0, 19:0,
                        20:0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0,
                        26: 0, 27:0, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34:0, 35: 1}
op_argmatch_dim = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:0, 
                    7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0,
                    14:0, 15:0, 16:0, 17:0, 18:0, 19:0,
                    20:0, 21: 0, 22: 0, 23: 1, 24: 0, 25: 1,
                    26: 1, 27: 1,  28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34:0, 35: 0}
# Indicates whether read addrs can be scalar or not.
op_inparg_scalar_ok = {0:1, 1:1, 2:0, 3:1, 4:1, 5:1, 6:0, 
                    7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1,
                    14:1, 15:1, 16:1, 17:1, 18:1, 19:0,
                    20:0, 21:0, 22:0, 23: 1, 24: 1, 25: 1,
                    26:1, 27:0, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34:1, 35: 1}
'''
Functions to be added:


'''

OPLIST = ['eltwise_sum', 'eltwise_prod', 'matmul', 
          'lt', 'gt', 'eq', 'AddNoise', 'Log',
          'AbsLog', 'Abs', 'Power', 'Exp', 'Normalize',
          'ReLU', 'Sign', 'Heaviside', 'elt_invert',
          'normal_init', 'frob_norm', 'det', 'logdet',
          'sym_eigen', 'eigen', 'sum', 'l1norm', 'hammingdist',
          'kldiv', 'cosinesimilarity', 'softmax', 'sigmoid', 'ones_like', 'zeros_like',
          'gt0', 'lt0', 'numel', 'elwise_sub']


addressing_dict = {"Wt_addr": wt_addr_range, "InpAct_addr": inpact_addr_range,
                   "PreAct_addr": preact_addr_range, "Act_addr": act_addr_range,
                   "Scalar_addr": scalar_addr_range}
############################## Double Arg Operators #####################
# sum
def OP0(A, B):
    return A+B

# EltwiseProduct
def OP1(A, B):
    return A*B

# MatMul
def OP2(A, B):
    if len(A.shape)==0:
        A = A.unsqueeze(0)
        
    if len(B.shape)==0:
        B = B.unsqueeze(0)
    return A@B

# <
def OP3(A, B):
    return (A<B).float()

# >
def OP4(A, B):
    return (A>B).float()

# ==
def OP5(A, B):
    return (A==B).float()

############################## Single Arg Operators #####################

def OP6(A):
    return A + (0.1**0.5)*torch.randn(A.shape)

# Log
def OP7(A):
    A[A<=0] = 1
    return torch.log(A)

# AbsLog
def OP8(A):
    A[A==0] = 1
    return torch.abs(torch.log(A))

# Abs
def OP9(A):
    return torch.abs(A)

# Power
def OP10(A):
    return torch.pow(A, 2)

# Exp 
def OP11(A):
    return torch.exp(A)

# Normalize to 0-1
def OP12(A):
    mean, std = A.mean(), A.std()
    z = (A - mean)/std
    z[z!=z] = 0
    return z

### Eigenvalue skipped?

# ReLU
def OP13(A):
    return torch.functional.F.relu(A, inplace=False)

# Sign
def OP14(A):
    return torch.sign(A)

# Heaviside
def OP15(A):
    return torch.heaviside(A, values=torch.Tensor([0]).float().cuda())

# elt-wise invert
def OP16(A):
    return 1/A

### InverseMat left out

# Gaussian Init
def OP17(A):
    return torch.normal(mean=0, std=1, size=A.shape)

# Uniform Init removed

############################## Scalar Single Arg Operators #####################

# Frobenius norm
def OP18(A):
    return torch.norm(A.float(), p='fro')
### L1, L0 norm not added yet


# Determinant
def OP19(A):
    z = torch.det(A)
    return z

# LogDeterminant
def OP20(A):
    z = torch.logdet(A)
    z[z!=z] = 0
    return z
    
# Symmetrized EigenValue first/last ratio
def OP21(A):
    if len(A.shape)!=0:
        sh = A.shape[0]
        A = A.reshape(sh, -1)
        A = A @ A.t()
        A = A + A.t()
        e,v = torch.symeig(A, eigenvectors=False)
        return e[-1]/e[0]
    else:
        return torch.Tensor([0])

# Square Tensor EigenValues first/last ratio
def OP22(A):
    if len(A.shape)!=0:
        lsize = A.shape[0]
        # Flatten, stack many batches, einsum 
        # Typically meant for grads with many batches?
        A = A.reshape(lsize, -1)
        A = torch.einsum('nc,mc->nm', [A, A])
        e,v = torch.eig(A)
        return (e[-1]/e[0])[0]
    else:
        return torch.Tensor([0])

# Normalize sum
def OP23(A):
    return torch.sum(A)/A.numel()

# L1 norm
def OP24(A):
    return torch.sum(abs(A))/A.numel()

############################## Scalar Double Arg Operators #####################

# Hamming distance
def OP25(A, B):
    a_bin = torch.heaviside(A.flatten(), values=torch.Tensor([0]).float())
    b_bin = torch.heaviside(B.flatten(), values=torch.Tensor([0]).float())
    return sum(a_bin!=b_bin).float()

# KL Div
def OP26(A, B):
    return torch.nn.KLDivLoss(reduction='batchmean')(A, B).float()

# Cosine Similarity
def OP27(A, B):
    if len(A.shape)>0:
        bs = A.shape[0]
        A = A.reshape(bs, -1)
        B = B.flatten(bs, -1)
        return torch.sum(torch.nn.CosineSimilarity()(A,B)).float()
    else:
        return torch.Tensor([0])


def OP28(A):
    return torch.functional.F.softmax(A)
    
def OP29(A):
    return torch.functional.F.sigmoid(A)

def OP30(A):
    return torch.ones_like(A)
    
def OP31(A):
    return torch.zeros_like(A)

def OP32(A):
    return (A>0).float()

def OP33(A):
    return (A<0).float()

def OP34(A):
    return torch.Tensor([A.numel()])

# sum
def OP35(A, B):
    return A-B
        