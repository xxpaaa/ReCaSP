
import torch.nn as nn
import torch
import torch.nn.functional as F

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


def trusted_loss(evidence, y, c, class_num, global_step, annealing_step, reduction='sum'):
    labels_dict = {}
    censor_num = 2
    for i in range(censor_num):
        for j in range(class_num):
            labels_dict[str(i) + "," + str(j)] = i*class_num + j
    # labels_dict = {"0,0": 0, "0,1": 1, "0,2": 2, "0,3": 3, "1,0": 4, "1,1": 5, "1,2": 6, "1,3": 7}

    class_num = class_num*censor_num

    y = y.type(torch.int64) # ground truth bin, 1,2,...,k
    c = c.type(torch.int64) #censorship status, 0 or 1

    # 检查是否为标量
    if y.dim() == 0:
        y = y.unsqueeze(0)
    # 检查是否为标量
    if c.dim() == 0:
        c = c.unsqueeze(0)

    # print("labels_dict: ", labels_dict) #labels_dict:  {'0,0': 0, '0,1': 1, '0,2': 2, '0,3': 3, '1,0': 4, '1,1': 5, '1,2': 6, '1,3': 7}
    # print("y.shape: ", y.shape) #y.shape:  torch.Size([32])
    # print("c.shape: ", c.shape) #c.shape:  torch.Size([32])

    labels_indices = [labels_dict[str(int(c_item)) + "," + str(int(y_item))] for c_item, y_item in zip(c, y)]
    labels_indices = torch.tensor(labels_indices).to(y.device)

    alpha = evidence + 1
    # print("alpha.shape: ", alpha.shape, alpha)
    loss = ce_loss(labels_indices, alpha, class_num, global_step, annealing_step)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


class TrustedSurvLoss(object): #这里结合 https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py, https://github.com/mahmoodlab/SurvPath/blob/main/utils/loss_func.py和https://github.com/hanmenghan/TMC/blob/main/TMC%20ICLR/model.py改的
    """
        The Cross-Entropy Loss function for the discrete time to event model (Zadeh and Schmid, 2020).

        ----------
    """
    def __init__(self, alpha=0.0, beta=0.0, eps=1e-7, reduction='sum', survLoss_type='nll_surv'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        self.survLoss_type = survLoss_type
        self.beta = beta

    def __call__(self, h, evidence, y, t, c, class_num, global_step, annealing_step):
        #class_num=2, global_step=current_epoch, annealing_step可自己设置，eg, 1, 50
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """
        if self.survLoss_type == 'nll_surv':
            loss1 = nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)
        loss2 = self.beta * trusted_loss(evidence, y, c, class_num, global_step, annealing_step, reduction=self.reduction)
        loss = loss1 + loss2
        return loss, loss1, loss2


class NLLSurvLoss(nn.Module): #from https://github.com/mahmoodlab/SurvPath/blob/main/utils/loss_func.py
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)

# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h) #hazard function
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)

    # print("S_padded.shape", S_padded.shape, S_padded)


    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss



from torch.autograd import Variable
class DQNCOSLoss(nn.Module):
    def __init__(self):
        super(DQNCOSLoss, self).__init__()

    def forward(self, input):
        batch_size = input.size(0)
        target = Variable(torch.LongTensor(range(batch_size))).to(input.device)
        loss = 0
        loss += nn.CrossEntropyLoss()(input, target)
        loss += nn.CrossEntropyLoss()(input.transpose(1, 0), target)
        return loss / 2