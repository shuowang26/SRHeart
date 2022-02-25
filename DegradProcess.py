import torch
from torch import nn
import torch.nn.functional as F


class MotionDegrad(nn.Module):
    """ Custom MRI downsmapling + motion  """
    def __init__(self, newD=13, newH=128, newW=128, mode='bilinear'):
        super(MotionDegrad, self).__init__()

        self.newD, self.newH, self.newW = newD, newH, newW
        self.mode = mode

        self.s_H = nn.Parameter(torch.zeros(newD).float())
        self.s_W = nn.Parameter(torch.zeros(newD).float())

        self.params = nn.ParameterDict({
            's_H': self.s_H,
            's_W': self.s_W,
        })


    def forward(self, batch_x):

        assert (batch_x.size()[0] == 1)  # deal with one volume in a batch

        device = batch_x.device

        D, H, W = batch_x.size()[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, self.newD), torch.linspace(-1, 1, self.newH), torch.linspace(-1, 1, self.newW)])  # (d, h, w)

        offset_d = grid_d.to(device)
        offset_h = grid_h.to(device) + self.params['s_H'].view(-1, 1, 1).expand(self.newD, self.newH, self.newW)
        offset_w = grid_w.to(device) + self.params['s_W'].view(-1, 1, 1).expand(self.newD, self.newH, self.newW)

        # caution: the stack order of x,y,z instead of d,h,w
        offsets = torch.stack((offset_w, offset_h, offset_d), -1).unsqueeze(0)

        batch_degrad = F.grid_sample(batch_x, offsets, mode=self.mode, align_corners=True)

        return batch_degrad


class MultiViewTrans(nn.Module):
    """ Project one view to another view  """
    def __init__(self, vol_s, affine_s, vol_t, affine_t, mode='bilinear'):
        super(MultiViewTrans, self).__init__()

        # test code
        Ds, Hs, Ws = vol_s.shape[-3:]
        Dt, Ht, Wt = vol_t.shape[-3:]

        R_s, b_s = affine_s[0:3, 0:3], affine_s[0:3, -1]
        R_t, b_t = affine_t[0:3, 0:3], affine_t[0:3, -1]

        A = torch.inverse(R_s).matmul(R_t)
        b = torch.inverse(R_s).matmul((b_t - b_s).reshape((3,1)))

        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(0, Dt-1, Dt), torch.linspace(0, Ht-1, Ht), torch.linspace(0, Wt-1, Wt)])  # (d, h, w)
        ijk = torch.cat((grid_d.reshape(1,-1), grid_h.reshape(1,-1), grid_w.reshape(1,-1)), dim=0)
        IJK = A.matmul(ijk) + b
        offset_D, offset_H, offset_W = IJK[0,:].reshape(grid_d.shape), IJK[1,:].reshape(grid_h.shape), IJK[2,:].reshape(grid_w.shape)

        offset_D, offset_H, offset_W = offset_D/Ds*2-1, offset_H/Hs*2-1, offset_W/Ws*2-1

        # caution: the stack order of x,y,z instead of d,h,w
        offsets = torch.stack((offset_W, offset_H, offset_D), -1).unsqueeze(0)

        self.offsets = offsets
        self.mode = mode
        self.Ds, self.Hs, self.Ws = Ds, Hs, Ws
        self.Dt, self.Ht, self.Wt = Dt, Ht, Wt


    def forward(self, batch_x):

        assert (batch_x.size()[0] == 1)  # deal with one volume in a batch

        device = batch_x.device
        batch_LA = F.grid_sample(batch_x, self.offsets, mode=self.mode, align_corners=False)

        return batch_LA




# degradation process
def batch_degrade(batch_x, new_D, s_H, s_W, mode='bilinear'):

    assert(batch_x.size()[0] ==1) # deal with one volume in a batch
    D, H, W = batch_x.size()[-3:]
    grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, new_D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])  # (h, w)
    grid_d = grid_d.cuda().float()
    grid_h = grid_h.cuda().float()
    grid_w = grid_w.cuda().float()


    offset_d = grid_d
    offset_h = grid_h + s_H.view(-1, 1, 1).expand(new_D, H, W)
    offset_w = grid_w + s_W.view(-1, 1, 1).expand(new_D, H, W)

    # caution: the stack order of x,y,z instead of d,h,w
    offsets = torch.stack((offset_w, offset_h, offset_d), -1).unsqueeze(0)

    HR_ds = F.grid_sample(batch_x, offsets, mode=mode, align_corners=True)

    return HR_ds




