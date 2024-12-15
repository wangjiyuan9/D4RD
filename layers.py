
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    disp = 1 / (depth + 1e-5)
    p_disp = (disp - min_disp) / (max_disp - min_disp)
    p_disp[depth <= 0] = 0
    p_disp[p_disp <= 0] = 0

    return p_disp


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class WavConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, norm_layer=None, use_refl=False):
        super(WavConvBlock, self).__init__()

        if kernel_size == 3:
            self.conv = Conv3x3(in_channels, out_channels, use_refl=use_refl)  # 3*3的卷积，padding=1
        elif kernel_size == 1:
            self.conv = Conv1x1(in_channels, out_channels)
        else:
            raise NotImplementedError

        self.nonlin = nn.ELU(inplace=True)
        if norm_layer is not None:
            self.norm_layer = norm_layer(out_channels)
        else:
            self.norm_layer = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm_layer(out)
        out = self.nonlin(out)
        return out

class ConvBlockDepth(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()

        self.conv = DepthConv3x3(in_channels, out_channels)
        self.nonlin = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(DepthConv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1(nn.Module):
    """Conv1x1
    """

    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        return out


##############################################################################################################
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
            requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


##############################################################################################################
'''
I have review the principle of Image-Backprojection and Projection-other-view part code and finish this note.
Hope can help you. You can delete it if you don't need it.
Sorry for Chinese version, but I don't want to translate it. You can do it by youself.
'''

# class BackprojectDepth(nn.Module):
#     """Layer to transform a depth image into a point cloud
#     """
#
#     def __init__(self, height, width):
#         super(BackprojectDepth, self).__init__()
#
#         self.height = height
#         self.width = width
#
#         meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
#
#         self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
#         """
#         mershgrid是个list，需要stack成一个array
#         生成2*H*W的网格，格式为
#         0 1 ... W-1
#         0 1 ... W-1
#         0 1 ... W-1
#         第一维度
#         0 0 ... 0
#         1 1 ... 1
#         ...
#         H-1 H-1 ... H-1
#         第二维度
#         第1维是x坐标,第2维是y坐标
#         """
#         self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
#             requires_grad=False).cuda()
#
#         self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
#             requires_grad=False).cuda()
#         """
#         1*1*（H*W）的全1矩阵
#         """
#         self.pix_coords = torch.unsqueeze(torch.stack(
#             [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
#         """
#         这里的view(-1)是将2*H*W的网格展开成2*H*W的一维向量
#         0 1 ... W-1 0 1 ... W-1 0 1 ... W-1和0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1
#         之后拼接成2*（H*W）的网格
#         [0 1 ... W-1 0 1 ... W-1 0 1 ... W-1
#         0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1]
#         增加第三维度
#         [[0 1 ... W-1 0 1 ... W-1 0 1 ... W-1
#         0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1]]
#         1*2*（H*W）
#         """
#         self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
#             requires_grad=False)
#         """
#         得到1*3*（H*W）的网格，第一个纬度是为了和深度图对应
#         depth 的形状通常是 (B, 1, H, W)。
#         结尾形如：
#         [[0 1 ... W-1 0 1 ... W-1 0 1 ... W-1
#         0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1]
#         [1 1 ... 1 1 1 ... 1 ... 1 1 ... 1]]
#         沿着第二个纬度取可以得到（x,y,1）
#         """
#
#     def forward(self, depth, inv_K):
#         B, _, H, W = depth.shape
#         cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
#         """
#         inv_K[:, :3, :3]是B*3*3的矩阵，self.pix_coords是1*3*（H*W）的网格，K[:, :3, :3]是相机内参:
#         fx 0 cx
#         0 fy cy
#         0 0  1
#         含义是
#         x=1/Z*(fx*X+cx*Z)，y=1/Z*(fy*Y+cy*Z)，1=1/Z*Z,原理可见论文笔记的第二部分
#         改为矩阵形式就是
#         [x,y,1]=1/Z*K*[X,Y,Z]
#         这里认为pix_coords的单体是(x,y,1)，得到pix_coords*inv_K[:, :3, :3]*Z=[X,Y,Z]
#         """
#         cam_points = depth.view(B, 1, H * W) * cam_points  # 这里是把B*H*W→B*1*（H*W）
#         cam_points = torch.cat([cam_points, self.ones.expand(B, -1, -1)], 1)  # 这里是把B*3*（H*W）→B*4*（H*W）
#
#         return cam_points
#
#
# class Project3D(nn.Module):
#     """Layer which projects 3D points into a camera with intrinsics K and at position T
#     """
#
#     def __init__(self, height, width, eps=1e-7):
#         super(Project3D, self).__init__()
#
#         self.height = height
#         self.width = width
#         self.eps = eps
#
#     def forward(self, points, K, T):
#         B, _, HW = points.shape
#
#         P = torch.matmul(K, T)[:, :3, :]
#         """
#         这里的K是相机内参，T是相机外参，P是投影矩阵
#         可以根据3D的像素变化得到为什么可以*T来进行平移或旋转，之后再乘以K就可以得到二维的坐标
#         结果是B*3*4的矩阵
#         这里T是B*4*4的矩阵，K是B*4*4的矩阵
#         T的单体是
#         [1 0 0 +-0.1
#         0 1 0 0
#         0 0 1 0
#         0 0 0 1]
#         说明是在x轴上平移了0.1或者-0.1，代表stereo的两个相机的距离，所以说0.1代表54cm的来源是这里
#         """
#         cam_points = torch.matmul(P, points)
#         pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
#         """
#         point是B*4*（H*W）的矩阵,P是B*3*4的矩阵，所以P*point是B*3*（H*W）的矩阵,3代表x,y,z
#         之后除以z就可以得到x/z,y/z,1,这里只保留了x/z,y/z
#         下面view的作用是把B*2*（H*W）的矩阵变成B*2*H*W的矩阵
#         之后permute的作用是把B*2*H*W的矩阵变成B*H*W*2的矩阵
#
#         """
#         pix_coords = pix_coords.view(-1, 2, self.height, self.width)
#         pix_coords = pix_coords.permute(0, 2, 3, 1)
#         """
#         下面进行归一化，把像素坐标变成[0,1]的坐标
#         最后-0.5*2是为了把坐标变成[-1,1]的坐标
#         其目的是为了与 grid_sample 函数的要求相匹配
#         """
#         pix_coords[..., 0] /= self.width - 1
#         pix_coords[..., 1] /= self.height - 1
#         pix_coords = (pix_coords - 0.5) * 2
#         return pix_coords

##############################################################################################################

def upsample(x, scale_factor=2, mode='nearest'):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def get_smooth_loss(disp, img, gamma=1):
    """Computes the smoothness loss for a disp, The img is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-gamma * grad_img_x)
    grad_disp_y *= torch.exp(-gamma * grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embedder_obj  # , embedder_obj.out_dim




class Gradient_Net(nn.Module):
    def __init__(self, device):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return torch.abs(grad_x), torch.abs(grad_y)
