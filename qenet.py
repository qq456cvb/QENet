import torch
import time
import torch.nn.functional as F
import numpy as np


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def quat_dist(a, b):
    # print(torch.sum(a * b, -1).max(), torch.sum(a * b, -1).min())
    res = 2 * torch.acos(torch.clamp(torch.sum(a * b, -1), -1. + 1e-7, 1. - 1e-7))
    assert not torch.any(torch.isnan(res))
    return res


def quat_prod(a, b):
    target_size = torch.max(torch.tensor(a.size()), torch.tensor(b.size()))
    a = a.expand(*target_size)
    b = b.expand(*target_size)
    return torch.cat([a[..., :1] * b[..., :1] - torch.sum(a[..., 1:] * b[..., 1:], -1, keepdim=True),
                      a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:])], -1)


def quat_avg(quats, weights=None, power_iteration=2):
    
    if weights is not None:
        assert not torch.any(torch.isnan(quats))
        assert not torch.any(torch.isnan(weights))
        # print(weights.max(), weights.min())
        cov = quats.transpose(-1, -2) @ torch.diag_embed(weights) @ quats
    else:
        cov = quats.transpose(-1, -2) @ quats
    
    # ident = torch.eye(4).reshape([*np.ones((len(cov.shape)-2,), dtype=np.int), 4, 4]) * 1e-4
    # cov += ident.expand_as(cov).cuda()
    # t = time.time()
    shape = cov.size()
    u = F.normalize(torch.normal(torch.zeros(shape[:-1])).cuda(), dim=-1)[..., None]
    v = F.normalize(torch.normal(torch.zeros(shape[:-1])).cuda(), dim=-1)[..., None]
    
    for _ in range(power_iteration):
        assert not torch.any(torch.isnan(cov))
        assert not torch.any(torch.isnan(u))
        v = F.normalize(cov.transpose(-1, -2) @ u, dim=-2)
        u = F.normalize(cov @ v, dim=-2)
    
    v = v[..., 0]

    # v = F.normalize(cov[..., 0], dim=-1)
    # print(time.time() - t)
    # _, _, v_ref = torch.svd(cov)
    # v_ref = v_ref[..., 0]
    # print(time.time() - t, torch.mean(1. - torch.abs(torch.sum(v * v_ref, -1))))
    assert not torch.any(torch.isnan(v))

    return v


def qedr(input_t, input_capsule, input_a, k=3):
    '''

    :param input_t: B x N x K x Nc x M x 4
    :param input_capsule: B x N x K x Nc x 4
    :param input_a: B x N x K x Nc
    :param k: iterations
    :return:
    '''
    shape = input_t.size()
    assert not torch.any(torch.isnan(input_capsule))
    assert not torch.any(torch.isnan(input_t))
    votes = quat_prod(input_capsule[..., None, :], input_t)  # B x N x K x Nc x M x 4
    assert not torch.any(torch.isnan(votes))
    votes = votes.permute([0, 1, 4, 2, 3, 5]).reshape([-1, shape[1], shape[4], shape[2] * shape[3], 4])  # B x N x M x KNc x 4
    activation = input_a.reshape([-1, shape[1], 1, shape[2] * shape[3]])  # B x N x 1 x KNc
    assert not torch.any(torch.isnan(activation))

    output_capsule = quat_avg(votes, activation)  # B x N x M x 4
    for _ in range(k):
        assignment = activation * torch.sigmoid(-quat_dist(output_capsule[..., None, :], votes))  # B x N x M x KNc
        output_capsule = quat_avg(votes, assignment)  # B x N x M x 4
    output_activation = torch.sigmoid(-torch.sum(quat_dist(output_capsule[..., None, :], votes), -1))
    return output_capsule, output_activation


class QENet(torch.nn.Module):
    def __init__(self, transform_units: list, input_channels, output_channels):
        super().__init__()
        transform_units.append(output_channels * 4)
        transform_units.insert(0, 3)

        nets = []
        for i in range(len(transform_units) - 1):
            nets.append(torch.nn.Linear(transform_units[i], transform_units[i + 1]))

        self.transform_net = torch.nn.Sequential(*nets)
        self.transform_net.apply(init_weights)
        self.output_channels = output_channels

    def forward(self, input_x, input_capsule, input_a):
        '''
        Quaternion Equivariant Network
        :param input_x: B x N x K x 3
        :param input_capsule: B x N x K x Nc x 4
        :param input_a: B x N x K x Nc
        '''
        shape = input_capsule.size()
        miu = quat_avg(input_capsule.permute([0, 1, 3, 2, 4]))  # B x N x Nc x 4
        x = torch.cat([torch.zeros([shape[0], shape[1], shape[2], 1]).cuda(), input_x], -1)  # B x N x K x 4

        x = quat_prod(quat_prod(torch.cat([miu[..., :1], -miu[..., 1:]], -1)[..., None, :, :], x[..., None, :]), 
            miu[..., None, :, :])  # we need conjugate for rotation

        # x should have 0 on real axis
        x = x[..., 1:]  # B x N x K x Nc x 3
        assert not torch.any(torch.isnan(x))
        # x = x[..., 1:].reshape([-1, shape[1], shape[2], shape[3] * 3])  # B x N x K x Nc3
        t = self.transform_net(x).reshape([-1, shape[1], shape[2], shape[3], self.output_channels, 4])  # B x N x K x Nc x M x 4
        t = F.normalize(t, dim=-1)
        
        output_capsule, output_activation = qedr(t, input_capsule, input_a)
        assert not torch.any(torch.isnan(output_activation))
        return output_capsule, output_activation 


if __name__ == "__main__":
    # mat = torch.zeros([3, 3]).normal_(0, 1).cuda()
    # u = F.normalize(torch.normal(torch.zeros([3,])).cuda(), dim=-1)[..., None]
    # v = F.normalize(torch.normal(torch.zeros([3,])).cuda(), dim=-1)[..., None]
    
    # for _ in range(0):
    #     v = F.normalize(mat.transpose(-1, -2) @ u, dim=-2)
    #     u = F.normalize(mat @ v, dim=-2)
    
    # v = v[..., 0]
    # _, _, v_ref = torch.svd(mat)
    # cos_err_v = 1.0 - torch.abs(torch.dot(v, v_ref[:, 0])).item()
    # print(cos_err_v)

    qenet = QENet([64, 64], 1, 64).cuda()
    x = torch.zeros([2, 64, 9, 3]).fill_(1).cuda()
    a = torch.zeros([2, 64, 9, 1]).fill_(1).cuda()
    cap = torch.zeros([2, 64, 9, 1, 4]).fill_(1).cuda()
    out_cap, out_a = qenet(x, cap, a)
    print(out_cap.shape, out_a.shape)



