import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# * currently, STN is unused
class STNkd(nn.Module):
    def __init__(self, k=65):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(k, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        device = x.device
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(batchsize, 1)
        )
        iden = iden.to(device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# * currently, regularizer on STN transforms is unused
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    device = trans.device
    I = torch.eye(d, device=device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


class PointNetEncodeModule(nn.Module):
    def __init__(self, dim_in=65):
        super().__init__()
        self.dim_in = dim_in
        # self.stn1 = STNkd(k=channel)
        self.conv1 = torch.nn.Conv1d(dim_in, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        # self.stn2 = STNkd(k=128)
        self.dim_out = 4992

    def forward(self, point_cloud):
        # pointcloud = TENSOR[NBATCH, 65, NPOINT]
        B, D, N = point_cloud.size()

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))  # TENSOR[NBATCH, 128, NPOINT]
        out2 = F.relu(self.bn2(self.conv2(out1)))  # TENSOR[NBATCH, 128, NPOINT]
        out3 = F.relu(self.bn3(self.conv3(out2)))  # TENSOR[NBATCH, 128, NPOINT]
        out4 = F.relu(self.bn4(self.conv4(out3)))  # TENSOR[NBATCH, 512, NPOINT]
        out5 = self.bn5(self.conv5(out4))  # TENSOR[NBATCH, 2048, NPOINT]
        out_max = torch.max(out5, 2, keepdim=True)[0]  # TENSOR[NBATCH, 2048, 1]
        expand = out_max.repeat(1, 1, N)  # TENSOR[NBATCH, 2048, NPOINT]
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)  # TENSOR[NBATCH, 4992, NPOINT]

        return concat


class PointNetDecodeModule(nn.Module):
    def __init__(self, dim_in=4992, dim_out=1):
        super().__init__()
        self.convs1 = torch.nn.Conv1d(dim_in, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, dim_out, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, pointnet_feat, extra_feat=None):
        # pointnet_feat = TENSOR[NBATCH, 4944, NPOINT]
        # extra_feat = TENSOR[NBATCH, 1, NPOINT]
        if extra_feat is not None:
            pointnet_feat = torch.cat((pointnet_feat, extra_feat), dim=1)
        net = F.relu(self.bns1(self.convs1(pointnet_feat)))  # TENSOR[NBATCH, 256, NPOINT]
        net = F.relu(self.bns2(self.convs2(net)))  # TENSOR[NBATCH, 256, NPOINT]
        net = F.relu(self.bns3(self.convs3(net)))  # TENSOR[NBATCH, 128, NPOINT]
        net = self.convs4(net)  # TENSOR[NBATCH, 1, NPOINT]
        return net
