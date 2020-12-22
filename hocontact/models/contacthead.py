import torch
import torch.nn as nn

from hocontact.models.pointnet import PointNetEncodeModule, PointNetDecodeModule


class ContactHead(nn.Module):
    def __init__(self, out_dim, base_neurons=None):
        super().__init__()
        if base_neurons is None:
            base_neurons = [65, 512, 512, 512]
        assert len(base_neurons) >= 1

        # returns k for each object vert
        self.out_dim = out_dim

        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Conv2d(inp_neurons, out_neurons, kernel_size=(1, 1), stride=1, padding=0))
            layers.append(nn.ReLU())
        self.final_layer = nn.Conv2d(out_neurons, self.out_dim, kernel_size=(1, 1), stride=1, padding=0)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        return out


class VertexContactHead(nn.Module):
    def __init__(self, base_neurons=None, out_dim=1):
        super().__init__()
        if base_neurons is None:
            base_neurons = [65, 512, 512, 65]
        assert len(base_neurons) >= 1

        # returns k for each object vert
        self.out_dim = out_dim

        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Conv2d(inp_neurons, out_neurons, kernel_size=(1, 1), stride=1, padding=0))
            layers.append(nn.ReLU())
        self.final_layer = nn.Conv2d(out_neurons, self.out_dim, kernel_size=(1, 1), stride=1, padding=0)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        # // out = self.sigmoid(out)
        return out


class PointNetContactHead(nn.Module):
    def __init__(self, feat_dim=65, n_region=17, n_anchor=4):
        super().__init__()

        # record input feature dimension
        self.feat_dim = feat_dim

        # returns k for each object vert
        self.n_region = n_region
        self.n_anchor = n_anchor

        # encode module
        self.encoder = PointNetEncodeModule(self.feat_dim)
        self._concat_feat_dim = self.encoder.dim_out
        self.vertex_contact_decoder = PointNetDecodeModule(self._concat_feat_dim, 1)
        self.contact_region_decoder = PointNetDecodeModule(self._concat_feat_dim + 1, self.n_region)
        self.anchor_elasti_decoder = PointNetDecodeModule(self._concat_feat_dim + 17, self.n_anchor)

    def forward(self, inp):
        # inp = TENSOR[NBATCH, 65, NPOINT, 1]
        batch_size, _, n_point, _ = inp.shape
        feat = inp.squeeze(3)  # TENSOR[NBATCH, 65, NPOINT]
        concat_feat = self.encoder(feat)  # TENSOR[NBATCH, 4992, NPOINT]
        vertex_contact = self.vertex_contact_decoder(concat_feat)  # TENSOR[NBATCH, 1, NPOINT]
        contact_region = self.contact_region_decoder(concat_feat, vertex_contact)  # TENSOR[NBATCH, 17, NPOINT]
        anchor_elasti = self.anchor_elasti_decoder(concat_feat, contact_region)  # TENSOR[NBATCH, 4, NPOINT]
        # post process
        vertex_contact = vertex_contact.squeeze(1).contiguous()
        contact_region = contact_region.transpose(1, 2).contiguous()  # TENSOR[NBATCH, NPOINT, 17]
        anchor_elasti = anchor_elasti.transpose(1, 2).contiguous()  # TENSOR[NBATCH, NPOINT, 4]

        # !: here sigmoid is compulsory, since we use bce (instead of  bce_with_logit)
        anchor_elasti = torch.sigmoid(anchor_elasti)
        return vertex_contact, contact_region, anchor_elasti
