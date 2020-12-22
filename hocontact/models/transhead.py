import torch
import torch.nn as nn

from hocontact.utils.logger import logger


class TransHead(nn.Module):
    def __init__(self, base_neurons=[515, 256], out_dim=3):
        super().__init__()

        if out_dim != 3 and out_dim != 6:
            logger.error(f"Unrecognized transhead out dim: {out_dim}")
            raise ValueError()

        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, out_dim)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        return out


def recover_3d_proj(objpoints3d, camintr, est_scale, est_trans, off_z=0.4, input_res=(128, 128)):
    """
    Given estimated centered points, camera intrinsics and predicted scale and translation
    in pixel world, compute the point coordinates in camera coordinate system
    """
    # Estimate scale and trans between 3D and 2D
    focal = camintr[:, :1, :1]
    batch_size = objpoints3d.shape[0]
    focal = focal.view(batch_size, 1)
    est_scale = est_scale.view(batch_size, 1)
    est_trans = est_trans.view(batch_size, 2)
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2]
    img_centers = (cam_centers.new(input_res) / 2).view(1, 2).repeat(batch_size, 1)
    est_XY0 = (est_trans + img_centers - cam_centers) * est_Z0 / focal
    est_c3d = torch.cat([est_XY0, est_Z0], -1).unsqueeze(1)  #TENSOR(B, 1, 3)
    recons3d = est_c3d + objpoints3d
    return recons3d, est_c3d


def test():
    trans_head = TransHead()
    x = torch.rand((16, 2048))
    x = trans_head(x)

if __name__ == '__main__':
    test()
