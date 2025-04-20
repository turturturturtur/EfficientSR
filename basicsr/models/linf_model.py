import torch
from basicsr.models.sr_model import SRModel
from basicsr.archs.linf_arch import LinfSRNet
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.flow_arch import Flow
from basicsr.archs.unet_arch import UNet

from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class LinfSRModel(SRModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = LinfSRNet(
            encoder=EDSR(n_resblocks=16, n_feats=64, no_upsampling=True),
            flow=Flow(flow_layers=10),
            unet=UNet()
        )
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        l_pix = self.cri_pix(self.output, self.gt)
        l_pix.backward()
        self.optimizer_g.step()
