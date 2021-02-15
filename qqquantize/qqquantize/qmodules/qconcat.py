import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QConcat']

class QConcat(nn.Module):
    """There is not _FLOAT_MODULE because it may come from outsides"""
    def __init__(self, dim, qconfig):
        super().__init__()
        self.d = dim
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
    
    def forward(self, tensors):
        qtensors = []
        for x in tensors:
            qx = self.act_quant(x)
            qtensors.append(qx)
        return torch.cat(qtensors, dim=self.d)

    @classmethod
    def from_float(cls, mod):
        qconfig = mod.qconfig
        qconcat = cls(mod.d, qconfig)
        return qconcat
