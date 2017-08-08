from torch_gbds.GenerativeModel import GenerativeModel, LDS
import torch
import numpy as np

def test_generative_model():
    gm = GenerativeModel(None, 5, 10)

    assert gm.Xsamp.size() == torch.Size([5])

def test_LDS():
    mm = LDS({}, 10, 5)

    assert {'x0', 'Q0Chol', 'A', 'RChol', 'QChol', 'Xsamp'}.issubset(dict(mm.named_parameters()).keys())

def test_LDS_sampling():
    mm = LDS({}, 10, 5)

    xx = mm.sampleX(100)
    yy = mm.sampleY()
    [x, y] = mm.sampleXY(100)
    assert xx.size() == torch.Size([100, 10])
    assert yy.size() == torch.Size([5])
    assert x.shape == (100, 10)
    assert y.shape == (100, 5)
