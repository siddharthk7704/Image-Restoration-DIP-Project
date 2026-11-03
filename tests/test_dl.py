import numpy as np
import pytest

torch = pytest.importorskip("torch")
from models import build_model
from models.dncnn import DnCNN
from models.losses import SobelGradLoss


def test_dncnn_shape_residual_and_receptive_field():
    net = DnCNN(depth=5, width=8)
    x = torch.rand(2, 1, 40, 37)
    assert net(x).shape == x.shape
    assert torch.allclose(net(x), x - net.predict_noise(x))   # restored = noisy - noise
    assert net.receptive_field == 11
    assert build_model("dncnn", in_ch=1, depth=4, width=8)(x).shape == x.shape


def test_sobel_loss_zero_and_ramp_and_backprop():
    loss = SobelGradLoss(1)
    x = torch.rand(2, 1, 16, 16)
    assert loss(x, x).item() == 0.0
    ramp = torch.arange(16.0).view(1, 1, 1, 16).repeat(1, 1, 16, 1)      # slope 1 per pixel along x
    gx = loss.gradients(ramp)[0, 0]
    assert torch.allclose(gx[2:-2, 2:-2], torch.ones(12, 12), atol=1e-5)  # Sobel/8 of a unit ramp = 1
    p = torch.rand(2, 1, 16, 16, requires_grad=True)
    loss(p, x).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


def test_sobel_matches_scipy_interior():
    nd = pytest.importorskip("scipy.ndimage")
    x = np.random.RandomState(1).rand(20, 20).astype(np.float32)
    ours = SobelGradLoss(1).gradients(torch.from_numpy(x)[None, None])[0]
    ref_x = nd.sobel(x, axis=1) / 8.0
    ref_y = nd.sobel(x, axis=0) / 8.0
    assert np.allclose(ours[0].numpy()[2:-2, 2:-2], ref_x[2:-2, 2:-2], atol=1e-5)
    assert np.allclose(ours[1].numpy()[2:-2, 2:-2], ref_y[2:-2, 2:-2], atol=1e-5)


def test_denoise_dl_contract_with_config(tmp_path):
    from dl_infer import denoise_dl
    net = build_model("dncnn", in_ch=1, depth=4, width=8)
    ck = tmp_path / "m.pth"
    torch.save(dict(model=net.state_dict(), arch="dncnn", arch_cfg=dict(in_ch=1, depth=4, width=8), step=0), ck)
    x = np.random.RandomState(0).rand(33, 41).astype(np.float32)
    x0 = x.copy()
    y = denoise_dl(x, {"ckpt": str(ck), "device": "cpu"})          # shared contract: (image, model_config)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    assert y.min() >= 0 and y.max() <= 1 and np.array_equal(x, x0)
