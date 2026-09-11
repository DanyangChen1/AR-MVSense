import torch

from lightrppg.losses import NegativePearsonLoss
from lightrppg.models import LightRPPGViT, RPPGViT, parameter_count


def test_model_shapes_and_backward() -> None:
    video = torch.randn(2, 16, 3, 32, 32)
    target = torch.randn(2, 16)
    models = [
        RPPGViT(dim=32, heads=4, cca_depth=1, swta_depth=2, window_size=8, spatial_tokens=4, dropout=0),
        RPPGViT(
            dim=32,
            heads=4,
            cca_depth=1,
            swta_depth=2,
            window_size=8,
            spatial_tokens=4,
            dropout=0,
            cca_variant="figure2",
        ),
        LightRPPGViT(dim=32, rep_depth=1, tasa_depth=1, dropout=0),
    ]
    for model in models:
        prediction = model(video)
        assert prediction.shape == target.shape
        NegativePearsonLoss()(prediction, target).backward()


def test_reparameterization_equivalence() -> None:
    torch.manual_seed(1)
    model = LightRPPGViT(dim=32, rep_depth=2, tasa_depth=1, dropout=0).eval()
    video = torch.randn(1, 16, 3, 32, 32)
    with torch.inference_mode():
        expected = model(video)
        actual = model.deployed_copy()(video)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_default_light_model_matches_reported_parameter_budget() -> None:
    model = LightRPPGViT().eval()
    assert abs(parameter_count(model) - 700_000) / 700_000 < 0.01
    assert abs(parameter_count(model.deployed_copy()) - 700_000) / 700_000 < 0.01
