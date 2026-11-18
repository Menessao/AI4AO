"""Tests for AI4AO.DeformableMirror."""
import pytest
import torch

from AI4AO.DeformableMirror import DeformableMirror


# ---------------------------------------------------------------------------
# @property scaling round-trips (see CLAUDE.md: all learnable misreg/DM
# quantities are reparameterized so they sit near unit order of magnitude
# for the optimizer).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr,value", [
    ("rotationAngle", 37.0),
    ("radialScaling", 0.05),
    ("tangentialScaling", -0.03),
    ("anamorphosisAngle", -12.0),
])
def test_scalar_property_round_trip(deformable_mirror, attr, value):
    setattr(deformable_mirror, attr, torch.tensor([value]))
    result = getattr(deformable_mirror, attr)
    assert torch.allclose(result, torch.tensor([value]), atol=1e-4)


def test_sign_property_round_trip(deformable_mirror):
    deformable_mirror.sign = torch.tensor([3.5e-6])
    assert torch.allclose(deformable_mirror.sign, torch.tensor([3.5e-6]), atol=1e-9)


def test_grid_shift_property_round_trip(deformable_mirror):
    value = torch.tensor([[1.5, -2.0]]).unsqueeze(-1).unsqueeze(-1)
    deformable_mirror.grid_shift = value
    assert torch.allclose(deformable_mirror.grid_shift, value, atol=1e-4)


def test_moffat_parameter_round_trip_and_validation(deformable_mirror):
    deformable_mirror.moffatParameter = torch.tensor([3.0])
    assert torch.allclose(deformable_mirror.moffatParameter, torch.tensor([3.0]), atol=1e-4)

    with pytest.raises(ValueError):
        deformable_mirror.moffatParameter = torch.tensor([0.0])
    with pytest.raises(ValueError):
        deformable_mirror.moffatParameter = torch.tensor([-1.0])


def test_mech_coupling_round_trip_and_validation(deformable_mirror):
    deformable_mirror.mechCoupling = torch.tensor([0.4])
    assert torch.allclose(deformable_mirror.mechCoupling, torch.tensor([0.4]), atol=1e-4)

    with pytest.raises(ValueError):
        deformable_mirror.mechCoupling = torch.tensor([0.0])
    with pytest.raises(ValueError):
        deformable_mirror.mechCoupling = torch.tensor([1.0])


def test_apply_and_get_misreg_round_trip(deformable_mirror):
    misreg = {
        "rotationAngle": 15.0,
        "shiftX": 0.01,
        "shiftY": -0.02,
        "radialScaling": 2.0,
        "tangentialScaling": -1.0,
        "anamorphosisAngle": 8.0,
    }
    deformable_mirror.ApplyMisreg(misreg)
    readback, dm_dict = deformable_mirror.GetMisreg()

    assert readback["rotationAngle"] == pytest.approx(misreg["rotationAngle"], abs=1e-3)
    assert readback["shiftX"] == pytest.approx(misreg["shiftX"], abs=1e-3)
    assert readback["shiftY"] == pytest.approx(misreg["shiftY"], abs=1e-3)
    assert readback["radialScaling"] == pytest.approx(misreg["radialScaling"], abs=1e-3)
    assert readback["tangentialScaling"] == pytest.approx(misreg["tangentialScaling"], abs=1e-3)
    assert readback["anamorphosisAngle"] == pytest.approx(misreg["anamorphosisAngle"], abs=1e-3)
    assert dm_dict["FlipLeftRight"] == deformable_mirror.flip_lr
    assert dm_dict["FlipTopBottom"] == deformable_mirror.flip_tb


# ---------------------------------------------------------------------------
# Shapes / actuator grid / IF
# ---------------------------------------------------------------------------

def test_actuator_grid_and_if_shapes(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    Nres = deformable_mirror.Nres

    assert deformable_mirror.actuator_positions.shape == (total_act, 2)
    assert deformable_mirror.IF.shape == (total_act, Nres, Nres)


def test_if_is_zero_mean_over_pupil(deformable_mirror):
    pupil = deformable_mirror.pupil
    if_over_pupil = deformable_mirror.IF[:, pupil]
    assert torch.allclose(
        if_over_pupil.mean(dim=-1), torch.zeros(if_over_pupil.shape[0]), atol=1e-4
    )


def test_get_dm_shape_forward(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    coefs = torch.zeros(2, total_act)
    coefs[0, 0] = 1.0  # single-actuator command, batch 0
    # batch 1 stays all-zero

    shape = deformable_mirror.GetDMShape(coefs)

    assert shape.shape == (2, deformable_mirror.Nres, deformable_mirror.Nres)
    assert torch.allclose(shape[1], torch.zeros_like(shape[1]))
    assert not torch.allclose(shape[0], torch.zeros_like(shape[0]))


# ---------------------------------------------------------------------------
# Save/load round trip
# ---------------------------------------------------------------------------

def test_save_and_load_calibration_round_trip(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    deformable_mirror.rotationAngle = torch.tensor([12.0])
    deformable_mirror.sign = torch.tensor([2e-5])
    # Property setters mutate the raw parameters but do NOT themselves rebuild
    # IF (only forward() in training mode, or an explicit call, does) -- so
    # a real comparison of "did save/load preserve the physical DM state"
    # needs IF rebuilt first, same as LoadCalibration does on the other side.
    deformable_mirror.MakeZonalModes()

    path = tmp_path / "dm.pth"
    deformable_mirror.SaveCalibration(str(path))

    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    fresh.LoadCalibration(str(path))

    misreg_orig, dmdict_orig = deformable_mirror.GetMisreg()
    misreg_loaded, dmdict_loaded = fresh.GetMisreg()

    assert misreg_orig == pytest.approx(misreg_loaded, abs=1e-4)
    # IF is now pure OPD (meters, no wavenumber factor), so its magnitude is
    # set directly by `sign` (~1e-5 m here) -- use a tighter atol than that scale.
    assert torch.allclose(deformable_mirror.IF, fresh.IF, atol=1e-9)


# ---------------------------------------------------------------------------
# Per-actuator calibration (moffatParameter/sign/mechCoupling are always
# length-totalAct vectors; per_actuator_calibration gates only how
# MakeZonalModes uses them -- see Ideas/09-per-actuator-dm-calibration.md).
# ---------------------------------------------------------------------------

def test_scalar_assignment_broadcasts_to_full_vector(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.sign = torch.tensor([4e-6])
    assert deformable_mirror.sign.shape == (total_act,)
    assert torch.allclose(deformable_mirror.sign, torch.full((total_act,), 4e-6), atol=1e-9)


def test_per_actuator_vector_stored_exactly(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    vector = torch.linspace(1.0, 2.0, total_act)
    deformable_mirror.moffatParameter = vector
    assert torch.allclose(deformable_mirror.moffatParameter, vector, atol=1e-4)


def test_wrong_length_vector_raises(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    with pytest.raises(RuntimeError):
        deformable_mirror.sign = torch.ones(total_act + 1)


def test_if_differs_per_actuator_when_flag_true(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = True
    deformable_mirror.sign = torch.linspace(0.8e-5, 1.2e-5, total_act)
    deformable_mirror.MakeZonalModes()

    peak = deformable_mirror.IF.reshape(total_act, -1).abs().max(dim=-1).values
    assert not torch.allclose(peak, peak.mean().expand(total_act), atol=1e-9)


def test_if_depends_only_on_mean_sign_when_flag_false(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = False

    vector = torch.linspace(0.8e-5, 1.2e-5, total_act)
    deformable_mirror.sign = vector
    # Setter stores the non-uniform vector regardless of the flag...
    assert not torch.allclose(deformable_mirror.sign, deformable_mirror.sign.mean().expand(total_act))
    deformable_mirror.MakeZonalModes()
    if_a = deformable_mirror.IF.clone()

    # ...but while per_actuator_calibration is False, only the MEAN feeds into
    # IF -- reassigning a different vector with the same mean (here, the same
    # values under a different actuator-to-value assignment) must leave IF
    # unchanged, since which actuator nominally "owns" which raw value is
    # irrelevant to the collapsed, tied value actually used.
    deformable_mirror.sign = torch.flip(vector, dims=[0])
    deformable_mirror.MakeZonalModes()
    if_b = deformable_mirror.IF.clone()

    assert torch.allclose(if_a, if_b, atol=1e-9)


def test_gradient_independent_per_actuator_when_flag_true(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = True
    deformable_mirror.MakeZonalModes()

    weights = torch.arange(1, total_act + 1, dtype=torch.float32)
    loss = (deformable_mirror.IF.reshape(total_act, -1).pow(2).sum(dim=-1) * weights).sum()
    loss.backward()

    grad = deformable_mirror._sign.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.unique().numel() > 1


def test_gradient_tied_per_actuator_when_flag_false(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = False
    deformable_mirror.MakeZonalModes()

    weights = torch.arange(1, total_act + 1, dtype=torch.float32)
    loss = (deformable_mirror.IF.reshape(total_act, -1).pow(2).sum(dim=-1) * weights).sum()
    loss.backward()

    grad = deformable_mirror._sign.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.allclose(grad, grad[0].expand(total_act), atol=1e-6)


def test_offset_change_rebroadcasts_mean_into_new_vector(deformable_mirror):
    deformable_mirror.per_actuator_calibration = True
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.moffatParameter = torch.linspace(1.0, 2.0, total_act)
    expected_raw_mean = deformable_mirror._moffatParameter.detach().mean()

    deformable_mirror.offset_to_fit_number_of_actuators = 0.5

    new_total_act = int(deformable_mirror.totalAct.item())
    assert deformable_mirror._moffatParameter.shape == (new_total_act,)
    assert torch.allclose(deformable_mirror._moffatParameter, expected_raw_mean.expand(new_total_act), atol=1e-5)


def test_load_calibration_preserves_per_actuator_vector_and_flag(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    deformable_mirror.per_actuator_calibration = True
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.sign = torch.linspace(1e-5, 2e-5, total_act)
    deformable_mirror.MakeZonalModes()

    path = tmp_path / "dm_per_actuator.pth"
    deformable_mirror.SaveCalibration(str(path))

    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    # per_actuator_calibration is saved as a top-level checkpoint key (not in
    # DMDict) and restored on load -- a fresh object starts False...
    assert fresh.per_actuator_calibration is False
    fresh.LoadCalibration(str(path))
    # ...but picks up the source's True after loading its checkpoint.
    assert fresh.per_actuator_calibration is True

    assert torch.allclose(fresh.sign, deformable_mirror.sign, atol=1e-9)


def test_load_calibration_defaults_flag_false_for_old_checkpoint_without_it(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    # A checkpoint saved without the "per_actuator_calibration" key (e.g. from
    # before this key existed) must still load cleanly, defaulting to False.
    misreg, DMDict = deformable_mirror.GetMisreg()
    path = tmp_path / "dm_legacy_checkpoint.pth"
    torch.save({"model": deformable_mirror.state_dict(), "config": DMDict, "misreg": misreg}, path)

    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    fresh.per_actuator_calibration = True
    fresh.LoadCalibration(str(path))

    assert fresh.per_actuator_calibration is False


def test_load_calibration_with_mismatched_construction_offset(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    deformable_mirror.per_actuator_calibration = True
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.sign = torch.linspace(1e-5, 2e-5, total_act)
    deformable_mirror.MakeZonalModes()

    path = tmp_path / "dm_mismatched_offset.pth"
    deformable_mirror.SaveCalibration(str(path))

    # Constructed with a different offset_to_fit_number_of_actuators than what
    # was saved (0.5 vs. the default 0.2), so totalAct differs from the
    # checkpoint's until LoadCalibration restores geometry before load_state_dict.
    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device, offset_to_fit_number_of_actuators=0.5)
    assert fresh.totalAct != deformable_mirror.totalAct

    fresh.LoadCalibration(str(path))

    assert fresh.totalAct == deformable_mirror.totalAct
    assert torch.allclose(fresh.sign, deformable_mirror.sign, atol=1e-9)


@pytest.mark.parametrize("attr,value", [
    ("rotationAngle", 10.0),
    ("grid_shift", [[[[0.7]], [[-0.4]]]]),
    ("radialScaling", 0.05),
    ("tangentialScaling", -0.03),
    ("anamorphosisAngle", 20.0),
    ("sign", 2e-5),
    ("moffatParameter", 3.0),
    ("mechCoupling", 0.2),
])
def test_setter_rebuilds_influence_functions_in_eval_mode(deformable_mirror, device, attr, value):
    """In eval mode forward() doesn't rebuild the IFs, so every geometry/shape
    setter must do it itself -- otherwise the DM silently keeps its old shape."""
    if attr == "anamorphosisAngle":  # only has an effect once the two scalings differ
        deformable_mirror.radialScaling = torch.tensor(0.05, device=device)
    if attr == "moffatParameter":  # only shapes the Moffat IF, not the default TPS
        deformable_mirror.if_type = "moffat"
    deformable_mirror.eval()
    IF_before = deformable_mirror.IF.clone()

    setattr(deformable_mirror, attr, torch.tensor(value, device=device))

    assert not torch.allclose(deformable_mirror.IF, IF_before)
    IF_after_setter = deformable_mirror.IF.clone()
    deformable_mirror.MakeZonalModes()
    assert torch.allclose(deformable_mirror.IF, IF_after_setter)


def test_eval_train_round_trip_keeps_manually_frozen_parameter(deformable_mirror):
    deformable_mirror._anamorphosisAngle.requires_grad_(False)

    deformable_mirror.eval()
    assert not any(p.requires_grad for p in deformable_mirror.parameters())

    deformable_mirror.train()
    assert not deformable_mirror._anamorphosisAngle.requires_grad
    assert deformable_mirror._rotationAngle.requires_grad


def test_influence_function_couples_at_one_actuator_pitch(tiny_wfs_params, tiny_dm_params, device):
    """In the Gaussian limit of the Moffat profile (large moffatParam) the
    influence function must equal MechCoupling one actuator pitch away, with
    the pitch being the actual actuator spacing Nres / (Nact - 1)."""
    # beta = 1e3: Gaussian to ~3e-4 of the amplitude, yet small enough that
    # (1 + r2 / beta) ** beta stays accurate in float32
    params = dict(tiny_dm_params, IFType="moffat", moffatParam=1e3, MechCoupling=0.3)
    dm = DeformableMirror(tiny_wfs_params(), params, device)
    dm.eval()

    pitch = dm.Nres / (dm.Nact - 1)
    spacing = torch.cdist(dm.actuator_positions, dm.actuator_positions)
    assert torch.isclose(spacing[spacing > 0].min(), torch.tensor(pitch, device=device))

    # Independent Gaussian model exp(-d^2 ln(1/m) / pitch^2), then the same
    # pupil masking + per-IF piston removal as the DM applies.
    d2 = ((dm.actuator_positions[:, :, None, None] - dm.positions) ** 2).sum(dim=1)
    expected = params["signedAmplitude"] * torch.exp(-d2 * torch.log(torch.tensor(1 / 0.3)) / pitch**2)
    expected = expected * dm.pupil
    expected[:, dm.pupil] -= expected[:, dm.pupil].mean(dim=-1, keepdim=True)

    assert torch.allclose(dm.IF, expected, atol=1e-3 * params["signedAmplitude"])


def test_load_calibration_converts_legacy_pitch_convention(tiny_wfs_params, tiny_dm_params, device, tmp_path):
    """Calibration files saved before the IF width used the Nres/(Nact-1)
    pitch must load to the same influence functions they were fit with, by
    re-expressing the coupling as m ** (Nact / (Nact - 1)) ** 2."""
    dm = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    path = tmp_path / "dm.pth"
    dm.SaveCalibration(str(path))

    checkpoint = torch.load(path)
    assert checkpoint.pop("if_width_pitch") == "Nres/(Nact-1)"
    legacy_path = tmp_path / "legacy_dm.pth"
    torch.save(checkpoint, legacy_path)

    new_format = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    new_format.LoadCalibration(str(path))
    assert torch.allclose(new_format.mechCoupling, dm.mechCoupling, atol=1e-6)

    legacy = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    legacy.LoadCalibration(str(legacy_path))
    Nact = tiny_dm_params["Nactuator"]
    expected = dm.mechCoupling ** ((Nact / (Nact - 1)) ** 2)
    assert torch.allclose(legacy.mechCoupling, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# GetActuatorPositions
# ---------------------------------------------------------------------------

def _peak_centroids(dm):
    """(x, y) centroid of each influence function above half its maximum, in the
    same pupil-pixel coordinates as GetActuatorPositions."""
    IF = dm.IF.detach().abs()
    w = IF * (IF > 0.5 * IF.amax(dim=(-2, -1), keepdim=True))
    w = w / w.sum(dim=(-2, -1), keepdim=True)
    x, y = dm.positions[0, 0], dm.positions[0, 1]
    return torch.stack(((w * x).sum(dim=(-2, -1)), (w * y).sum(dim=(-2, -1))), dim=-1)


@pytest.mark.parametrize("flip_lr,flip_tb", [(False, False), (True, False), (False, True), (True, True)])
def test_get_actuator_positions_matches_influence_function_centres(tiny_wfs_params, tiny_dm_params, device,
                                                                   flip_lr, flip_tb):
    """The positions include anamorphosis, rotation, flips and shift: they sit
    where MakeZonalModes centres each influence function."""
    wfs_params = dict(tiny_wfs_params(), Nres=40)
    dm_params = dict(tiny_dm_params, Nactuator=9, moffatParam=50.0, MechCoupling=0.2,
                     FlipLeftRight=flip_lr, FlipTopBottom=flip_tb, IFType="moffat")
    misreg = dict(rotationAngle=17.0, shiftX=0.03, shiftY=-0.02, anamorphosisAngle=25.0,
                  tangentialScaling=4.0, radialScaling=-3.0)
    dm = DeformableMirror(wfs_params, dm_params, device, misreg=misreg)
    dm.eval()
    positions = dm.GetActuatorPositions()
    assert positions.shape == (int(dm.totalAct), 2)

    # interior actuators only: the pupil clips the edge ones
    interior = positions.norm(dim=-1) < 0.35 * dm.Nres
    assert interior.sum() >= 9
    centres = _peak_centroids(dm)
    assert torch.allclose(positions[interior], centres[interior], atol=0.15)


def test_get_actuator_positions_is_differentiable(deformable_mirror):
    deformable_mirror.train()
    positions = deformable_mirror.GetActuatorPositions()
    # weighted sum: the plain sum over the symmetric grid doesn't depend on the rotation
    weights = torch.arange(positions.numel(), dtype=positions.dtype, device=positions.device)
    (positions.flatten() * weights).sum().backward()
    for name in ("_rotationAngle", "_grid_shift", "_radialScaling", "_tangentialScaling"):
        grad = getattr(deformable_mirror, name).grad
        assert grad is not None and grad.abs().sum() > 0, name


# ---------------------------------------------------------------------------
# Thin-plate-spline influence functions (IFType="tps", the default)
# ---------------------------------------------------------------------------

def _on_pixel_misreg(wfs_params):
    """Half-pixel shift: with an integer pitch it puts every actuator centre
    exactly on a pupil-pixel centre (where r = 0 in the TPS kernel)."""
    half_pixel = 0.5 * wfs_params["D"] / wfs_params["Nres"]
    return dict(rotationAngle=0.0, shiftX=half_pixel, shiftY=half_pixel, anamorphosisAngle=0.0,
                radialScaling=0.0, tangentialScaling=0.0)


def _node_pixels(dm):
    """(row, col) of the pixel under each actuator, and which ones are on the grid."""
    pos = dm.GetActuatorPositions().detach()
    col = (pos[:, 0] + dm.Nres / 2 - 0.5).round().long()
    row = (pos[:, 1] + dm.Nres / 2 - 0.5).round().long()
    on_grid = (row >= 0) & (row < dm.Nres) & (col >= 0) & (col < dm.Nres)
    return row, col, on_grid


def test_tps_is_the_default_if_type(deformable_mirror):
    assert deformable_mirror.if_type == "tps"
    assert deformable_mirror.GetMisreg()[1]["IFType"] == "tps"


def test_invalid_if_type_raises(tiny_wfs_params, tiny_dm_params, device):
    with pytest.raises(ValueError):
        DeformableMirror(tiny_wfs_params(), dict(tiny_dm_params, IFType="spline"), device)


def test_tps_moffat_param_is_optional(tiny_wfs_params, tiny_dm_params, device):
    params = {k: v for k, v in tiny_dm_params.items() if k != "moffatParam"}
    dm = DeformableMirror(tiny_wfs_params(), params, device)
    assert torch.isfinite(dm.IF).all()


@pytest.mark.parametrize("coupling", [0.1, 0.3])
def test_tps_interpolates_coupling_pattern_at_nodes(tiny_wfs_params, tiny_dm_params, device, coupling):
    """Before pupil masking, IF_j at node i is mechCoupling ** (d_ij / pitch)^2:
    1 on its own actuator, mechCoupling on the nearest neighbours."""
    wfs_params = dict(tiny_wfs_params(), Nres=24)
    dm = DeformableMirror(wfs_params, dict(tiny_dm_params, MechCoupling=coupling), device,
                          misreg=_on_pixel_misreg(wfs_params))
    with torch.no_grad():
        raw = dm._ThinPlateSplineIF(dm.mechCoupling)
    row, col, on_grid = _node_pixels(dm)
    assert on_grid.sum() >= 9

    d2 = torch.cdist(dm.actuator_positions, dm.actuator_positions) ** 2 / dm.pitch ** 2
    expected = coupling ** d2[:, on_grid]
    assert torch.allclose(raw[:, row[on_grid], col[on_grid]], expected, atol=1e-4)


def test_tps_if_is_bounded_in_pupil(tiny_wfs_params, tiny_dm_params, device):
    wfs_params = dict(tiny_wfs_params(), Nres=40)
    dm = DeformableMirror(wfs_params, dict(tiny_dm_params, Nactuator=9, MechCoupling=0.2), device)
    with torch.no_grad():
        raw = dm._ThinPlateSplineIF(dm.mechCoupling)
    assert raw[:, dm.pupil].abs().max() <= 1.0 + 1e-3


def test_tps_gradients_finite_with_actuators_on_pixels(tiny_wfs_params, tiny_dm_params, device):
    """r^2 ln r is 0 * log(0) at r = 0: the kernel must give finite gradients
    when a pixel sits exactly on an actuator (and on the K diagonal)."""
    wfs_params = dict(tiny_wfs_params(), Nres=24)
    misreg = dict(_on_pixel_misreg(wfs_params), radialScaling=0.0)
    dm = DeformableMirror(wfs_params, tiny_dm_params, device, misreg=misreg).train()
    row, col, on_grid = _node_pixels(dm)
    assert on_grid.any()

    coefs = torch.randn(2, int(dm.totalAct), device=device)
    dm(coefs).pow(2).sum().backward()
    for name, param in dm.named_parameters():
        if name == "_moffatParameter":
            assert param.grad is None  # unused by the TPS IF
            continue
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), name


def test_tps_kernel_value_and_gradient_at_zero():
    r2 = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64, requires_grad=True)
    phi = DeformableMirror._TPSKernel(r2)
    assert torch.allclose(phi.detach(), torch.tensor([0.0, 0.0, 2.0 * torch.log(torch.tensor(4.0)).item()],
                                                     dtype=torch.float64))
    phi.sum().backward()
    # d/dr2 [r2 ln(r2) / 2] = (ln(r2) + 1) / 2, and 0 (not NaN) at r2 = 0
    expected = torch.tensor([0.0, 0.5, (torch.log(torch.tensor(4.0)).item() + 1) / 2], dtype=torch.float64)
    assert torch.allclose(r2.grad, expected)


@pytest.fixture
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_tps_gradcheck_geometry_and_coupling(tiny_wfs_params, tiny_dm_params, float64_default):
    """Analytic gradients of the TPS IF match finite differences for every
    geometry parameter and the coupling (float64 DM, generic misregistration)."""
    misreg = dict(rotationAngle=7.0, shiftX=0.03, shiftY=-0.02, anamorphosisAngle=25.0,
                  radialScaling=3.0, tangentialScaling=-2.0)
    dm = DeformableMirror(tiny_wfs_params(), dict(tiny_dm_params, MechCoupling=0.3), "cpu", misreg=misreg)
    dm.per_actuator_calibration = True
    names = ["_rotationAngle", "_grid_shift", "_radialScaling", "_tangentialScaling",
             "_anamorphosisAngle", "_mechCoupling"]
    originals = {n: getattr(dm, n) for n in names}
    probe = torch.randn(int(dm.totalAct), dm.Nres, dm.Nres)

    def loss(*values):
        for n, v in zip(names, values):
            dm._parameters[n] = v
        dm.MakeZonalModes()
        return (dm.IF / 1e-5 * probe).sum()

    inputs = tuple(originals[n].detach().clone().requires_grad_() for n in names)
    try:
        assert torch.autograd.gradcheck(loss, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    finally:
        for n in names:
            dm._parameters[n] = originals[n]


def test_tps_per_actuator_coupling_only_changes_that_actuator(tiny_wfs_params, tiny_dm_params, device):
    dm = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device).eval()
    dm.per_actuator_calibration = True
    IF_before = dm.IF.clone()

    coupling = dm.mechCoupling.detach().clone()
    coupling[3] = 0.1
    dm.mechCoupling = coupling

    changed = ~torch.isclose(dm.IF, IF_before, atol=1e-12).flatten(start_dim=1).all(dim=-1)
    assert changed.tolist() == [i == 3 for i in range(int(dm.totalAct))]


def test_save_load_keeps_if_type_and_old_checkpoints_load_as_moffat(tiny_wfs_params, tiny_dm_params, device, tmp_path):
    moffat = DeformableMirror(tiny_wfs_params(), dict(tiny_dm_params, IFType="moffat"), device)
    path = tmp_path / "moffat.pth"
    moffat.SaveCalibration(str(path))
    assert torch.load(path)["if_type"] == "moffat"

    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    assert fresh.if_type == "tps"
    fresh.LoadCalibration(str(path))
    assert fresh.if_type == "moffat"
    assert torch.allclose(fresh.IF, moffat.IF, atol=1e-12)

    # Checkpoints from before IFType existed were fit with Moffat IFs.
    checkpoint = torch.load(path)
    del checkpoint["if_type"]
    legacy_path = tmp_path / "legacy.pth"
    torch.save(checkpoint, legacy_path)
    legacy = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    legacy.LoadCalibration(str(legacy_path))
    assert legacy.if_type == "moffat"
    assert torch.allclose(legacy.IF, moffat.IF, atol=1e-12)


@pytest.mark.parametrize("flip_lr,flip_tb", [(False, False), (True, False), (False, True), (True, True)])
def test_tps_follows_misregistration(tiny_wfs_params, tiny_dm_params, device, flip_lr, flip_tb):
    """Under rotation, shift, anamorphosis and flips, the raw TPS IF_j sampled
    at the (misregistered) actuator positions is still the coupling pattern:
    1 on actuator j, mechCoupling on its nearest neighbours."""
    wfs_params = dict(tiny_wfs_params(), Nres=40)
    dm_params = dict(tiny_dm_params, Nactuator=9, MechCoupling=0.2, FlipLeftRight=flip_lr, FlipTopBottom=flip_tb)
    misreg = dict(rotationAngle=17.0, shiftX=0.03, shiftY=-0.02, anamorphosisAngle=25.0,
                  tangentialScaling=4.0, radialScaling=-3.0)
    dm = DeformableMirror(wfs_params, dm_params, device, misreg=misreg).eval()
    with torch.no_grad():
        raw = dm._ThinPlateSplineIF(dm.mechCoupling)
        positions = dm.GetActuatorPositions()
    interior = positions.norm(dim=-1) < 0.35 * dm.Nres
    assert interior.sum() >= 9

    # bilinear sampling of every IF at every interior actuator position
    grid = (positions[interior] / (dm.Nres / 2)).view(1, 1, -1, 2).expand(raw.shape[0], 1, -1, 2)
    sampled = torch.nn.functional.grid_sample(raw[:, None], grid, align_corners=False)[:, 0, 0]

    d2 = torch.cdist(dm.actuator_positions, dm.actuator_positions)[:, interior] ** 2 / dm.pitch ** 2
    expected = 0.2 ** d2
    # atol covers bilinear sampling flattening the peak (~4% at this pitch)
    assert torch.allclose(sampled[interior], expected[interior], atol=0.06)
