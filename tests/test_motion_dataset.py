import importlib.util
from pathlib import Path
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
spec = importlib.util.spec_from_file_location('motion_dataset_builder', Path(__file__).parents[1]/'tools/build_motion_dataset.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def params():
    return dict(velocity_px_per_frame=[12., 0.], acceleration_px_per_frame2=[.1, -.1],
                exposure_frames=1., exposure_samples=129)


def test_exposure_kernel_normalized_and_centered():
    kernel, offsets = builder.exposure_kernel(0., params())
    assert np.isclose(kernel.sum(), 1.)
    assert (kernel >= 0).all() and np.isfinite(kernel).all()
    assert offsets.shape == (129, 2)
    assert np.ptp(offsets[:, 0]) > 11


def test_translation_and_exposure_orientation():
    p = params()
    p['acceleration_px_per_frame2'] = [0., 0.]
    image = np.zeros((80, 80, 3), np.float32)
    image[40, 40] = 1.
    pixels, displacement, _ = builder.render_frame(image, 1., p)
    assert np.allclose(displacement, [12., 0.])
    mass = pixels[:, :, 0].astype(float)
    xx = (mass*np.arange(80)[None, :]).sum()/mass.sum()
    yy = (mass*np.arange(80)[:, None]).sum()/mass.sum()
    assert abs(xx-52) < .1 and abs(yy-40) < .1
    assert (mass.sum(0) > 0).sum() >= 11


def test_source_references_are_not_written(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(7)
    private = tmp_path/'private.png'
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(private)
    row = dict(private_path=str(private), source_id='source_000001', group_id='group_000001',
               priority=False, width=128, height=128, split='train')
    settings = dict(output=str(tmp_path/'public'), side=128, priority_variants=12, seed=42, jpeg_quality=94)
    a, frames = builder.generate_source((row, settings))
    assert len(a) == 1 and len(frames) == 5
    hashes = [r['sha256'] for r in frames]
    assert len(set(hashes)) == 5
    assert builder.sha256_file(private) not in hashes
    assert not list((tmp_path/'public').rglob('*.png'))
    assert all(r['synthetic'] for r in frames)
    _, repeated = builder.generate_source((row, settings))
    assert hashes == [r['sha256'] for r in repeated]


def test_srgb_roundtrip_and_seed():
    pixels = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, axis=2)
    assert np.array_equal(builder.encode_srgb(builder.linearize(pixels)), pixels)
    assert builder.stable_seed('x', 42) == builder.stable_seed('x', 42)
    assert builder.stable_seed('x', 42) != builder.stable_seed('y', 42)
