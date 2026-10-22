"""Equation (11): means over samples/elements, orthonormal complex FFT loss."""
import torch
from torch.nn import functional as F


def ssim_per_image(x, y, window=11, sigma=1.5):
    """RGB channel mean SSIM, data_range=1, Gaussian 11x11, valid support."""
    size = min(window, x.shape[-2], x.shape[-1])
    if size % 2 == 0:
        size -= 1
    coords = torch.arange(size, device=x.device, dtype=x.dtype) - (size-1)/2
    kernel = torch.exp(-coords.square() / (2*sigma*sigma))
    kernel = kernel / kernel.sum()
    weight = (kernel[:, None] * kernel[None, :]).expand(x.shape[1], 1, size, size)
    def avg(z):
        return F.conv2d(z, weight, groups=x.shape[1])
    ux, uy = avg(x), avg(y)
    vx, vy = avg(x*x)-ux*ux, avg(y*y)-uy*uy
    cov = avg(x*y)-ux*uy
    score = ((2*ux*uy+0.01**2)*(2*cov+0.03**2)) / ((ux*ux+uy*uy+0.01**2)*(vx+vy+0.03**2))
    return score.mean(dim=(1, 2, 3))


def restoration_loss(pred, target, classifier=None, labels=None, lambda_s=0.1, lambda_f=0.01, lambda_c=0.0):
    pixel = F.l1_loss(pred, target)
    structure = 1-ssim_per_image(pred, target).mean()
    frequency = (torch.fft.fft2(pred.float(), norm="ortho") - torch.fft.fft2(target.float(), norm="ortho")).abs().mean()
    recognition = pred.new_zeros(())
    if lambda_c:
        if classifier is None or labels is None or (labels < 0).any():
            raise ValueError("Recognition supervision requires a frozen classifier and valid labels")
        # Do not use no_grad: classifier must transmit gradients to restored image.
        recognition = F.cross_entropy(classifier(pred), labels)
    total = pixel + lambda_s*structure + lambda_f*frequency + lambda_c*recognition
    return total, dict(pixel=pixel, structure=structure, frequency=frequency, recognition=recognition)


def psnr_per_image(pred, target):
    mse = (pred-target).square().mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-12))
