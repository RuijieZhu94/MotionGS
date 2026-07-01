import torch


def conic_to_matrix(conic):
    K, _, H, W = conic.shape
    conic = torch.cat(
        (
            conic[:, :2],
            torch.zeros_like(conic[:, :1]).to(conic.device),
            conic[:, 2:],
        ),
        dim=1,
    )  # K, 4, H, W
    conic = conic.view(K, 2, 2, H, W)
    return conic


def calculate_gs_flow(
    gs_per_pixel,
    weight_per_gs_pixel,
    next_conic_2D,
    conic_2D_inv,
    proj_2D,
    next_proj_2D,
    x_mu,
):
    del next_conic_2D, conic_2D_inv, x_mu

    gs_per_pixel = gs_per_pixel.long()
    valid_mask = gs_per_pixel >= 0
    safe_gs_per_pixel = gs_per_pixel.clamp_min(0)

    flow_per_pixel = (
        next_proj_2D[safe_gs_per_pixel] - proj_2D[safe_gs_per_pixel].detach()
    )
    flow_per_pixel = flow_per_pixel * valid_mask.unsqueeze(-1)

    weight_per_gs_pixel = weight_per_gs_pixel * valid_mask
    weight_per_gs_pixel = weight_per_gs_pixel / (
        weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7
    )
    flow_gs = torch.einsum(
        "khw,khwa->ahw", [weight_per_gs_pixel.detach(), flow_per_pixel]
    )
    return flow_gs
