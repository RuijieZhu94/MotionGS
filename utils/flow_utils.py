import torch

def conic_to_matrix(conic):
    K, _, H, W = conic.shape
    conic = torch.cat((conic[:, :2], 
                       torch.zeros_like(conic[:, :1]).to(conic.device), 
                       conic[:, 2:]), dim=1) # K, 4, H, W
    conic = conic.view(K, 2, 2, H, W)
    return conic

def calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, proj_2D, next_proj_2D, x_mu, iteration=None):
    # gs_per_pixel = gs_per_pixel.long() # K H W
    # # deal with empty gs
    # valid_mask = ~(gs_per_pixel < 0).any(dim=0) # H W
    # proj_2D_per_pixel = proj_2D[gs_per_pixel].permute(0, 3, 1, 2) # K 2 H W
    # next_proj_2D_per_pixel = next_proj_2D[gs_per_pixel].permute(0, 3, 1, 2) # K 2 H W
    # next_conic_2D_per_pixel = conic_to_matrix(next_conic_2D[gs_per_pixel].permute(0, 3, 1, 2)) # K 3 H W -> K 2 2 H W
    # conic_2D_inv_per_pixel = conic_to_matrix(conic_2D_inv[gs_per_pixel].permute(0, 3, 1, 2)) # K 3 H W -> K 2 2 H W
    # flow_per_pixel = torch.einsum("kabhw, kbchw, kchw -> kahw", [next_conic_2D_per_pixel, conic_2D_inv_per_pixel, 
    #                     x_mu]) + next_proj_2D_per_pixel - (x_mu + proj_2D_per_pixel) # K 2 H W
    # weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # K H W
    # flow_gs = torch.einsum("khw, kahw -> ahw", [weight_per_gs_pixel, flow_per_pixel]) # 2 H W
    # return flow_gs * valid_mask.float()


    conic_2D_inv = conic_2D_inv.detach() # K 3

    gs_per_pixel = gs_per_pixel.long() # K H W
    # valid_mask = ~(gs_per_pixel < 0).any(dim=0) # H W
    conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], device=conic_2D_inv.device) # K 2 2
    conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + next_conic_2D[:, 1] * conic_2D_inv[:, 1]
    conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + next_conic_2D[:, 1] * conic_2D_inv[:, 2]
    conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + next_conic_2D[:, 2] * conic_2D_inv[:, 1]
    conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + next_conic_2D[:, 2] * conic_2D_inv[:, 2]
    #实验1 把conv_conv 强制clip=10
   # conv_conv = torch.clamp(
        #conv_conv,
        #min=-10.0,
        #max=10.0
    #)
    #实验2 clip=2
    conv_conv = torch.clamp(conv_conv, -2.0, 2.0)
    # isotropic gs flow
    # flow_per_pixel = next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() # K H W 3

    # anisotropic gs flow
    #调试输出、新加、记得删
    conv_multi = (conv_conv[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze() # K H W 2
    if iteration in [2900, 2950, 3000, 3050, 3100, 3150, 3200, 3300]:
        print(f"\n==== FLOW INTERNAL @ {iteration} ====")

        print("conv_conv")
        print(" mean_abs =", conv_conv.abs().mean().item())
        print(" max      =", conv_conv.abs().max().item())

        print("conv_multi")
        print(" mean_abs =", conv_multi.abs().mean().item())
        print(" max      =", conv_multi.abs().max().item())

    if torch.rand(1).item() < 0.0001:  # 先防止刷屏
        print("\n===== GS FLOW DEBUG =====")
        print("x_mu")
        print(" mean_abs =", x_mu.abs().mean().item())
        print(" max      =", x_mu.abs().max().item())

        print("conv_conv")
        print(" mean_abs =", conv_conv.abs().mean().item())
        print(" max      =", conv_conv.abs().max().item())

        print("conv_multi")
        print(" mean_abs =", conv_multi.abs().mean().item())
        print(" max      =", conv_multi.abs().max().item())

    flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) # K H W 2
    if iteration in [2900, 2950, 3000, 3050, 3100, 3150, 3200, 3300]:
        delta_proj = (
            next_proj_2D[gs_per_pixel]
            -
            proj_2D[gs_per_pixel]
        )

        print("delta_proj")
        print(" mean_abs =", delta_proj.abs().mean().item())
        print(" max      =", delta_proj.abs().max().item())

        print("flow_per_pixel")
        print(" mean_abs =", flow_per_pixel.abs().mean().item())
        print(" max      =", flow_per_pixel.abs().max().item())

    weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # K H W
    flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), flow_per_pixel]) # 2 H W
    return flow_gs
    # return flow_gs * valid_mask.float().detach()
def calculate_gs_flow_pytorch(weight_per_gs_pixel, proj_2D, next_proj_2D, H, W):
    """
    PyTorch reimplementation of Gaussian flow calculation.
    Bypasses the buggy CUDA gs_per_pixel indexing.
    Vectorized scatter-based splatting.
    """
    # Per-Gaussian flow: P x 2
    flow_per_gs = (next_proj_2D - proj_2D.detach())  # P x 2

    # Get pixel coordinates for each Gaussian
    px = proj_2D[:, 0].detach().long().clamp(0, W - 1)  # P
    py = proj_2D[:, 1].detach().long().clamp(0, H - 1)  # P
    pix_id = py * W + px  # P

    # Gaussian visibility weight: sum over K dimension
    # weight_per_gs_pixel: K x H x W
    vis_weight = weight_per_gs_pixel.sum(dim=0).detach()  # H x W
    gs_vis = vis_weight.reshape(-1)[pix_id]  # P, weight at each Gaussian's pixel

    # Weighted flow
    flow_x_weighted = flow_per_gs[:, 0] * gs_vis  # P
    flow_y_weighted = flow_per_gs[:, 1] * gs_vis  # P

    # Scatter onto image plane
    flow_x = torch.zeros(H * W, device=proj_2D.device)
    flow_y = torch.zeros(H * W, device=proj_2D.device)
    weight_map = torch.zeros(H * W, device=proj_2D.device)

    flow_x.scatter_add_(0, pix_id, flow_x_weighted)
    flow_y.scatter_add_(0, pix_id, flow_y_weighted)
    weight_map.scatter_add_(0, pix_id, gs_vis)

    # Normalize
    weight_map = weight_map.clamp(min=1e-7)
    flow_x = flow_x / weight_map
    flow_y = flow_y / weight_map

    return torch.stack([
        flow_x.reshape(H, W),
        flow_y.reshape(H, W)
    ], dim=0)  # 2 x H x W