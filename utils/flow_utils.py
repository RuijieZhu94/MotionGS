import torch

def conic_to_matrix(conic):
    K, _, H, W = conic.shape
    conic = torch.cat((conic[:, :2], 
                       torch.zeros_like(conic[:, :1]).to(conic.device), 
                       conic[:, 2:]), dim=1) # K, 4, H, W
    conic = conic.view(K, 2, 2, H, W)
    return conic

def calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, proj_2D, next_proj_2D, x_mu):
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

    # isotropic gs flow
    # flow_per_pixel = next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() # K H W 3

    # anisotropic gs flow
    conv_multi = (conv_conv[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze() # K H W 2
    flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) # K H W 2

    weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # K H W
    flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), flow_per_pixel]) # 2 H W
    return flow_gs
    # return flow_gs * valid_mask.float().detach()