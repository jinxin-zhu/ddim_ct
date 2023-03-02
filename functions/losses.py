import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    if e.dim() == 4:
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    elif e.dim() == 5:
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
    else:
        raise Exception(r"Sorry, the shape of inputed image and noise is not right.")
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()    # noisy tensor
    output = model(x, t.float())        # noise prediction
    
    if e.dim() == 4:
        loss = (e - output).square().sum(dim=(1, 2, 3))
    elif e.dim() == 5:
        loss = (e - output).square().sum(dim=(1, 2, 3, 4))

    if keepdim:
        return loss
    else:
        return loss.mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
