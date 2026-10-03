import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    wi = W_in.clone()
    wo = W_out.clone()
    vc = W_in[center_id]
    uo = W_out[pos_id]
    un = W_out[neg_ids]
    so = torch.dot(vc, uo)
    sn = un @ vc
    po = torch.sigmoid(so)
    pn = torch.sigmoid(sn)
    g_uo = (po - 1) * vc
    g_un = pn[:, None] * vc
    g_vc = (po - 1) * uo + (pn[:, None] * un).sum(dim=0)
    wi[center_id] -= lr * g_vc
    wo[pos_id] -= lr * g_uo
    for k, idx in enumerate(neg_ids):
        wo[idx] -= lr * g_un[k]
    return {"W_in": wi, "W_out": wo}