import torch

def col_fn(batch):
    out = dict()
    out['data'] = torch.stack([x['data']['data'] for x in batch])
    out['seg'] = torch.stack([x['seg']['data'] for x in batch])
    return out