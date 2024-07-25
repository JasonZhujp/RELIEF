import datetime

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

       
def get_time():
    return (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%H:%M:%S')


def logargs(args, width=120):
    length = 1
    L=[]
    l= "|"
    for id,arg in enumerate(vars(args)):
        name,value = arg, str(getattr(args, arg))
        nv = name+":"+value
        if length +(len(nv)+2)>width:
            L.append(l)
            l = "|"
            length = 1
        l += nv + " |"
        length += (len(nv)+2)
        if id+1 == len(vars(args)):
            L.append(l)
    printstr = niceprint(L)
    print(printstr)


def niceprint(L,mark="-"):
    printstr = []
    printstr.append("-"*len(L[0]))
    printstr.append(L[0])
    for id in range(1,len(L)):
        printstr.append("-"*max(len(L[id-1]),len(L[id])))
        printstr.append(L[id])
    printstr.append("-"*len(L[-1]))
    printstr = "\n".join(printstr)
    return printstr


def index_new_batch(old_batch, pop_idx, return_pop=False):
    device = old_batch.x.device
    
    if len(pop_idx) == 0 and isinstance(old_batch, Batch):
        return old_batch
    
    last_data_list = None
    if isinstance(old_batch, Batch):
        old_data_list = old_batch.to_data_list()
    elif isinstance(old_batch, list):
        if old_batch[0]:
            last_data_list = old_batch[0].to_data_list()
        old_data_list = old_batch[1].to_data_list()

    # ordered_pop_idx = torch.arange(idx_mask.size(0))[idx_mask].sort(dim=0, descending=True)[0].tolist()
    ordered_pop_idx = torch.sort(pop_idx, dim=0, descending=True)[0].tolist()
    pop_data_list = []
    for idx in ordered_pop_idx:
        pop_data_list.append(old_data_list.pop(idx))

    if last_data_list is None:
        if len(old_data_list):
            if return_pop:
                return Batch.from_data_list(old_data_list).to(device), pop_data_list
            else:
                return Batch.from_data_list(old_data_list).to(device)
        else:
            if return_pop:
                return None, pop_data_list
            else:
                return None
    else:
        return Batch.from_data_list(last_data_list + old_data_list).to(device)
