# import torch
# import logging as log

# def batch_data_to_device(data, device):
#     batch_x, y = data
#     y = y.to(device)
#     for x in batch_x:
#         x = x.to(device)
#     return [batch_x, y]


import torch
import logging as log

def batch_data_to_device(data, device):
    batch_x, y = data
    y = y.to(device)

    seq_num, x = batch_x
    seq_num = seq_num.to(device)
    x_len = len(x[0])
    # x_len = 8
    # log.info('x length {:d}'.format(x_len))
    for i in range(0, len(x)):
        for j in range(0, x_len):
            x[i][j] = x[i][j].to(device)

    return [[seq_num, x], y]
