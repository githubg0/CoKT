'''用来跑原来的CoKT的'''
import os
import torch
import torch.nn.functional as F
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score
# from sklearn.metrics import hamming_loss
# from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
# import pytorch_lightning.metrics.functional as light_func
import logging as log
import numpy
import tqdm
import pickle
from utils import batch_data_to_device
import datetime

def train(model, loaders, update_loaders, args):
    log.info("training...")
    # relation_matrix, p_emb = load_relation(args)
    # BCELoss = torch.nn.BCELoss()
    BCELoss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_sigmoid = torch.nn.Sigmoid()
    # pre_show_loss = 100
    train_len = len(loaders['train'].dataset)
    # update_len = len(update_loaders['train'].dataset)
    # print('train_len:', train_len, 'update_len:', update_len)
    db_name = args.data_dir.split('/')[-2]
    sear_indexes = [db_name + '_train_search', db_name + '_valid_search']

    for _, dt in enumerate(update_loaders['train']):
        with torch.no_grad():
            s_x, _ = batch_data_to_device(dt, args.device)
        model.eval()
        # print(len(s_x[0]))
        model.update_hidden(s_x, 'train')
    
    for epoch in range(args.n_epochs):
        loss_all = 0
        acc_all = 0
        auc_all = 0
        count_all = 0
        # for step, data in tqdm.tqdm(enumerate(loaders['train'])):
        # for step, data in list(enumerate(loaders['train'])):
        # for step, data in tqdm.tqdm(list(enumerate(loaders['train']))):
        for step, data in list(enumerate(loaders['train'])):
            '''obtain the states'''
            
            # starttime = datetime.datetime.now()
            # for _, dt in enumerate(update_loaders['train']):
            #     with torch.no_grad():
            #         s_x, _ = batch_data_to_device(dt, args.device)
            #     model.eval()
            #     # print(len(s_x[0]))
            #     model.update_hidden(s_x, 'train')
            # endtime = datetime.datetime.now()
            # duringtime = endtime - starttime
            # print('update during time:', duringtime.seconds)
            
            '''update end'''

            # starttime = datetime.datetime.now()

            with torch.no_grad():
                x, y = batch_data_to_device(data, args.device)
            model.train()
            # print(x)
            # with torch.autograd.set_detect_anomaly(True):
            hat_y_prob = None
            # logits = model(x)
            logits = model(x, 'train')
            # print(logits)
            hat_y_prob = train_sigmoid(logits)
            hat_y = hat_y_prob

            '''calculation loss in graph generation'''

            loss = BCELoss(logits, y.float())

            optimizer.zero_grad() 
            loss.backward()

            # for name, parms in model.named_parameters():
            #     # print(name)
                # print('grad_None: -->name:', name)
            # #     if parms.grad == None:
                # if 'prob_emb' in name:
                    # print('grad_None: -->name:', name, ' -->grad_value:',parms.grad)

            optimizer.step()
            step += 1
            # model.whole_graph_gen()
            for _, dt in enumerate(update_loaders['train']):
                with torch.no_grad():
                    s_x, _ = batch_data_to_device(dt, args.device)
                model.eval()
                # print(len(s_x[0]))
                model.update_hidden(s_x, 'train')

            with torch.no_grad():
                # loss_all += loss.item() * batch_size
                
                loss_all += loss.item()
                hat_y_bin = (hat_y_prob > 0.5).int()
                acc = accuracy_score(y.int().cpu().numpy(), hat_y_bin.cpu().numpy())
                # print(y.detach().int().cpu().numpy(), hat_y_prob.detach().cpu().numpy())
                fpr, tpr, thresholds = metrics.roc_curve(y.detach().int().cpu().numpy(), hat_y_prob.detach().cpu().numpy(),pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auc_all += auc
                acc_all += acc
                count_all += 1

            # endtime = datetime.datetime.now()
            # duringtime = endtime - starttime
            # print('predict during time:', duringtime.seconds)
        # print(step, sst)
        show_loss = loss_all / train_len
        show_acc = acc_all / count_all
        show_auc = auc_all / count_all
        acc, auc, auroc = evaluate(model, loaders['valid'], args, epoch, sear_indexes[1])
        acc_t, auc_t, auroc_t = evaluate(model, loaders['test'], args, epoch, sear_indexes[1])
        log.info('Epoch: {:03d}, Loss: {:.7f}, train_acc: {:.7f}, train_auc: {:.7f}, valid_acc: {:.7f}, valid_auc: {:.7f}, test_acc: {:.7f}, test_auc: {:.7f}'.format(
                        epoch, show_loss, show_acc, show_auc, acc, auc, acc_t, auc_t))
        # print('Epoch: {:03d}, Loss: {:.7f}, train_acc: {:.7f}, train_auc: {:.7f}, acc: {:.7f}, auc: {:.7f}'.format(epoch, show_loss, show_acc, show_auc, acc, auc))
        
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model, os.path.join(args.run_dir, 'params_%i.pt' % epoch))


def evaluate(model, loader, args,signal, sear_index):
    model.eval()
    rre_list = []
    eval_sigmoid = torch.nn.Sigmoid()
    y_list, hat_y_list = [], []
    with torch.no_grad():
        for data in loader:
            # x, y = data
            x, y = batch_data_to_device(data, args.device)
       
            hat_y_prob = model(x, 'valid')
            y_list.append(y)
            hat_y_list.append(eval_sigmoid(hat_y_prob))
            # rre_list.append(re_list)
            # if signal == 100:
            #     actual = torch.where(hat_y_prob > 0.5, torch.tensor(1).to(args.device), torch.tensor(0).to(args.device))
            #     for i in range(0, len(y)):
            #         print(y[i], actual[i], re_list[i])

    y_tensor = torch.cat(y_list, dim = 0).int()
    hat_y_prob_tensor = torch.cat(hat_y_list, dim = 0)
    # log.info('number of positives: {:d}'.format(torch.sum(torch.where(y_tensor == 1, torch.tensor(1).to(args.device), torch.tensor(0).to(args.device)).int())))
    # log.info('number of negatives: {:d}'.format(torch.sum(torch.where(y_tensor == 0, torch.tensor(1).to(args.device), torch.tensor(0).to(args.device)).int())))
    acc = accuracy_score(y_tensor.cpu().numpy(), (hat_y_prob_tensor > 0.5).int().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(y_tensor.cpu().numpy(), hat_y_prob_tensor.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    auroc = 0
    
    return acc, auc, auroc


