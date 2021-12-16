'''涉及到使用related——index去取inistate的，没有使用0去拼接的都不对，因为减一变成取最后一个了'''

# from typing_extensions import Required
from matplotlib.pyplot import show
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import math
import modules
import logging as log
# import handrnn_modules
import random
# import torch.multiprocessing as mp
# from multiprocessing import Pool
from torch.distributions import Categorical
from elasticsearch import Elasticsearch
import pickle
import tqdm
import multiprocessing as mp
from utils import batch_data_to_device

class CoKT(nn.Module): #
    '''
    继承于ubrkt_345
    改进的点: 参考两个方向的状态的时候使用了attention
    '''
    def __init__(self, args):
        super().__init__()
        # print('THIS IS PLAN1...')
        self.see_ref = args.see_ref
        self.node_dim = args.dim
        self.problem_dim = args.prob_dim
        self.concept_num = args.concept_num
        self.max_concepts = args.max_concepts
        self.node_dim = args.dim
        self.data_dir = args.data_dir
        self.device = args.device
        self.heads_num = args.heads
        self.sigmoid = torch.nn.Sigmoid()
        self.ones = torch.tensor(1).to(args.device)
        self.zeros = torch.tensor(0).to(args.device)
        # self.sc_func = modules.funcs(args.n_layer, args.concept_num, args.concept_num - 1, args.dropout)
        self.predictor = modules.funcs(args.n_layer, 9 * args.dim, 1, args.dropout)
        self.gru_cell = modules.mygru(0, args.dim * 4, args.dim)
        # self.bg_emb = modules.Embedding(self.concept_num * 4, self.concept_num * 4)
        # self.batch_emb = modules.Embedding(self.concept_num * 2, self.concept_num * 2)
        self.bg_emb = nn.ModuleList([modules.Embedding(self.node_dim * 3, self.node_dim) for i in range(self.heads_num)])
        self.batch_emb = nn.ModuleList([modules.Embedding(self.node_dim * 3, self.node_dim) for i in range(self.heads_num)])

        self.bg_integrate = modules.funcs(args.n_layer, self.node_dim * 6, self.node_dim * 6, args.dropout)
        self.his_integrate = modules.funcs(args.n_layer, self.node_dim * 2, self.node_dim * 6, args.dropout)
        self.direction_weight = nn.Parameter(torch.randn(2, 1), requires_grad=True).to(args.device)

        self.in_map = modules.funcs(args.n_layer, self.node_dim * 10, self.node_dim * 4, args.dropout)
        # self.v_emb = nn.ModuleList([modules.Embedding(self.node_dim, self.node_dim) for i in range(self.heads_num)])
        # self.batch_emb = nn.ModuleList([modules.Embedding(self.node_dim, self.node_dim) for i in range(self.heads_num)])

        # self.heads = nn.ParameterList([nn.Parameter(torch.randn(self.concept_num * 2, self.concept_num * 4), requires_grad=True) for i in range(self.heads_num)]).to(args.device)
        self.heads_map = nn.Parameter(torch.randn(self.heads_num, 1), requires_grad=True).to(args.device)
        # self.his_map = nn.Parameter(torch.randn(self.heads_num, 1), requires_grad=True).to(args.device)
        # self.prob_rep = modules.funcs(args.n_layer, args.concept_num, args.dim, args.dropout)
        
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num - 1, args.dim), requires_grad=True).to(args.device)
        
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number - 1, args.dim), requires_grad=True).to(args.device)
        # self.prob_cat = torch.cat([
        #     torch.zeros(1, args.dim).to(args.device),
        #     prob_emb], dim = 0)

        # self.dt_name = args.data_dir.split('/')[-1]
        showi0 = []
        for i in range(0, 10000):
            showi0.append(i)
        self.show_index0 = torch.tensor(showi0).to(args.device)
        self.sigmoid = torch.nn.Sigmoid()
        with open(args.data_dir + 'train_data_len.pkl', 'rb') as fp:
            data_len = pickle.load(fp)
        self.states = torch.zeros(data_len, args.dim)#.to(args.device)
        # self.v_ids = torch.zeros(data_len, args.dim * 2).to(args.device)
        self.problem_ids = torch.zeros(data_len, 1).long()#.to(args.device)
        self.response_ids = torch.zeros(data_len).long()#.to(args.device)
        self.concept_ids = torch.zeros(data_len, self.max_concepts).long()#.to(args.device)
        self.max_sample_num = args.max_sample_num
        self.seq_length = args.seq_len
    
    def cell_prop(self, h, x):
        prob_id, skills, item_id, response, all_bg_index_tensor, interval_time, elapse_time = x
        data_len = prob_id.size()[0]
        
        self.states[item_id] = h.cpu()
        # print('state position:', self.states[item_id][-1])

        # skill_multihot = torch.zeros(data_len, self.concept_num).to(self.device)
        # sf_index = self.show_index0[0: data_len].unsqueeze(1).repeat(1, self.max_concepts)
        # skill_multihot[sf_index, skills] = 1.0
        # prob_representation = self.prob_rep(skill_multihot)
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb], dim = 0)
        prob_representation = prob_cat[prob_id]

        filter0 = torch.where(skills == 0, self.zeros, self.ones)
        filter_sum = torch.sum(filter0, dim = 1).float()        
        div = torch.where(filter_sum == 0, 
            torch.tensor(1.0).to(self.device), 
            filter_sum
            ).unsqueeze(1).repeat(1, self.node_dim)
        # r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
            self.concept_emb],
            dim = 0)
        related_concepts = concepts_cat[skills]
        mean_concept_emb = torch.sum(related_concepts, dim = 1) / div

        v = torch.cat(
            [mean_concept_emb,
            prob_representation],
            dim = 1)

        # self.v_represent[item_id] = v.detach()
        self.problem_ids[item_id] = prob_id.unsqueeze(1).cpu()
        self.response_ids[item_id] = response.cpu()
        self.concept_ids[item_id] = skills.cpu().cpu()


        resp_t = response.unsqueeze(1).repeat(1, self.node_dim * 2).float()
        re_resp_t = (1 - response).unsqueeze(1).repeat(1, self.node_dim * 2).float()
        gru_in = torch.cat([v.mul(resp_t),
                            v.mul(re_resp_t)], dim = 1)
        new_h = self.gru_cell(gru_in, h)
        # resp_t = response.unsqueeze(1).repeat(1, self.node_dim).float()
        # re_resp_t = (1 - response).unsqueeze(1).repeat(1, self.node_dim).float()
        # gru_in = torch.cat([prob_representation.mul(resp_t),
        #                     prob_representation.mul(re_resp_t)], dim = 1)
        # new_h = self.gru_cell(gru_in, h)
        return new_h.detach()      
    
    def update_hidden(self, inputs, this_split):
        # self.target_prepare(this_split)
        # probs = []
        seq_num = inputs[0]
        data_len = len(seq_num)
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        for i in range(0, self.seq_length):
            h = self.cell_prop(h, inputs[1][i])
           
    def forward(self, inputs, this_split):
        # self.target_prepare(this_split)
        probs = []
        seq_num = inputs[0]
        data_len = len(seq_num)
        h = torch.zeros(data_len, self.node_dim).to(self.device)

        h_list = [torch.zeros(data_len, self.node_dim * 1).to(self.device)]
        resp_list = [torch.zeros(data_len).long().to(self.device)]
        v_list = [torch.zeros(data_len, self.node_dim * 2).to(self.device)]
        for i in range(0, self.seq_length):
            h_tensors = torch.stack(h_list, dim = 1)
            resp_tensor = torch.stack(resp_list, dim = 1).float()
            v_tensor = torch.stack(v_list, dim = 1)
            h, prob, v = self.cell(h, inputs[1][i], h_tensors, resp_tensor, v_tensor)
            probs.append(prob)
            operate = inputs[1][i][3]
            h_list.append(h) 
            resp_list.append(operate)
            v_list.append(v)

        # for i in range(0, self.seq_length):
        #     h, prob = self.cell(h, inputs[1][i])
        #     probs.append(prob) 
        prob_tensor = torch.stack(probs, dim = 1)
        predict = []
        
        for i in range(0, data_len):
            predict.append(prob_tensor[i][0 : seq_num[i]])
        return torch.cat(predict, dim = 0)

    def obtain_prob_rep(self, prob_id, skills, skill_dim = 2):
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb], dim = 0)
        prob_representation = prob_cat[prob_id]

        filter0 = torch.where(skills == 0, self.zeros, self.ones)
        # filter_sum = torch.sum(filter0, dim = 1).float()  
        filter_sum = torch.sum(filter0, dim = -1).float()   
        if skill_dim == 2:     
            div = torch.where(filter_sum == 0, 
                torch.tensor(1.0).to(self.device), 
                filter_sum
                ).unsqueeze(-1).repeat(1, self.node_dim)
        elif skill_dim == 3:
            div = torch.where(filter_sum == 0, 
                torch.tensor(1.0).to(self.device), 
                filter_sum
                ).unsqueeze(-1).repeat(1, 1, self.node_dim)
        # r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
            self.concept_emb],
            dim = 0)
        related_concepts = concepts_cat[skills]
        mean_concept_emb = torch.sum(related_concepts, dim = skill_dim - 1) / div

        v = torch.cat(
            [mean_concept_emb,
            prob_representation],
            dim = -1)
        return v

    def cell(self, h, x, h_tensors, resp_tensor, v_tensor):
        prob_id, skills, item_id, response, all_bg_index_tensor, interval_time, elapse_time = x
        
        originial_ref_num = all_bg_index_tensor.size()[-1]
        all_bg_index_tensor = all_bg_index_tensor.split([self.see_ref, originial_ref_num - self.see_ref], dim = 1)[0]
        # print('all_bg_index_tensor size:', all_bg_index_tensor.size())

        data_len = prob_id.size()[0]


        '''obtain the problem representation'''
        v = self.obtain_prob_rep(prob_id, skills, skill_dim = 2)
        hidden_v = torch.cat([h, v], dim = -1)

        '''obtain the background state'''
        all_bg_hidden = self.states[all_bg_index_tensor].to(self.device)
        all_prob = self.problem_ids[all_bg_index_tensor].squeeze(-1).to(self.device)
        all_concepts = self.concept_ids[all_bg_index_tensor].to(self.device)
        all_bg_v = self.obtain_prob_rep(all_prob, all_concepts, skill_dim = 3)
        # all_bg_r = self.see_tensor[1][all_bg_index_tensor].unsqueeze(-1).repeat(1, 1, self.node_dim * 3).float().to(self.device)
        all_bg_r = self.response_ids[all_bg_index_tensor].unsqueeze(-1).repeat(1, 1, self.node_dim * 3).float().to(self.device)
        all_hidden_v = torch.cat([all_bg_hidden, all_bg_v], dim = 2)
        bg_context_resp = torch.cat([all_hidden_v.mul(all_bg_r),
                                    all_hidden_v.mul(1- all_bg_r)], dim = 2)

        # batch_bg_tensor = h.unsqueeze(1)

        heads_list = []
        '''state 之间算attention'''
        for i in range(0, self.heads_num):
            # trans_bg = torch.matmul(batch_bg_tensor, self.heads[i])
            weight = F.softmax(
                torch.matmul(self.batch_emb[i](hidden_v.unsqueeze(1)), self.bg_emb[i](all_hidden_v).transpose(2, 1)) / math.sqrt(self.node_dim * 3), dim = 2)
                # torch.matmul(self.batch_emb[i](v.unsqueeze(1)), self.bg_emb[i](all_bg_v).transpose(2, 1)) / math.sqrt(self.node_dim * 2), dim = 2)
            this_rep = torch.matmul(weight, bg_context_resp)
            heads_list.append(this_rep)
        heads_tensor_trans = torch.cat(heads_list, dim = 1).transpose(2, 1)
        bg_atts = torch.matmul(heads_tensor_trans, self.heads_map).squeeze(2)

        '''obtain the history related state， 用搜出来的算'''
        '''original code start'''
        hr_cat = torch.cat([h_tensors.mul(resp_tensor.unsqueeze(-1).repeat(1, 1, self.node_dim)),
                    h_tensors.mul((1 - resp_tensor).unsqueeze(-1).repeat(1, 1, self.node_dim))], dim = 2)
        weight = F.softmax(
                torch.matmul(v.unsqueeze(1), v_tensor.transpose(2, 1)) / math.sqrt(self.node_dim * 2), dim = 2)

        # print(weight)
        his_atts = torch.matmul(weight, hr_cat).squeeze(1)           
        '''original code end'''
     
        # peer = self.bg_integrate(bg_atts) + self.his_integrate(his_atts)
        direction_stack = torch.stack([self.bg_integrate(bg_atts), self.his_integrate(his_atts)], dim = -1)
        direction_weight = F.softmax(self.direction_weight, dim = 0)
        peer = torch.matmul(direction_stack, direction_weight).squeeze(-1)

        # predict_x = torch.cat([bg_atts, his_atts, v, h], dim = 1)
        predict_x = torch.cat([peer, v, h], dim = 1)
        probs = self.predictor(predict_x).squeeze(1)

        resp_t = response.unsqueeze(1).repeat(1, self.node_dim * 2).float()
        re_resp_t = (1 - response).unsqueeze(1).repeat(1, self.node_dim * 2).float()
        v_resp = torch.cat([v.mul(resp_t),
                            v.mul(re_resp_t)], dim = 1)
        # up_peer = self.up_peer_funcs(peer)
        gru_in = torch.cat([peer, v_resp], dim = -1)
        gru_in = self.in_map(gru_in)
        new_h = self.gru_cell(gru_in, h)
        return new_h, probs, v        

