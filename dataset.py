'''
multi + search score
有单独的response和history
数据集是从data resp里面处理的， 和data_resp 一样的train 和valid
在第二步生成训练数据的时候使用了并发
https://rebootcat.com/2017/08/20/Elasticsearch_handle_with_python/
https://blog.csdn.net/liaodaoluyun/article/details/82346086
'''
import tqdm
import os
import pickle
import logging as log
import torch
from torch.utils import data
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import math
import random
import operator
from functools import reduce
from collections import Counter
import torch.multiprocessing as mp
import time

class Dataset(data.Dataset):
    def __init__(self, problem_number, concept_num, max_sample_num, root_dir,  split='train'):
        super().__init__()
        self.seq_len = 200
        self.map_dim = 0
        self.prob_encode_dim = 0
        self.path = root_dir
        self.problem_number = problem_number
        self.concept_num = concept_num
        self.show_len = 100
        self.split = split
        # self.data_list = [[], []]
        # self.data_list = []
        self.manager = mp.Manager
        self.data_list = self.manager().list()
        # self.raw_data_list = []
        self.data_seq_list = []
        log.info('Processing data...')
        self.item_count = 1
        self.total_train_num = 0
        self.stu_count = 0
        self.sear_index = None
        self.my_txt = None
        self.max_sample_num = max_sample_num
        # self.ques_concept_relation
        self.process()
        log.info('Processing data done!')


        # self.count_0 = 0
        # self.total_pb = 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def collate(self, batch):
        seq_num, y  = [], [] #seq_num: the actual length of history record 
        # [[for _ in range(0, 4)] for _ in range(0, 200)], 
        x = []
        seq_length = len(batch[0][1][1]) # the unifrom length of hitory record
        x_len = len(batch[0][1][0][0])
        # x_len = 9
        for i in range(0, seq_length):
            this_x = []
            for j in range(0, x_len):
                this_x.append([])
            x.append(this_x)
        for data in batch:
            this_seq_num, [this_x, this_y] = data
            seq_num.append(this_seq_num)
            for i in range(0, seq_length):
                # print(i, 'iiiiiiiiiiiiiiiiiiii')
                for j in range(0, x_len):
                    # print('iiiiiiiiiiiiiiii', i, j)
                    x[i][j].append(this_x[i][j])
                # y[i].append(this_y[i])
            # y += this_y[1 : this_seq_num]
            y += this_y[0 : this_seq_num]
            # y += this_y
        batch_x, batch_y =[], []
        # problem, related_concept_index, operate, filter0, prob_ids, _, _
        # prob_id, skills, item_id, response, all_bg_index_tensor, own_same_concept_index
        for i in range(0, seq_length):
            x_info = []
            for j in range(0, x_len):
                x_info.append(torch.tensor(x[i][j]))
                # if j == 1 or j == 5 or j == 6:    
                #     x_info.append(torch.tensor(x[i][j]))
                # elif j == 4:
                #     x_info.append(torch.tensor(x[i][j]).long())
                # else:
                #     x_info.append(torch.tensor(x[i][j]).float())
            batch_x.append(x_info)
        return [torch.tensor(seq_num), batch_x], torch.tensor(y).float()

    def getbackgroud(self, previous_txt, skills, response, problem):
        g_skills = skills.copy()
        g_skills.sort()
        add_ptxt = 'p' + str(problem) + ' '
        skill_str = ''
        full_skill_str = ''
        for s in g_skills:
            full_skill_str += str(s) + '-'
            if s != 0:
                add_ptxt += 'c' + str(s) + ' '
                skill_str += 'c' + str(s) + ' '
        
        if response == 1:
            previous_txt  = add_ptxt + 'R ' + previous_txt
        elif response == 0:
            previous_txt  = add_ptxt + 'W ' + previous_txt
        # previous_txt  = add_ptxt + previous_txt
        
        return previous_txt, skill_str[:-1], full_skill_str[:-1]

    def get_pickle_id(self, results):
        id_list = []
        score_list = []
        for item in results:
            this_id = item['_source']['pickle_id']
            if item['_score'] is None:
                this_score = -1000
            else:
                this_score = float(item['_score'])
            id_list.append(this_id)
            score_list.append(this_score)
        return id_list, score_list
    
    def get_difffques_id(self, results):
        ques_list = []
        resp_list = []
        # score_list = []
        for item in results:
            ques_id = item['_source']['problem']
            ques_resp = item['_source']['response']
            # if item['_score'] is None:
            #     this_score = -1000
            # else:
            #     this_score = float(item['_score'])
            ques_list.append(ques_id)
            resp_list.append(ques_resp)
            # score_list.append(this_score)
        return ques_list, resp_list#, score_list

    def pick_index(self, c_txt, sq_id, stu_id): 
        c_query = {'query':{
                            'bool' : 
                            {
                                'must' : 
                                [
                                 
                                    { 'match': {'skills': c_txt}},
                                    { 'term': {'user':  stu_id}}, 
                                    { 'range': {
                                                'sq_id': {
                                                    'lt': sq_id
                                                }
                                    }}
                                ]
                            }
                            }
                    }
        dt_name = self.path.split('/')[-2]
        heads = {'index': dt_name + '_train_search'}
        this_action = [heads, c_query]
        
        return this_action
      
    def question_action_dict(self, prob_id, bg_txt, stu_id):
        first_query = {'query':{
                            'bool' : 
                            {
                                'must' : 
                                # 'should':
                                [
                                    # { 'term': {'problem': int(prob_id.cpu().numpy())}},
                                    { 'term': {'problem': prob_id}},  
                                    { 'match': {'history': bg_txt}} 
                                ],
                                'must_not': {'term': {'user':  stu_id}}
                            }
                            }
                        }
        
        heads = {'index': self.sear_index}
        this_action = [heads, first_query]
        return this_action

    def concepts_action_dict(self, c_txt, bg_txt, stu_id):
        second_query = {'query':{
                            'bool' : 
                            {
                                # 'must' : 
                                'should':
                                [
                                    { 'term': {'skills': c_txt}}, 
                                    { 'match': {'history': bg_txt}} 
                                ],
                                'must_not':
                                {'term': {'user':  stu_id}}
                            }
                            }
                        }

        heads = {'index': self.sear_index}
        this_action = [heads, second_query]
        return this_action


    def obtain_bg_action(self, prob_id, item_id, stu_id):
        # bg_index = self.search(prob_id, item_id, stu_id)
        bg_txt, c_txt = self.my_txt[item_id]
        if len(bg_txt) > 1024:
            bg_txt = bg_txt[-1024:]
        question_action = self.question_action_dict(prob_id, bg_txt, stu_id)
        concepts_action = self.concepts_action_dict(c_txt, bg_txt, stu_id)
        return question_action, concepts_action

        
        

    def data_reader(self, stu_records, es, stu_id, actual_len, e_index):
        '''
        @params:
            stu_record: learning history of a user
        @returns:
            x: list of response(0 or 1), embedding of problem, user embedding, related_concept_index
            y: response
        '''
        # print('stuid:', stu_id)
        x_list = []
        y_list = []
        # background_tensor_list = []
        # background_noresp_list = []
        # response_list = []
        background_txt_list = []
        # background_ques_seq_list = []
        # background_resp_seq_list = []
        # question_seq = [0] * self.seq_len
        # resp_seq = [-1] * self.seq_len
        '''initial previous'''
        # previous_true, previous_all, 
        previous_txt, skill_str = '', ''
        # for i in range(0, actual_len):
        sq_id = 0
        for i in range(0, len(stu_records)):
            # _, problem_id,skills, response= stu_records[i]
            problem_id, skills, interval_time, elapse_time, response = stu_records[i]
            # this_x = [problem_id, skills, self.item_count, response]
            this_x = None
            this_y = response
            if i < actual_len:
                # this_x = [problem_id, skills, self.item_count, response, sq_id, stu_id]
                this_x = [problem_id, skills, self.item_count, response, sq_id, stu_id, interval_time, elapse_time]
                x_list.append(this_x)
                y_list.append(this_y)
                # self.data_list.append([this_x, this_y])
                
                # ptrue_tensor = torch.tensor(previous_true).float()
                # pall_tensor = torch.tensor(previous_all).float()
                # background_tensor_list.append(torch.cat([ptrue_tensor, pall_tensor], dim = 0))
                # true_all_cat = torch.cat([ptrue_tensor, pall_tensor], dim = 0)
                # background_noresp_list.append(true_all_cat)
                # response_list.append(torch.tensor(response))
                current_his_txt, skill_str, full_skill_str = self.getbackgroud(previous_txt, skills, response, problem_id)
                background_txt_list.append([previous_txt, skill_str])

                '''insert data to db'''
                skill_dict = dict()
                skill_dict['skills'] = skill_str
                
                other_dict = {'problem': problem_id,
                        'user': stu_id,
                        'sq_id': sq_id,
                        'history': previous_txt,
                        'pickle_id': self.item_count,
                        'response': response, 
                        'full_skill': skills}
                        # 'full_skill_str': full_skill_str}
                this_dict = {**skill_dict, ** other_dict}
                sq_id += 1
                
                previous_txt = current_his_txt
                ####################
                
                if self.split == 'train':
                    es.index(index=e_index[0], doc_type='train', id=self.item_count,body=this_dict)
                ######################
                
                self.item_count += 1
            else:
                # this_x = [problem_id, skills, 0, response, sq_id, stu_id]
                # this_x = [problem_id, skills, 0, response, sq_id, stu_id, interval_time, elapse_time]
                this_x = [problem_id, skills, 0, 0, sq_id, stu_id, interval_time, elapse_time]
                x_list.append(this_x)
                y_list.append(this_y)

        # return background_noresp_list, response_list, background_ques_seq_list, background_resp_seq_list, background_txt_list, 
        return background_txt_list, x_list, y_list

    def target_prepare(self, my_txt):
        dt_name = self.path.split('/')[-2] #db_name + '_train_search', db_name + '_valid_search'
        self.sear_index = dt_name + '_train_search'
        self.my_txt = my_txt
    
    def merge_ques_concept_list(self, ques_index, ques_score, concept_index, concept_score):
        final_index = ques_index
        final_score = ques_score
        con_len = len(concept_index)
        for i in range(0, con_len):
            this_con_index = concept_index[i]
            this_con_score = concept_score[i]
            if this_con_index in final_index:
                this_pos = final_index.index(this_con_index)
                this_final_score = final_score[this_pos]
                if this_final_score < this_con_score:
                    final_score[this_pos] = this_con_score
            else:
                final_index.append(this_con_index)
                final_score.append(this_con_score)
        return final_index, final_score

    # def get_different_questions(diffques_indexes, diffques_score):
    #     TODO
    def bulk_seach(self, es, action_list):
        '''type == same: serach the record to answer similar question, else search the records of different question'''    
        # if type == 'same':
        #     id_identify_funcs = self.get_pickle_id
        # else:
        #     id_identify_funcs = self.get_difffques_id
        result = es.msearch(body = action_list)
        # indexes = []
        # scores = []
        first_return, second_return = [], []
        for r in result['responses']: #针对multisearch里所有的每个search
            # item, score = self.get_pickle_id(r['hits']['hits'])
            first, second = self.get_pickle_id(r['hits']['hits'])
            first_return.append(first)
            second_return.append(second)
        return first_return, second_return
    # def process_diff_ques(problem_id, my_diffques_indexes, my_diffques_resp):
        
    def raw_to_final(self, data):
        actual_len = data[0]
        this_x, this_y = [], []
        org_x, org_y = data[1] 
        # for record in org_x:
        item_sq_map = dict()
        ques_action_list, con_action_list = [], []
        #  = []
        # rnn_index_action_list = []
        # bg_index_list = []
        
        # for ir in range(0, len(org_x)):
        for ir in range(0, actual_len):
            record = org_x[ir]
            # print(record)
            '''
            sq_id: 在序列中的编号
            stu_id: 序列所代表的学生的id
            '''
            # problem_id, skills, item_num, response, sq_id, stu_id = record
            problem_id, skills, item_num, response, sq_id, stu_id, interval_time, elapse_time = record
            item_sq_map[item_num] = sq_id
            # bg_index = None
            # own_same_concept_index = []
            # if ir < actual_len:
                # bg_index = self.obtain_bg_index(problem_id, item_num, stu_id)

            '''
            bg_ques_action: 其他用户做了这个这个问题的记录
            bg_con_action: 其他用户做了相关concept的记录
            '''    
            bg_ques_actions, bg_con_actions = self.obtain_bg_action(problem_id, item_num, stu_id)

            # skill_str = self.my_txt[item_num][1]
            # pick_rnn_index_actions = self.pick_index(skill_str, sq_id, stu_id)

            ques_action_list += bg_ques_actions
            con_action_list += bg_con_actions
            
        '''parallel search'''
        es = Elasticsearch(['localhost:9200'], timeout=30)
        # TODO 11-23 0:29做到这里
        ques_indexes, ques_score = self.bulk_seach(es, ques_action_list)
        concepts_indexes, concepts_score = self.bulk_seach(es, con_action_list)
        # diffques_indexes, diffques_resp = self.bulk_seach(es, diff_ques_action_list, type='diff')

        for ir in range(0, len(org_x)):
            record = org_x[ir]
            problem_id, skills, item_num, response, sq_id, stu_id, interval_time, elapse_time = record
            bg_index, bg_score = None, None
            if ir < actual_len:
                bg_index, bg_score = self.merge_ques_concept_list(ques_indexes[ir], ques_score[ir], concepts_indexes[ir], concepts_score[ir])
                current_len = len(bg_index)
                '''去掉多余的或者不足数的做填充'''
                if current_len < self.max_sample_num:
                    bg_index += [0] * (self.max_sample_num - current_len)
                    bg_score += [-1000] * (self.max_sample_num - current_len)
                elif current_len > self.max_sample_num: 
                    bg_index = bg_index[0: self.max_sample_num]
                    bg_score = bg_score[0: self.max_sample_num]
                
            else:
                bg_index = [0] * self.max_sample_num
                bg_score = [-1000] * self.max_sample_num
                # my_diffques_indexes = [0] * self.max_sample_num
                # my_diffques_resp = [0] * self.max_sample_num
                # my_actual_diffques_tag = [0] * self.max_sample_num

            this_x.append([problem_id, skills, item_num, response, bg_index, interval_time, elapse_time])
            
        self.data_list.append([actual_len, [this_x, org_y]])
        es.close()

    def process(self):
        '''index 要分数据集'''
        # es = Elasticsearch(['127.0.0.1:9200'],ignore=[201, 200, 405, 502])
        es = Elasticsearch(['localhost:9200'])
        dt_name = self.path.split('/')[-2]
        # e_index = [dt_name + '_train_search', dt_name + '_valid_search']
        e_index = [dt_name + '_train_search']
        # e_doctype = ['train_doc', 'valid_doc']
        if self.split == 'train':
            for ei in e_index:
                if es.indices.exists(index=ei, ignore=400):
                    es.indices.delete(index=ei, ignore=[400, 404])
                es.indices.create(index=ei)
        
        self.prob_encode_dim = int(math.log(self.problem_number,2)) + 1
        with open(self.path + 'history_' + self.split + '.pkl', 'rb') as fp:
            histories = pickle.load(fp)
        loader_len = len(histories.keys())
        self.total_train_num = loader_len
        self.stu_count = 0
        log.info('loader length: {:d}'.format(loader_len))
        # proc_count = 0
        # all_backrgound_tensor_list = [torch.zeros(self.concept_num * 2)]
        # response_tensor_list = [torch.tensor(0)]
        # ques_seq_all_list = [torch.tensor([0] * self.seq_len)]
        # resp_seq_all_list = [torch.tensor([-1] * self.seq_len)]
        all_txt_list = [['', '']]
        train_valid_id_gap = 0
        if self.split == 'valid':
            train_valid_id_gap = 500000
        for k in tqdm.tqdm(histories.keys()):
            stu_record = histories[k]
            if stu_record[0] < 10:
                continue
            # self, stu_records, es, stu_id, actual_len, e_index
            
            # background_noresp_list, response_list, \
            #             background_ques_seq_list, background_resp_seq_list,\
            #             background_txt_list, 
            background_txt_list, x_list, y_list  = self.data_reader(stu_record[1], es, int(k) + train_valid_id_gap, stu_record[0], e_index)
            # background_tensor_list, background_txt_list, x_list, y_list  = self.data_reader(stu_record[1], es, k + train_valid_id_gap, stu_record[0], e_index)
            # print('len x_list: ', len(x_list), 'len y_list: ', len(y_list))
            if x_list != [] and y_list != []:
                self.data_seq_list.append([stu_record[0], [x_list, y_list]])
            self.stu_count += 1
            
            # all_backrgound_tensor_list += background_noresp_list
            # response_tensor_list += response_list
            # ques_seq_all_list += background_ques_seq_list
            # resp_seq_all_list += background_resp_seq_list
            all_txt_list += background_txt_list
            # proc_count += 1
        

        # self.target_prepare(self.split, all_txt_list)
        with open(self.path + self.split + '_data_len.pkl', 'wb') as fp:
            pickle.dump(self.item_count, fp)
        self.target_prepare( all_txt_list)

        '''not multiprocess'''
        for data in tqdm.tqdm(self.data_seq_list):
            self.raw_to_final(data)
            
        '''start multiprocess'''
        
        print('data length after multprocess:', len(self.data_list))

        # print('data list type before:', type(self.data_list))
        self.data_list = list(self.data_list)
        # print('data list type after:', type(self.data_list))
        log.info('final length {:d}'.format(len(self.data_list)))
