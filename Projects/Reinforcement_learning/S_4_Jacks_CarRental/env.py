import numpy as np
import math

class Environment():

    def __init__(self):
        self.lambda_in = np.array([3,2])
        self.lambda_req = np.array([3,4])
        self.max_car = 20
        self.action_list = list(range(-5,6))
        print(self.action_list)
        self.prob_matrix_0 = None
        self.prob_matrix_1 = None
        self.reward_matrix_0 = None
        self.reward_matrix_1 = None
        self.gen_matrix()
        self.req_reward = 10
        self.tranfer_penalty = 2
        # print(self.prob_matrix_0[:,:,5].shape)
        # print(self.prob_matrix_0[4,:,:])
        # print_str = "    "
        # for i in range(-5,6):
        #     print_str += "%2d   " % (i)
        # for i in range(21):
        #     print_str +="\n"
        #     print_str += "%3d " %(i)
        #     for j in range(11):
        #         print_str += "%0.2f " %(self.prob_matrix_0[5,i,j])
        #
        # print(print_str)
        # print(np.sum(self.prob_matrix_0[:,:,10],axis=1))
        # print(np.sum(self.prob_matrix_1[:,:,10], axis=1))

    def gen_matrix(self):
        mat_size = self.max_car + 1
        act_size =len(self.action_list)
        self.prob_matrix_0 = np.zeros((mat_size,mat_size,act_size))
        self.prob_matrix_1 = np.zeros((mat_size, mat_size, act_size))
        self.reward_matrix_0 = np.zeros((mat_size, mat_size, act_size))
        self.reward_matrix_1 = np.zeros((mat_size, mat_size, act_size))

        lambda_in_by_n = np.ones((2,mat_size))

        lambda_in_exp = np.exp(-1.0*self.lambda_in)
        lambda_req_exp = np.exp(-1.0 * self.lambda_req)

        for n in range(1,mat_size):
            lambda_in_by_n[0][n] = lambda_in_by_n[0][n-1]*self.lambda_in[0]/n
            lambda_in_by_n[1][n] = lambda_in_by_n[1][n-1]*self.lambda_in[1]/n

        lambda_req_by_n = np.ones((2, mat_size))

        for n in range(1, mat_size):
            lambda_req_by_n[0][n] = lambda_req_by_n[0][n - 1] * self.lambda_req[0] / n
            lambda_req_by_n[1][n] = lambda_req_by_n[1][n - 1] * self.lambda_req[1] / n


        #for station 0
        # if action is positive car tranfered from station 0 to station 1 by that value
        # else if action is negative car tranfered from station 1 to station 0 by that value

        def gen_station_prob(station_id,prob_mat,reward_mat):
            direction = None
            if station_id==0:
                direction = -1.0
            else:
                direction = 1.0

            for cur_car_count in range(mat_size):
                for cur_req_count in range(mat_size):
                    cur_req_prob = lambda_req_exp[station_id]*lambda_req_by_n[station_id][cur_req_count]
                    for cur_in_count in range(mat_size):
                        cur_in_prob = lambda_in_exp[station_id] * lambda_in_by_n[station_id][cur_in_count]
                        cur_prob = cur_req_prob * cur_in_prob
                        for cur_action_id in range(act_size):
                            req_full_filled = min(cur_car_count + cur_in_count,cur_req_count)
                            req_reward = req_full_filled*self.req_reward
                            cur_action = self.action_list[cur_action_id]
                            new_car_count = cur_car_count - cur_req_count + cur_in_count
                            transfer_panalty = 0
                            if station_id==0:
                                if cur_action > 0:
                                    car_transfered = min(cur_action,new_car_count)
                                    transfer_panalty = self.tranfer_penalty*car_transfered
                            else:
                                if cur_action < 0:
                                    car_transfered = min(-cur_action,new_car_count)
                                    transfer_panalty = self.tranfer_penalty * car_transfered

                            new_car_count += direction * cur_action


                            if new_car_count > self.max_car:
                                new_car_count = self.max_car

                            if new_car_count < 0:
                                new_car_count = 0

                            prob_mat[cur_car_count][new_car_count][cur_action] += cur_prob
                            reward_mat[cur_car_count][new_car_count][cur_action] += cur_prob*cur_prob


        gen_station_prob(0,self.prob_matrix_0,self.reward_matrix_0)
        gen_station_prob(1,self.prob_matrix_1,self.reward_matrix_1)

    def get_prob(self, s_0, s_1, new_s_0, new_s_1, actionid):
        return self.prob_matrix_0[s_0][new_s_0][actionid]*self.prob_matrix_1[s_1][new_s_1][actionid]

    def get_reward(self, s_0, s_1, new_s_0, new_s_1, actionid):
        return self.reward_matrix_0[s_0][new_s_0][actionid]*self.reward_matrix_1[s_1][new_s_1][actionid]


Environment()

