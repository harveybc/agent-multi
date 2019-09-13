# agent_dcn_v4: Uses RETURN DE MACD ADELANTADO 10 ticks(signal 8 from q-datagen_c_v4) regression signal to decide action 
# v4 do not use timedistributed and swap axes of inputs and also seed the random seed for reproducible results

# seed numpy random number generator to enable reproducible results
print("Seed numpy random number generator")
from numpy.random import seed
seed(1)
print("Seed tensorflow random number generator")
from tensorflow import set_random_seed
set_random_seed(2)

import gym
import gym.wrappers
import gym_forex
from gym.envs.registration import register
import sys
import neat
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from joblib import load
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import operator
from numpy import genfromtxt
import csv
from sklearn import svm
from operator import add, sub
from joblib import dump, load
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import SGD, Adamax
import copy
            
import random
            
## \class QAgent
## \brief Q-Learning agent that uses an OpenAI gym environment for fx trading 
## estimating for each tick, the optimal SL, TP, and Volume.
class QAgent():    
    ## init method
    ## Loads the validation dataset, loads the pre-trained models
    #  initialize forex environment.
    def __init__(self):
        # percentage of noise to add to an action
        # TODO: cambiar acciones para que solo se cierren ordenes por SL o TP (dep de volatility)
        self.noise = 0.0
        # TODO: probar con órdenes con duración mínima en ticks (solo se puden cerrar por TP/SL y por acttion si se ha superado el min_duartion)
        # noise0, min_duratopn = 0          bal=241k
        # noise0, min_duration = 20         bal=43k
        # noise 0.25, min_duration = 20     bal=1k
        self.duration = 5
        self.min_duration = 0
        self.th_open = 0.2
        self.th_close = 0.1
        
        # TODO: probar con órdenes que solo se cierran por SL/TP
        # TODO: hacer gridsearch de SL/TP
        # TODO: en caso ideal sin ruido, probar si ganancia incrementa con volumen controlado por volatility
        # TODO: probar si mejora SL/TP controlados por volatilidad respecto a los mejores fijos encontrados por gridsearch
        # First argument is the validation dataset, including headers indicating maximum and minimum per feature
        self.vs_f = sys.argv[1]
        # Second argument is the prefix (including path) for the dcn pre-trained models 
        # for the actions, all modes are files with .svm extention and the prefix is
        # concatenated with a number indicating the action:
        # 0 = Buy/CloseSell/nopCloseBuy
        # 1 = Sell/CloseBuy/nopCloseSell
        # 2 = No Open Buy
        # 3 = No Open Sell
        self.model_prefix = sys.argv[2]
        # third argument is the path of the datasset to be used in the gym environment (not q-datagen generated, without headers) 
        self.env_f = sys.argv[3]
        # initialize gym-forex env (version 4)
        self.test_episodes = []
        self.generation = 0
        self.min_reward = -15
        self.max_reward = 15
        self.episode_score = []
        self.episode_length = []
        self.svr_rbf = svm.SVR(kernel='rbf')
        self.num_s = 19
        self.model = [self.svr_rbf] * self.num_s 
        
        self.max_index = 0
        self.vs_data = []
        self.vs_num_ticks = 0
        self.vs_num_columns = 0
        self.obsticks = 30
        self.window_size = self.obsticks 
        # TODO: obtener min y max de actions from q-datagen dataset headers
        self.min_TP = 50
        self.max_TP = 1000
        self.min_SL = 50
        self.max_SL = 1000 
        self.min_volume = 0.0
        self.max_volume = 0.1
        self.security_margin = 0.1
        self.test_action = 0
        self.num_f = 0
        self.num_features = 0
        self.action_prev = [0]
        self.action = [0]
        self.raw_action = [0]
        # load pre-processing settings 
        self.pt = preprocessing.PowerTransformer()
        print("loading pre-processing.PowerTransformer() settings for the generated dataset")
        self.pt = load(self.vs_f+'.powertransformer')
        # load feature-selection mask
        print("loading pre-processing feature selection mask")
        self.mask = load(self.vs_f+'.feature_selection_mask')
        # variables for output csv files for observations and prediction
        self.out_obs = []
        self.out_act = []
        # register the gym-forex openai gym environment
        # TODO: extraer obs_ticks como el window_size, desde los headers de  salida de q-datagen
        register(
            id='ForexValidationSet-v1',
            entry_point='gym_forex.envs:ForexEnv6',
            kwargs={'dataset': self.env_f ,'max_volume':self.max_volume, 'max_sl':self.max_SL, 
                    'max_tp':self.max_TP, 'min_sl':self.min_SL,
                    'min_tp':self.min_TP,'obsticks':self.obsticks, 
            'capital':800, 'leverage':100, 'num_features': 13}
        )
        # make openai gym environments
        self.env_v = gym.make('ForexValidationSet-v1')
        # Shows the action and observation space from the forex_env, its observation space is
        # bidimentional, so it has to be converted to an array with nn_format() for direct ANN feed. (Not if evaluating with external DQN)
        print("action space: {0!r}".format(self.env_v.action_space))
        print("observation space: {0!r}".format(self.env_v.observation_space))
        # read normalization maximum and minimum per feature
        # n_data_full = genfromtxt(self.vs_f, delimiter=',',dtype=str,skip_header=0)    
        with open(self.vs_f, newline='') as f:
            reader = csv.reader(f)
            n_data = next(reader)  # gets the first line
        # read header from vs_f
        #n_data = n_data_full[0].tolist()
        self.num_columns = len(n_data)
        print("vs_f num_columns = ", self.num_columns)
        # minimum and maximum per feature for normalization before evaluation in pretrained models
        self.max = [None] * self.num_columns
        self.min = [None] * self.num_columns
        for i in range(0, self.num_columns-self.num_s):
            header_cell = n_data[i]
            #print("header_cell = ", header_cell, "type = " ,type(header_cell))
            data = header_cell.split("_")
            num_parts = len(data)
            self.max[i] = float(data[num_parts-1])
            self.min[i] = float(data[num_parts-2])
            # data was mormalized as: my_data_n[0, i] = (2.0 * (my_data[0, i] - min[i]) / (max[i] - min[i])) - 1
    
    ## Generate DCN  input matrix
    def dcn_input(self, data):
        #obs_matrix = np.array([np.array([0.0] * self.num_features)]*len(data), dtype=object)
        obs_matrix = []
        obs_row = []
        obs_frame = []
        obs = np.array([np.array([0.0] * self.num_features)] * self.window_size)
        # for each observation
        data_p = np.array(data)
        num_rows = len(data)
        # counter of rows of data array
        c_row = 0
        while c_row < num_rows:
            # invert the order of the observations, in the first element is the newest value
            obs_frame = []
            for j in range(0,self.window_size):
                # create an array of size num_features 
                obs_row = []
                for k in range(0,self.num_features):
                    obs_row.append(data_p[c_row, k*self.window_size + j ])
                # obs_frame contains window_size rows with num_features columns with the newest observation in cell[0]
                obs_frame.append(copy.deepcopy(obs_row))
            # obs_matrix contains files with observations of size (window_Size, num_features)
            obs_matrix.append(copy.deepcopy(obs_frame))
            c_row = c_row + 1
        #print("Formating of data for DCN input performed succesfully.")
        return np.array(obs_matrix)

    ## the action model is the same q-datagen generated dataset
    def load_action_models(self, signal):
        self.svr_rbf = load_model(self.model_prefix + str(signal)+'.dcn')
        # get the number of observations
        self.vs_data = genfromtxt(self.vs_f, delimiter=',')
        self.vs_num_ticks = len(self.vs_data)
        self.vs_num_columns = len(self.vs_data[0])
        self.num_f = self.vs_num_columns - self.num_s
        self.num_features = self.num_f // self.window_size
        self.num_ticks = self.vs_num_ticks

    ## For an observation for each tick, returns 0 if the slope of the future(10) MACD signal (output 16 zero-based) is negative, 1 if its positive. 
    def decide_next_action(self, normalized_observation):
        # TODO: evaluar el modelo de regresion y retornar como action un arreglo con el valor predicho por cada modelo. (0= clasif,1=regresion)
        # evaluate all models with the observation data window 
        self.action_prev = copy.deepcopy(self.action)
        self.action = []
        self.max_index = 0 
        action_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        vs = np.array(normalized_observation)
        # evaluate all models with the observation data window 
        self.action = []
        vs = np.array(normalized_observation)
        vs_r = np.reshape(vs, (1, -1))
        #print ("vs_r = ",vs_r)
        obs = self.dcn_input(vs_r)
        np.set_printoptions(threshold=sys.maxsize)
        obs = np.swapaxes(obs, 1, 2)
        #print("obs = ", obs)
        #print("obs.shape = ", obs.shape)
        action_list[0] = self.svr_rbf.predict(obs)
        print("action_list[0] = ", action_list[0])
        # TODO: Add observation to output csv file array, Quitar cuando pretreiner y agent_dcn tengan las mismas salidas y entradas
        # TODO: Add Normalized obervation to test if the cdn_input function is working well
        self.out_obs.append(copy.deepcopy(obs))
        #self.out_obs.append(obs)
        # TODO: Add action to output csv file array, Quitar cuando pretreiner y agent_dcn tengan las mismas salidas y entradas
        self.out_act.append(copy.deepcopy(action_list[0][0]))
        # seto the returned action to actionlist
        self.action = copy.deepcopy(action_list)
        #print("action=",self.action)

        return self.action

    ## normalize the observation matrix, converts it to a list feedable to a pretrained DcN
    # oldest data is first in dataset and also in observation matrix
    # input obs_matrix, prev obs_matrix, output:row
    def normalize_observation(self, observation, observation_prev):
        # observation is a list with size num_features of numpy.deque of size 30 (time window) 
        # TODO: PORQUE num_columns_o es 29?
        n_obs = [] 
        l_diff = []
        #print("observation = ", observation)
        num_columns_o = len(observation)
        # TODO: Cambiar a recorrido de l_obs restando el anterior y solo usar l_obs_prev para el primer elemento
        for i in range (0, num_columns_o):
            l_obs = list(observation[i])   
            l_obs_prev = list(observation_prev[i])   
            for j in range (0, self.window_size):
                diff = l_obs[j] - l_obs_prev[j]
                l_diff.append(diff)
            for l in l_obs:
                n_obs.append(l)
                
        for l in l_diff:
            n_obs.append(l) 
        n_obs_n = np.array(n_obs).reshape(1,-1)
        n_obs_o = self.pt.transform(n_obs_n)
        n_o = n_obs_o[0].tolist()
        n_obs=np.array(n_o)
        n_obs = n_obs[self.mask]
        return n_obs  
    
    ## Function transform_action: convert the output of the raw_action into the
    ## denormalized values to be used in the simulation environment.
    ## increase the SL in the sec_margin% and decrease the TP in the same %margin, volume is also reduced in the %margin  
    def transform_action(self, order_status):
        # order_status:  0 nop, -1=sell,1=buy
        # the variable self.raw_action contains the output of decide_next_action, which is an array of 3 values, MACD signal return, RSI return and MACD main - signal >0?
        # the output actions are: 0=TP,1=SL,2=volume(dInv). 
        # if there is no opened order
        act = []
        # initialize values for next order , dir: 1=buy, -1=sell, 0=nop
        dire = 0.0
        tp = 1.0
        tp_a=tp
        sl = 1.0
        vol  = 1.0
        
        action_diff = self.raw_action[self.test_action] - self.action_prev[self.test_action]
        # TODO: if there is an opened order, increases de duration counter, else set it to 0
        if (order_status==0):
            self.duration = 0
        else:
            self.duration = self.duration + 1
        # TODO: add min_duration constraint to evaluate if closing an open order with an action
        # if there is no opened order
        if order_status == 0:
            # si el action[0] > 0, compra, sino vende
            if (self.raw_action[0] > 0.3):
                # opens buy order  
                dire = 1.0
                tp_a = 0.1
            if (self.raw_action[0] < -0.3):
                # opens sell order  
                dire = -1.0
                tp_a = 0.1
        # if there is an existing buy order
        if (order_status == 1) and (self.duration > self.min_duration):
            # si action[0] == 0 cierra orden de buy 
            if (self.raw_action[0] < 0):
                # closes buy order  
                dire = -1.0
        # if there is an existing sell order               
        if (order_status == -1) and (self.duration > self.min_duration):
            # if action[0]>0, closes the sell order
            if (self.raw_action[0] > 0):
                # closes sell order  
                dire = 1.0 
        # verify limits of sl and tp, TODO: quitar cuando estén desde fórmula
        sl_a = 1.0
            
        # Create the action list output [tp, sl, vol, dir]
        act.append(tp_a)
        # TODO: en el simulador, implmeentar min_tp ysl
        act.append(sl_a)
        act.append(vol)  
        act.append(dire)
        return act
    
    ## Evaluate all the steps on the simulation choosing in each step the best 
    ## action, given the observations per tick. 
    ## \returns the final balance and the cummulative reward
    # Posssible actions: 
    # 0 = Buy/CloseSell/nopCloseBuy
    # 1 = Sell/CloseBuy/nopCloseSell
    # 2 = No Open Buy
    # 3 = No Open Sell
    def evaluate(self, max_ticks):
        # calculate the validation set score
        hist_scores = []
        #perform first observation
        observation = self.env_v.reset()
        #print("observation = ", observation)
        observation_prev = copy.deepcopy(observation) 
        # action = nop
        action = []
        # initialize values for next order , dir: 1=buy, -1=sell, 0=nop
        dire = 0.0
        tp = 1.0
        sl = 1.0
        vol  = 1.0
        score = 0.0
        step = 1
        order_status=0
        equity=[]
        balance=[]        # Create the action list output [tp, sl, vol, dir]
        action.append(tp)
        action.append(sl)
        action.append(vol)  
        action.append(dire)
        #do second observation tobe able to normalize following observations
        observation, reward, done, info = self.env_v.step(action)
        order_status=info['order_status']
        equity.append(info['equity'])
        balance.append(info['balance'])
        
        #TODO: ERROR
        # normalize observation appended with its return obtained from previous and current observations
        normalized_observation = agent.normalize_observation(observation, observation_prev) 
        #print("normalized_observation = ", normalized_observation)

        while 1:
            step += 1
            # si el step > 2, hacce el resto, sono usa vector e zeros como accion 
            observation_prev = copy.deepcopy(observation) 
            # TODO: Test, quitar cuando coincidan observations de agent_dcn y pretrainer
            #if step > 1:
            #    #print("a=", raw_action[0], " order_status=",info['order_status'], " num_closes=", info['num_closes']," balance=",info['balance'], " equity=", info['equity'])
            #    print("observation")
            
            if (step < ((3*self.num_ticks)//4)+3) or (step > (self.vs_num_ticks-self.obsticks)):
                #print ("Skippig limits, step = ", step)
                # action = nop
                action = []
                # initialize values for next order , dir: 1=buy, -1=sell, 0=nop
                dire = 0.0
                tp = 1.0
                sl = 1.0
                vol  = 1.0
                # Create the action list output [tp, sl, vol, dir]
                action.append(tp)
                action.append(sl)
                action.append(vol)   
                action.append(dire)
            else:
                self.raw_action = self.decide_next_action(normalized_observation)
                action = self.transform_action(order_status)
                equity.append(info['equity'])
                balance.append(info['balance'])
            
            observation, reward, done, info = self.env_v.step(action)
            order_status=info['order_status']
            
            
            # TODO: Hacer gráfico de balance y equity
            if (step < ((3*self.num_ticks)//4)+3) or (step > (self.vs_num_ticks-self.obsticks)):
                normalized_observation = normalized_observation
            else: 
                normalized_observation = self.normalize_observation(observation, observation_prev)
            score += reward
            #env_v.render() 
            if done or (step > max_ticks):
                break

        # TODO : Hacer skip de valores leídos por agent hasta el primero del vs
        # TODO: export output csv with observations for the validation set, Quitar cuando pretrainer y agent_dcn tengan las mismas obs y act
        out_obs_n = np.array(self.out_obs)
        print("out_obs_n.shape = ", out_obs_n.shape)
        with open('a_output_obs.csv' , 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(out_obs_n)
        # TODO: Add action to output csv file array, Quitar cuando pretreiner y agent_dcn tengan las mismas salidas y entradas
        print("Finished generating validation set observations.")
        # export output csv with actions for the validation set
        with open('a_output_act.csv' , 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(self.out_act)
        print("Finished generating validation set actions per observation.")
        lw = 2
        y_rbf = balance
        y_v = equity
        x_seq = list(range(0, len(balance)))
        fig=plt.figure()
        plt.plot(x_seq, y_v, color='darkorange', label='Equity')
        plt.plot(x_seq, y_rbf, color='navy', lw=lw, label='Balance')
        plt.xlabel('tick')
        plt.ylabel('value')
        plt.title('Performance')
        plt.legend()
        fig.savefig('agent_test_8.png')
        #plt.show()
        
        hist_scores.append(score)
        avg_score = sum(hist_scores) / len(hist_scores)
        print("Validation Set Score = ", avg_score)
        print("*********************************************************")
        return info['balance'], avg_score     

    def show_results(self):
        test=0

# main function 
if __name__ == '__main__':
    agent = QAgent()
    #agent.svr_rbf = agent.set_dcn_model()
    training_signal = 7
    agent.load_action_models(training_signal)
    scores = []
    balances = []
    for i in range(0, 1):
        print("Testing signal ",training_signal +i)
        agent.test_action = i
        agent.load_action_models(training_signal)
        balance,score = agent.evaluate(10000000)
        scores.append(score)
        balances.append(balance)
    print("Results:")
    for i in range(0, 1):
        print("Signal ", 8+i, " balance=",balances[i], " score=",scores[i])
        