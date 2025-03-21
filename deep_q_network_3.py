#!/usr/bin/env python

import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
# import dummy_game
from collections import deque
import random
import numpy as np
import matris_test
from replay_buffer import ReplayBuffer
import pygame

import os 
os.environ['SDL_VIDEODRIVER'] = 'dummy'

GAME = 'tetris' # the name of the game being played for log files
ACTIONS = 48 # number of valid actions
GAMMA = 0.99 # decay rate of past observations

FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
OBSERVE = 50000. # timesteps to observe before training
EXPLORE = 100000. # frames over which to anneal epsilon
def prepro(img):
    #preprocess 300*600*3 image to 100*100*1 size. (N,C,H,W)
    img[img!=0]=1
    img = img[::3,::6,0]
    img = img.T
    img = img.reshape(img.shape[0], img.shape[1],1)
    return img

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([4, 4, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 100, 100, 1])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 5) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2 # including Q values for all actions

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, network_params, s_target, readout_target, h_fc1_target, target_network_params,sess,screen):
    # define the cost function

    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1) # Q-value for this action?
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # # get the first state by doing nothing and preprocess the image to 80x80x4
    # do_nothing = np.zeros(ACTIONS)
    # do_nothing[4] = 1
    # x_t, r_0, terminal = game_state.update(0 , do_nothing)
    # x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    # s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2) # 80*80*4

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/dqn_3")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    # open up a game state to communicate with emulator
    game_state = matris_test.Matris()
    x_t = pygame.surfarray.array3d(game_state.surface)
    # x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    # s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2) # 80*80*4
    s_t = prepro(x_t)

    epsilon = INITIAL_EPSILON
    t = 0
    while 1:
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        if t<=OBSERVE:
            '''purely Observation -- action choosen state 1'''
            action = game_state.choose_action()
            a_t = game_state.encode_action(action)
            returns = game_state.update(0,a_t)
            '''purely Observation -- action choosen state 1'''
        else:
            '''After observation of the awesome AI -- action choosen state 2'''
            a_t, _ = game_state.choose_action_dqn(readout_t,epsilon)
            returns = game_state.update(0,a_t)

            if returns[2] != 0:
                print 'Awesome!!! line reduced!!!',returns[2]

            # scale down epsilon
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # for i in range(0, K):
        # run the selected action and observe next state and reward
        x_t1_col, terminal, r_t = returns
        # x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        # ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        # x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)
        s_t1 = prepro(x_t1_col)

        


        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        # update the old values
        s_t = s_t1
        t += 1

        if terminal:
            game_state = matris_test.Matris()
            x_t = pygame.surfarray.array3d(game_state.surface)
            s_t = prepro(x_t)
            
        # only train when the size is large enough
        if t > 1000:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout_target.eval(feed_dict = {s_target : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    _, readout_j1_batch[i] = game_state.choose_action_dqn(readout_j1_batch[i],0)
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            if t % 4 == 0:
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch})

            if t % 1000 == 0:
                # tau: Soft target update param
                tau = 0.001
                update_target_network_params = \
                        [target_network_params[i].assign(tf.mul(network_params[i], tau) + \
                            tf.mul(target_network_params[i], 1. - tau))
                            for i in range(len(target_network_params))]
                sess.run(update_target_network_params)
        

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + 'dqn_3/', global_step = t)
        # print info
        if (t % 1000 == 0) & (t>OBSERVE):
            print "TIMESTEP", t, "/ LINES", r_t, "/ EPSILON", epsilon, "/ Q_MAX %e" % np.max(readout_t), "/ action",game_state.decode_action(a_t)
        
        # state = ""
        # if t <= OBSERVE:
        #     state = "observe"
        # elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        #     state = "explore"
        # else:
        #     state = "train"
        # print "TIMESTEP", t, "/ STATE", state, "/ LINES", game_state.lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)
        # print "TIMESTEP", t, "/ LINES", game_state.lines, "/ EPSILON", epsilon,"/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    pygame.init()
    WIDTH = 700
    HEIGHT = 660
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    network_params = tf.trainable_variables()
    s_target, readout_target, h_fc1_target = createNetwork()
    target_network_params = tf.trainable_variables()[len(network_params):]
    # screen = pygame.Surface((WIDTH,HEIGHT))
    screen = pygame.display.set_mode((WIDTH, HEIGHT),0,32)
    trainNetwork(s, readout, h_fc1, network_params, s_target, readout_target, h_fc1_target, target_network_params, sess, screen)

def main():
    playGame()

if __name__ == "__main__":
    main()