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

FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others


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
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess,screen):
    # define the cost function
    OBSERVE = 5000. # timesteps to observe before training
    EXPLORE = 10000. # frames over which to anneal epsilon
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
    checkpoint = tf.train.get_checkpoint_state("saved_networks/dqn_2")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    # open up a game state to communicate with emulator
    game_state = matris_test.Matris()
    x_t = pygame.surfarray.array3d(game_state.surface)
    # x_t = cv2.cvtColor(cv2.resize(x_t, (100, 100)), cv2.COLOR_BGR2GRAY)
    # ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    # s_t = x_t.reshape(100,100,1) # 80*80*4
    s_t = prepro(x_t)


    epsilon = INITIAL_EPSILON
    t = 0
    while 1:
        if t<=OBSERVE:
            '''purely Observation -- action choosen state 1'''
            readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
            action = game_state.choose_action()
            a_t = game_state.encode_action(action)
            returns = game_state.update(0,a_t)
            '''purely Observation -- action choosen state 1'''
        elif t<=EXPLORE:
            '''After observation of the awesome AI -- action choosen state 2'''
            returns = False
            while returns == False:
                # choose an action epsilon greedily
                readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
                a_t = np.zeros([ACTIONS])
                action_index = 0
                if random.random() <= epsilon:
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1

                returns = game_state.update(0,a_t)

            if returns[3] != 0:
                print 'Awesome!!! line reduced!!!',returns[3]

            # scale down epsilon
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            '''After observation of the awesome AI -- action choosen state 2'''
        else:
            OBSERVE += 10000
            EXPLORE += 10000
            continue


        # for i in range(0, K):
        # run the selected action and observe next state and reward
        x_t1_col, r_t, terminal, lines = returns
        # x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (100, 100)), cv2.COLOR_BGR2GRAY)
        # ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        # x_t1 = np.reshape(x_t1, (100, 100, 1))
        # s_t1 = x_t1.reshape(100,100,1)
        # print s_t1.shape
        s_t1 = prepro(x_t1_col)
        if terminal:
            game_state = matris_test.Matris()


        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        

        # only train when the size is large enough
        if t > 500:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            if t % 4 == 0:
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + 'dqn_2/', global_step = t)
        # print info
        if t % 1000 == 0:
            print "TIMESTEP", t, "/ LINES", game_state.lines, "/ EPSILON", epsilon, "/ Q_MAX %e" % np.max(readout_t), "/ action",game_state.decode_action(a_t)
        
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
    # screen = pygame.Surface((WIDTH,HEIGHT))
    screen = pygame.display.set_mode((WIDTH, HEIGHT),0,32)
    trainNetwork(s, readout, h_fc1, sess, screen)

def main():
    playGame()

if __name__ == "__main__":
    main()