#!/usr/bin/env python

import tensorflow as tf
import random
import numpy as np
import pygame
from matris_test import *

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
    
''' initalize state variable'''
def predict():
	s, readout, h_fc1 = createNetwork()

	saver = tf.train.Saver()
	sess.run(tf.initialize_all_variables())
	checkpoint = tf.train.get_checkpoint_state("saved_networks/dqn_3")
	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print "Successfully loaded:", checkpoint.model_checkpoint_path
	else:
		print "Could not find old network weights"

	background = Surface(screen.get_size())
	game_state = Matris()
	x_t = pygame.surfarray.array3d(game_state.surface)
	s_t = prepro(x_t)

	while 1:
		
		readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
		a_t = game_state.choose_action_dqn(readout_t,0)
		returns = game_state.update(0,a_t)
		x_t1_col, r_t, terminal, lines = returns
		s_t1 = prepro(x_t1_col)
		s_t = s_t1

		background.blit(game_state.surface, (MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH))
		screen.blit(background, (0, 0))
		pygame.display.flip()
		if terminal:
			 game_state = Matris()


"""
predict on a test set
"""
pygame.init()
WIDTH = 700
HEIGHT = 660
ACTIONS = 48 # number of valid actions
sess = tf.InteractiveSession()
screen = pygame.display.set_mode((WIDTH, HEIGHT),0,32)
# open up a game state to communicate with emulator
BORDERWIDTH = 10
MATRIS_OFFSET = 20
predict()