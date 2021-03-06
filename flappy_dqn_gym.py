import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense
from tensorflow.keras import Sequential
import numpy as np
from numpy import savez_compressed
from numpy import load
from pathlib import Path
import sys
import os
import random
import time
import imageio
import cv2
from PIL import Image
import logging
import pygame
sys.path.append("game/")
import wrapped_flappy_bird as game


class Network:

	def __init__(self, img_width, img_height, name = "network"):

		self.name = name
		self.img_width = img_width
		self.img_height = img_height
		self.q_values = [0, 0]
		self.net = Sequential()
		self.net.add(Input(name = "input_state", shape = [self.img_width, self.img_height, 4], dtype = tf.dtypes.float32))
		self.net.add(Conv2D(filters = 32, kernel_size = (8, 8), strides = (4, 4), padding = 'same'))
		self.net.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))
		self.net.add(Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
		self.net.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
		self.net.add(Flatten())
		self.net.add(Dense(units = 512))
		self.net.add(Dense(units = 2))

class Flappy:

	def __init__(self):

		# Defining the hyper parameters

		self.img_width = 80
		self.img_height = 80
		self.img_depth = 4
		self.eps = 0.1

		self.num_episodes = 10000
		self.pre_train_steps = 10000
		self.save_hist_freq = 500
		self.update_freq = 100  # frequency of updating the target network
		self.batch_size = 32
		self.gamma = 0.99
		self.lr = 0.000001
		self.max_steps = 10000

	def copy_network(self, target_model, q_model):
		target_model.set_weights(q_model.get_weights())

	def pre_process(self, img):

		x_t = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

		return x_t

	def policy(self, algo, img_batch):

		if algo == "epsilon_greedy":
			
			temp = random.random()
			
			if temp < self.eps:
				temp_action = random.randint(0,1)
			else:
				temp_q_values = self.main_net.net(np.reshape(np.stack(img_batch, axis = 2),[-1, 80, 80, 4]))
				temp_action = np.argmax(temp_q_values.numpy())

			return temp_action

	def train(self, mode = 'fixed target'):
		
		assert mode in ['fixed target', 'vanilla']
		log_dir = "training history/train_hist_{}.txt".format(mode)
		logger = logging.getLogger(name='train hist')
		logger.setLevel(logging.INFO)
		handler = logging.FileHandler(log_dir)
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter('%(levelname)s: %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		
		self.main_net = Network(self.img_width, self.img_height, name="main_net")
		if mode == 'fixed target':
			self.target_net = Network(self.img_width, self.img_height, name="target_net")
		if len(os.listdir('./checkpoints/main_net/')) != 0:
			self.main_net.net.load_weights('./checkpoints/main_net/main_checkpoint')
			if mode == 'fixed target':	
				self.target_net.net.load_weights('./checkpoints/target_net/target_checkpoint')

		loss_fn = tf.keras.losses.MeanSquaredError()
		optimizer = tf.keras.optimizers.SGD(learning_rate = self.lr)

		print("Initialized the model")
		if mode == 'fixed target':
			self.copy_network(self.target_net.net, self.main_net.net)
			
		total_steps = 0
		total_reward_list = []
		base_path = Path(__file__).parent
		file_path = str(base_path / "hist.npz")
		dict_data = load('D:\Q Learning - Flappy Bird\Flappy\hist.npz', allow_pickle=True)
		hist_buffer = dict_data['arr_0'].tolist()
		#hist_buffer = []
		best_reward = 0.0

		for i in range(self.num_episodes):
			# Adding initial 4 frames to the image buffer array
			game_state = game.GameState()
			img_batch = []
			total_reward = 0.0

			temp_action = random.randint(0,1)
			action = np.zeros([2])
			action[temp_action] = 1
			new_state, reward, done = game_state.frame_step(action)

			total_steps += 1
			
			temp_img = self.pre_process(new_state)
			img_batch = [temp_img]*4  # 4 * (80, 80)

			while True:

				if total_steps < 100:
					temp_action = random.randint(0,1)
					# print("Temp action is "+ str(temp_action))
				else:
					temp_action = self.policy("epsilon_greedy", img_batch)
					
				action = np.zeros([2])
				action[temp_action] = 1
				new_state, reward, done = game_state.frame_step(action)
					
				temp_img = self.pre_process(new_state)   # temp_img.shape = (80, 80)

				total_reward += reward

				new_img_batch = img_batch[1:]
				new_img_batch.insert(3, temp_img)
				# append (old_state, action, reward, new_state, done) into hist_buffer
				hist_buffer.append((np.stack(img_batch, axis = 2), temp_action, reward, np.stack(new_img_batch,axis = 2), done))
				
				if len(hist_buffer) >= 50000:
					hist_buffer.pop(0)
				
				# Adding the image to the batch

				img_batch.insert(len(img_batch), temp_img)
				img_batch.pop(0)

				# Breaking the loop if the state is terminated

				if total_steps > self.pre_train_steps:

					rand_batch = random.sample(hist_buffer, self.batch_size)
					reward_hist = [m[2] for m in rand_batch]
					state_hist = [m[0] for m in rand_batch]
					action_hist = [m[1] for m in rand_batch]
					next_state_hist = [m[3] for m in rand_batch]
					
					if mode == 'fixed target':
						temp_target_q = self.target_net.net(np.stack(next_state_hist))
						temp_target_q = np.amax(temp_target_q, 1)
						temp_target_reward = reward_hist + self.gamma*temp_target_q
						temp_target_reward = np.reshape(temp_target_reward, [self.batch_size, 1])
					
					elif mode == 'vanilla':
						temp_target_q = self.main_net.net(np.stack(next_state_hist))
						temp_target_q = np.amax(temp_target_q, axis=1)
						temp_target_reward = reward_hist + self.gamma*temp_target_q
						temp_target_reward = np.reshape(temp_target_reward, [self.batch_size, 1])

					with tf.GradientTape() as tape:
						q_values = self.main_net.net(np.stack(state_hist))
						observed_reward = tf.reduce_sum(q_values*tf.one_hot(tf.reshape(np.reshape(np.stack(action_hist),[self.batch_size, 1]),[-1]),2,dtype=tf.float32),1,keepdims=True)
						loss = loss_fn(observed_reward, temp_target_reward)
					grads = tape.gradient(loss, self.main_net.net.trainable_weights)
					optimizer.apply_gradients(zip(grads, self.main_net.net.trainable_weights))

					if mode == 'fixed target':	
						if total_steps % self.update_freq == 0:
							self.copy_network(self.target_net.net, self.main_net.net)

				if done:
					#print(game_state.getScore())
					record = "Episode {}:".format(i+1) + str(game_state.getScore())
					logger.info(record)
					break

				total_steps += 1
			
			if i % self.save_hist_freq == 0:
					savez_compressed('hist.npz', np.asarray(hist_buffer))

			if total_reward > best_reward:
				best_reward = total_reward
				self.main_net.net.save_weights('./checkpoints/main_net/main_checkpoint')
				if mode == 'fixed target':	
					self.target_net.net.save_weights('./checkpoints/target_net/target_checkpoint')
			print("Total rewards in episode " + str(i) + " is " + str(total_reward) + " total number of steps are " + str(total_steps))
		

	def play(self, mode = "random", fps = 30):
		clock = pygame.time.Clock()
		assert mode in ['random', 'ai', 'manual']
		self.main_net = Network(self.img_width, self.img_height, name="main_net")
		if len(os.listdir('./checkpoints/main_net/')) != 0:
			self.main_net.net.load_weights('./checkpoints/main_net/main_checkpoint')	
		#writer = imageio.get_writer('gif/demo.gif', mode='I')
		game_state = game.GameState()
		total_steps = 0
		img_batch = []

		action = np.zeros([2])
		action[0] = 1
		new_state, reward, done =  game_state.frame_step(action)

		temp_img = self.pre_process(new_state)

		for j in range(4):
			img_batch.insert(len(img_batch), temp_img)
		
		for j in range(self.max_steps):

			if mode == "random":
				temp_action = random.randint(0,1)
			elif mode == 'ai':
				#print(img_batch[3])
				temp_weights = self.main_net.net(np.reshape(np.stack(img_batch,axis=2),[-1, 80, 80, 4]))
				temp_action = np.argmax(temp_weights)
				#print(temp_weights)
			elif mode == 'manual':
				for event in pygame.event.get():
					if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
						temp_action = 1
					else:
						temp_action = 0
			action = np.zeros([2])
			action[temp_action] = 1

			new_state, reward, done =  game_state.frame_step(action)

			temp_new_state = np.flip(np.rot90(new_state, k=1, axes=(1,0)), 1)

			temp_img = self.pre_process(new_state)
			img_batch.insert(0, temp_img)
			img_batch.pop(len(img_batch)-1)

			total_steps += 1
			clock.tick(fps)
			if done:
				print("Score:",game_state.getScore())
				break

		print("Total steps:", str(total_steps))

		sys.exit()

def main():

	mod = Flappy()
	mod.train(mode='fixed target')
	#mod.play("manual")

if __name__ == "__main__":
	main()
