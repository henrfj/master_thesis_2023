import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

# My own libraries
#import a_star_utils as autils

from Vehicle import Vehicle

from Agent import Agent # DC agents
from .ddpg_torch import MPC_Agent
from limo import Limo
from visualization import Visualization
from Object import Object

class OpenField_v00(gym.Env):
	"""Custom Environment that follows gym interface.
	- In this env. the vehicle will be spawned in an open field.
	- state space is x, y, and heading, and a fixed goal pose. 
	- The goal of the vehicle is to get as close as possible to the goal pose, 
	while limiting "fuel consumption".
	- Goal state is always (100, 100, 0) ~ "goal coordinate axis"
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False):
		super(OpenField_v00, self).__init__()
		"""
		Example when using discrete actions:
		-> self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
		Example for using image as input:
		-> self.observation_space = spaces.Box(low=0, high=255, shape=
					(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
		"""
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([0, -1]), # NOTE has turned off reversing for now!
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the current x, y as well as heading, and the goal state
		self.observation_space = spaces.Box(low=np.array([0, 0, -2*np.pi]),
											high=np.array(
												[200, 200, 2*np.pi, ]),
											dtype=np.float32)
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1200, 1200)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7) # Also initializes the display
			
		self.state = []

	@staticmethod
	def generate_new_start_state():
		""" Based on current position, generate a new goal pose for the agent to reach.
				- Start state is always about (100, 100), so it will be easier to visualize (0-200)
		"""
		x_pos, y_pos = np.array([100, 100]) - np.random.randint(-50, 50, 2)
		heading = np.random.uniform(-2*np.pi, 2*np.pi)
		return x_pos, y_pos, heading

	def step(self, action):
		"""Execute one time step within the environment"""

		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action
		# NOTE Do not need to take new decision every simulation step (0.1 sec)
		times = np.int32(self.decision_dt/self.sim_dt)
		self.render_frames = []
		for _ in range(times):
			self.vehicle.one_step_algorithm(alpha_ref, v_ref)
			# For rendering in sim time
			xpos, ypos = self.vehicle.position_center
			heading = self.vehicle.heading
			# 
			if self.will_render:
				self.render_frames.append([xpos, ypos, heading])

		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		new_state = np.array([xpos, ypos, heading], dtype=np.float32)
		self.state = new_state


		######################################
		# Reward function! (Sparse for now?) #
		######################################
		reward = 0
		done = False
		info = "..."
		# Left the viewpoint (0-200)
		if xpos > 200 or xpos < 0 or ypos > 200 or ypos < 0:
			reward += -100  # punished
			done = True
			info = "'left the viewpoint...'"

		current = np.array([xpos, ypos])
		goal = np.array([100, 100])
		dist = np.linalg.norm(current - goal)
		

		# Goal is reached!
		if dist < self.goal_threshold:  # some threshold
			reward += 400  # Goal reached!
			done = True
			info = "'Goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.01
			#reward += -1

		# Time is up?
		if self.time_step > 200:  # (100 sek)
			done = True
			reward += -100  # Goal not reached :(
			info = "'Time is up!'"
		self.time_step += 1
		return new_state, reward, done, info

	def reset(self):
		# Reset the state of the environment to an initial state'
		x_pos, y_pos, heading = self.generate_new_start_state()
		# Volkswagen
		self.vehicle = Vehicle(np.array([x_pos, y_pos]), length=4, width=2,
								heading=heading, tau_steering=0.4, tau_throttle=0.4, dt=self.sim_dt)
		# TODO; add these to the vehicle!
		self.v_max = 7
		self.v_min = -2
		self.alpha_max = 0.8
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		return x_pos, y_pos, heading

	def render(self, mode='human', close=False, render_all_frames=False):
		if render_all_frames:
			for _, frame in enumerate(self.render_frames):
				self.render_one_frame(frame)
		else:
			self.render_one_frame(self.state)

	def render_one_frame(self, state):
		##################
		self.gfx.clear_canvas()
    	##################
		# Extract geometry
		length=self.vehicle.length
		width=self.vehicle.width
		center_pos = state[0:2]
		heading = state[2]

		# Vertices:
		verticesCCF = [np.array([width/2,  length/2 ]),
						np.array([-width/2, length/2 ]),
						np.array([-width/2, -length/2]),
						np.array([width/2,  -length/2])]

		angle = heading-np.pi/2
		R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])

		verticesWCF = []
		for vertex in verticesCCF:
			verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
		# Sides
		sides = [[verticesWCF[-1], verticesWCF[0]]]
		for i in range(len(verticesWCF)-1):
			sides.append([verticesWCF[i], verticesWCF[i+1]])

		self.gfx.draw_sides(sides)
		# Draw heading and center of vehicle
		self.gfx.draw_center_and_headings_simple(heading, center_pos)

		# Draw the goal and goal limit
		self.gfx.draw_goal_state(np.array([100, 100]), threshold=self.goal_threshold)

		##################
		self.gfx.update_display()
		##################
class OpenField_v01(gym.Env):
	"""Custom Environment that follows gym interface.
	- In this env. the vehicle will be spawned in an open field.
	- Unlike v0, state space is x, y, heading, normed_vel and steering angle!.
	- Also, the action space is extended to also hold reversing actions. 
	- The goal of the vehicle is to get as close as possible to the goal pose, 
	while limiting "fuel consumption".
	- Goal state is always (100, 100, 0) ~ "goal coordinate axis"
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, vmax=8, alpha_max=0.8):
		super(OpenField_v01, self).__init__()
		"""
		Example when using discrete actions:
		-> self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
		Example for using image as input:
		-> self.observation_space = spaces.Box(low=0, high=255, shape=
					(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
		"""
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the current x, y as well as heading, and the goal state
		self.observation_space = spaces.Box(low=np.array([0, 0, -2*np.pi, -vmax, -alpha_max]),
											high=np.array(
												[200, 200, 2*np.pi, vmax, alpha_max]),
											dtype=np.float32)
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1200, 1200)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7) # Also initializes the display
		self.state = []

	@staticmethod
	def generate_new_start_state():
		""" Based on current position, generate a new goal pose for the agent to reach.
				- Start state is always about (100, 100), so it will be easier to visualize (0-200)
		"""
		x_pos, y_pos = np.array([100, 100]) - np.random.randint(-50, 50, 2)
		heading = np.random.uniform(-2*np.pi, 2*np.pi)
		return x_pos, y_pos, heading

	def step(self, action):
		"""Execute one time step within the environment"""

		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action
		# NOTE Do not need to take new decision every simulation step (0.1 sec)
		times = np.int32(self.decision_dt/self.sim_dt)
		self.render_frames = []
		for _ in range(times):
			self.vehicle.one_step_algorithm(alpha_ref, v_ref)
			# For rendering in sim time
			xpos, ypos = self.vehicle.position_center
			heading = self.vehicle.heading
			# 
			if self.will_render:
				self.render_frames.append([xpos, ypos, heading])

		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		new_state = np.array([xpos, ypos, heading, normed_velocity, steering_angle], dtype=np.float32)
		self.state = new_state


		######################################
		# Reward function! (Sparse for now?) #
		######################################
		reward = 0
		done = False
		info = "..."
		# Left the viewpoint (0-200)
		if xpos > 200 or xpos < 0 or ypos > 200 or ypos < 0:
			reward += -100  # punished
			done = True
			info = "'left the viewpoint...'"

		current = np.array([xpos, ypos])
		goal = np.array([100, 100])
		dist = np.linalg.norm(current - goal)
		

		# Goal is reached!
		if dist < self.goal_threshold:  # some threshold
			reward += 400  # Goal reached!
			done = True
			info = "'Goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.1
			#reward += -1

		# Time is up?
		if self.time_step > 200:  # (100 sek)
			done = True
			reward += -100  # Goal not reached :(
			info = "'Time is up!'"
		self.time_step += 1
		return new_state, reward, done, info

	def reset(self):
		# Reset the state of the environment to an initial state'
		x_pos, y_pos, heading = self.generate_new_start_state()
		# Volkswagen
		self.vehicle = Vehicle(np.array([x_pos, y_pos]), length=4, width=2,
								heading=heading, tau_steering=0.4, tau_throttle=0.4, dt=self.sim_dt)
		# TODO; add these to the vehicle!
		self.v_max = 7
		self.v_min = -2
		self.alpha_max = 0.8
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_vel = 0
		steering_angle = 0
		return x_pos, y_pos, heading, 0, steering_angle

	def render(self, mode='human', close=False, render_all_frames=False):
		if render_all_frames:
			for _, frame in enumerate(self.render_frames):
				self.render_one_frame(frame)
		else:
			self.render_one_frame(self.state)

	def render_one_frame(self, state):
		##################
		self.gfx.clear_canvas()
    	##################
		# Extract geometry
		length=self.vehicle.length
		width=self.vehicle.width
		center_pos = state[0:2]
		heading = state[2]

		# Vertices:
		verticesCCF = [np.array([width/2,  length/2 ]),
						np.array([-width/2, length/2 ]),
						np.array([-width/2, -length/2]),
						np.array([width/2,  -length/2])]

		angle = heading-np.pi/2
		R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])

		verticesWCF = []
		for vertex in verticesCCF:
			verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
		# Sides
		sides = [[verticesWCF[-1], verticesWCF[0]]]
		for i in range(len(verticesWCF)-1):
			sides.append([verticesWCF[i], verticesWCF[i+1]])

		self.gfx.draw_sides(sides)
		# Draw heading and center of vehicle
		self.gfx.draw_center_and_headings_simple(heading, center_pos)

		# Draw the goal and goal limit
		self.gfx.draw_goal_state(np.array([100, 100]), threshold=self.goal_threshold)

		##################
		self.gfx.update_display()
		##################
class OpenField_v10(gym.Env): # THE MILKMAN
	"""Custom Environment that follows gym interface.
	- In this env, the goal pose is *GIVEN* in the state space vector; but it is given in driver-coordinated (CCF)
	- This will result in a simpler SSV, of only [Goal pose, speed, steering angle].
	- To make the agent able to chase down sequential goals, it would make more sense to generate random goal poses,
	rather than random initial vehicle poses.
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, vmax=8, alpha_max=0.8):
		super(OpenField_v10, self).__init__()

		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the goal_x, goal_y (IN CCF), speed and steering angle
		self.observation_space = spaces.Box(low=np.array([0, 0, -vmax, -alpha_max]),
											high=np.array(
												[200, 200, vmax, alpha_max]),
											dtype=np.float32)
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1200, 1200)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7) # Also initializes the display

	@staticmethod
	def generate_new_goal_state():
		""" 
		Generates a random goal state about the starting position, and translatest it into CCF.
		"""
		goal_x, goal_y = np.array([100, 100]) - np.random.randint(-50, 50, 2)
		return goal_x, goal_y
	
	@staticmethod
	def generate_new_goal_state_2(min, max):
		""" 
		Generates a random goal state about the starting position, and translatest it into CCF.
		"""
		goal_x, goal_y = np.random.randint(min+10, max-10, 2)
		return goal_x, goal_y
	
	def reset(self, vmax=7, v_min=-2, alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4):
		# Reset the state of the environment to an initial state'
		goal_x, goal_y = self.generate_new_goal_state()
		self.goal_x, self.goal_y = goal_x, goal_y # This allows for setting new goal states!

		# Volkswagen
		self.vehicle = Vehicle(np.array([100, 100]), length=4, width=2,
								heading=-np.pi, tau_steering=tau_steering, tau_throttle=tau_throttle, dt=self.sim_dt)
		
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		# TODO; add these to the vehicle!
		self.v_max = vmax
		self.v_min = v_min
		self.alpha_max = alpha_max

		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_vel = 0
		steering_angle = 0

		
		return self.goal_CCF[0], self.goal_CCF[1], normed_vel, steering_angle

	def step(self, action):
		"""Execute one time step within the environment"""

		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action
		# NOTE Do not need to take new decision every simulation step (0.1 sec)
		times = np.int32(self.decision_dt/self.sim_dt)
		if self.will_render:
			self.render_frames = []
		for _ in range(times):
			self.vehicle.one_step_algorithm(alpha_ref, v_ref)
			# For rendering in sim time
			xpos, ypos = self.vehicle.position_center
			heading = self.vehicle.heading
			# 
			if self.will_render:
				self.render_frames.append([xpos, ypos, heading])
		
		# After running n simulations steps:
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		# Update goal poses in CCF (as CCF's origin has moved)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle], dtype=np.float32)

		######################################
		# Reward function! (Sparse for now?) #
		######################################
		reward = 0
		done = False
		info = "..."

		# Calculate goal distance in WCF
		current_pos = np.array([xpos, ypos])
		goal_pos = np.array([self.goal_x, self.goal_y])
		dist = np.linalg.norm(current_pos - goal_pos)
		# NOTE could also have used CCF version...
		if np.round(dist,1) != np.round(np.linalg.norm(self.goal_CCF), 1):
			print(dist)
			print(np.linalg.norm(self.goal_CCF))
			raise ValueError("Should be same value!")
	
		# Goal is reached!
		if dist < self.goal_threshold:  # some threshold
			reward += 400  # Goal reached!
			done = True
			info = "'Goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.1
			#reward += -1

		# Time is up?
		if self.time_step > 30*1/self.decision_dt:  # (30 sek)
			done = True
			reward += -200  # Goal not reached :(
			info = "'Time is up!'"
		self.time_step += 1
		return new_state, reward, done, info

	def render(self, mode='human', close=False, render_all_frames=False):
		if render_all_frames:
			for _, frame in enumerate(self.render_frames):
				self.render_one_frame(frame)
		else:
			self.render_one_frame(np.array([self.vehicle.X[0], self.vehicle.X[1], self.vehicle.heading]))

	def render_one_frame(self, state):
		##################
		self.gfx.clear_canvas()
    	##################
		# Extract geometry
		length=self.vehicle.length
		width=self.vehicle.width
		center_pos = state[0:2]
		heading = state[2]

		# Vertices:
		verticesCCF = [np.array([width/2,  length/2 ]),
						np.array([-width/2, length/2 ]),
						np.array([-width/2, -length/2]),
						np.array([width/2,  -length/2])]

		angle = heading-np.pi/2
		R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])

		verticesWCF = []
		for vertex in verticesCCF:
			verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
		# Sides
		sides = [[verticesWCF[-1], verticesWCF[0]]]
		for i in range(len(verticesWCF)-1):
			sides.append([verticesWCF[i], verticesWCF[i+1]])

		self.gfx.draw_sides(sides)
		# Draw heading and center of vehicle
		self.gfx.draw_center_and_headings_simple(heading, center_pos)

		# Draw the goal and goal limit
		self.gfx.draw_goal_state(np.array([self.goal_x, self.goal_y]), threshold=self.goal_threshold)

		##################
		self.gfx.update_display()
		##################
class ClosedField_v20(gym.Env): # THE MILKMAN
	"""Custom Environment that follows gym interface.
	- This environment continous the V10 CCF trend, but adds SC!
	- Adds a single SC for each decision step, to the state space (only the distances)
	- Adds walls, and some other obstacles. Collision added as an "episode ender" with large negative consequence.
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, v_max=8, v_min=-2, alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150):
		super(ClosedField_v20, self).__init__()
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		self.v_max = v_max
		self.v_min = v_min
		self.alpha_max = alpha_max
		self.tau_steering = tau_steering
		self.tau_throttle = tau_throttle
		self.horizon = horizon
		self.edge = edge
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the goal_x, goal_y (IN CCF), speed and steering angle
		self.observation_space = spaces.Box(low=np.array([0, 0, -self.v_min, -alpha_max,
						    								0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0]),
											high=np.array(
											[200, 200, 	self.v_max, alpha_max,
	    												self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon]),
											dtype=np.float32)
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1080, 1920)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7) # Also initializes the display
		self.objects = []
	@staticmethod
	def generate_new_goal_state(min, max):
		""" 
		Generates a random goal state about the starting position, and translatest it into CCF.
		"""
		goal_x, goal_y = np.random.randint(min+10, max-10, 2)
		return goal_x, goal_y
	
	def add_objects(self, vertices):
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
	
	def reset(self):
		# Volkswagen
		self.vehicle = Vehicle(np.array([100, 100]), length=4, width=2,
								heading=-np.pi, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		
		# Reset the state of the environment to an initial state'
		goal_x, goal_y = self.generate_new_goal_state(10, 150)
		self.goal_x, self.goal_y = goal_x, goal_y # This allows for setting new goal states!
		
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		# Generate an outer wall + some obstacles

		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, self.edge], [self.edge, self.edge], [self.edge, 5]])
		outer_rim = Object(np.array([0, 0]), vertices=vertices)
		
		vertices = np.array([[30, 30], [30, 35], [35, 35], [35, 30]])
		box_1 = Object(np.array([0, 0]), vertices=vertices)

		vertices = np.array([[75, 75], [75, 80], [80, 80], [80, 75]])
		box_2 = Object(np.array([0, 0]), vertices=vertices)

		vertices = np.array([[75, 115], [75, 120], [80, 120], [80, 115]])
		box_3 = Object(np.array([0, 0]), vertices=vertices)

		self.objects = [outer_rim, box_1, box_2, box_3]
		#######################################
		#######################################
		#######################################
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_velocity = 0
		steering_angle = 0
		#
		# Get the new static circogram
		SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _ = SC
		real_distances = d2 - d1


		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle,
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)

		return new_state

	def step(self, action):
		"""Execute one time step within the environment"""

		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action
		# NOTE Do not need to take new decision every simulation step (0.1 sec)
		times = np.int32(self.decision_dt/self.sim_dt)
		if self.will_render:
			self.render_frames = []
		for _ in range(times):
			self.vehicle.one_step_algorithm(alpha_ref, v_ref)
			# For rendering in sim time
			xpos, ypos = self.vehicle.position_center
			heading = self.vehicle.heading
			# 
			if self.will_render:
				self.render_frames.append([xpos, ypos, heading])
		
		# After running n simulations steps:
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		# Update goal poses in CCF (as CCF's origin has moved)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		# Get the new static circogram
		SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _ = SC
		self.vehicle.collision_check(d1, d2)
		real_distances = d2 - d1

		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle,
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)

		######################################
		# Reward function! (Sparse for now?) #
		######################################
		reward = 0
		done = False
		info = "..."

		# Calculate goal distance
		dist = np.linalg.norm(self.goal_CCF)
		# Goal is reached!
		if dist < self.goal_threshold:  # some threshold
			reward += 1000  # Goal reached!
			done = True
			info = "'Goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.1

		if self.vehicle.collided:
			done = True
			reward = -400
			info = "'Collided"

		# Time is up?
		elif self.time_step > 30*1/self.decision_dt:  # (30 sek)
			done = True
			reward += -200  # Goal not reached :(
			info = "'Time is up!'"
		self.time_step += 1

		return new_state, reward, done, info

	def render(self, mode='human', close=False, render_all_frames=False):
		if render_all_frames:
			for _, frame in enumerate(self.render_frames):
				self.render_one_frame(frame)
		else:
			self.render_one_frame(np.array([self.vehicle.X[0], self.vehicle.X[1], self.vehicle.heading]))

	def render_one_frame(self, state):
		##################
		self.gfx.clear_canvas()
    	##################
		# Extract geometry
		length=self.vehicle.length
		width=self.vehicle.width
		center_pos = state[0:2]
		heading = state[2]

		# Vertices:
		verticesCCF = [np.array([width/2,  length/2 ]),
						np.array([-width/2, length/2 ]),
						np.array([-width/2, -length/2]),
						np.array([width/2,  -length/2])]

		angle = heading-np.pi/2
		R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])

		verticesWCF = []
		for vertex in verticesCCF:
			verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
		# Sides
		sides = [[verticesWCF[-1], verticesWCF[0]]]
		for i in range(len(verticesWCF)-1):
			sides.append([verticesWCF[i], verticesWCF[i+1]])

		self.gfx.draw_sides(sides)
		# Draw heading and center of vehicle
		self.gfx.draw_center_and_headings_simple(heading, center_pos)

		# Draw the goal and goal limit
		self.gfx.draw_goal_state(np.array([self.goal_x, self.goal_y]), threshold=self.goal_threshold)

		# Draw all static obstacles
		for obj in self.objects:
			self.gfx.draw_sides(obj.sides)
		##################
		self.gfx.update_display()
		##################
class ClosedField_v21(gym.Env): # THE MILKMAN
	"""Custom Environment that follows gym interface.
	- This environment continous the V10 CCF trend, but adds SC!
	- Adds a single SC for each decision step, to the state space (only the distances)
	- Adds walls, and some other obstacles. Collision added as an "episode ender" with large negative consequence.
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, v_max=8, v_min=-2,
	       alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150,
		   episode_s=60):
		super(ClosedField_v21, self).__init__()
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		self.v_max = v_max
		self.v_min = v_min
		self.alpha_max = alpha_max
		self.tau_steering = tau_steering
		self.tau_throttle = tau_throttle
		self.horizon = horizon
		self.edge = edge
		self.selection = -1
		self.episode_seconds = episode_s
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the goal_x, goal_y (IN CCF), speed and steering angle
		self.observation_space = spaces.Box(low=np.array([0, 0, -self.v_min, -alpha_max,
						    								0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0]),
											high=np.array(
											[200, 200, 	self.v_max, alpha_max,
	    												self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon]),
											dtype=np.float32)
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1080, 1920)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7) # Also initializes the display
		self.objects = []

	def add_objects(self, vertices):
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))

	def generate_new_goal_state(self):
		prev_selection = self.selection
		selection = -1
		while selection == prev_selection:
			selection = np.random.randint(0, 4) # [low, high)
		if selection == 0: # WEST
			goal = np.array([25, 75])
		elif selection == 1: # NORTH
			# ADD LID - NOTE was too hard, for now... :(
			#vertices = np.array([[120, 65], [144, 65]])
			#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
			goal = np.array([132, 25])
			#self.vehicle.heading=-np.pi/2
		elif selection == 2: # EAST
			goal = np.array([245, 75])
		elif selection == 3: # SOUTH
			goal = np.array([132, 135])
			# ADD LID
			#vertices = np.array([[120, 85], [144, 85]])
			#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
			#self.vehicle.heading=np.pi/2
		self.goal_x, self.goal_y = goal
		
	def reset(self):
		# Volkswagen
		self.vehicle = Vehicle(np.array([132, 75]), length=4, width=2,
								heading=-np.pi/2, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		
		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
		self.outer_rim = Object(np.array([0, 0]), vertices=vertices)
		self.objects = [self.outer_rim]
		# Add the base walls
		vertices = np.array([[35, 90], [35, 60], [40, 60], [40, 90]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[115, 35], [115, 40], [145, 40], [145, 35]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[230, 60], [230, 90], [235, 90], [235, 60]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[115, 125], [115, 130], [145, 130], [145, 125]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		# Add side tunnels
		#vertices = np.array([[120, 65], [120, 85], [115, 85], [115, 65]])
		#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		#vertices = np.array([[144, 65], [144, 85], [149, 85], [149, 65]])
		#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		#Goals and special obstacles added
		self.generate_new_goal_state()
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		#######################################
		#######################################
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_vel = 0
		steering_angle = 0
		#
		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		real_distances = d2 - d1

		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_vel, steering_angle,
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)

		return new_state

	def step(self, action):
		"""Execute one time step within the environment"""

		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action
		# NOTE Do not need to take new decision every simulation step (0.1 sec)
		times = np.int32(self.decision_dt/self.sim_dt)
		if self.will_render:
			self.render_frames = []
		for _ in range(times):
			self.vehicle.one_step_algorithm(alpha_ref, v_ref)
			# For rendering in sim time
			xpos, ypos = self.vehicle.position_center
			heading = self.vehicle.heading
			# 
			if self.will_render:
				self.render_frames.append([xpos, ypos, heading])
		
		# After running n simulations steps:
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		# Update goal poses in CCF (as CCF's origin has moved)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		self.vehicle.collision_check(d1, d2)
		real_distances = d2 - d1

		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle,
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)

		######################################
		# Reward function! (Sparse for now?) #
		######################################
		reward = 0
		done = False
		info = "..."

		# Calculate goal distance
		dist = np.linalg.norm(self.goal_CCF)
		# Goal is reached!
		if dist < self.goal_threshold:  # some threshold
			reward += 40  # Goal reached!
			done = True
			info = "'Goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.01

		if self.vehicle.collided:
			done = True
			reward = -10
			info = "'Collided'"

		# Time is up?
		elif self.time_step > self.episode_seconds*1/self.decision_dt:  # (30 sek)
			done = True
			reward += -10  # Goal not reached :(
			info = "'Time is up!'"
		self.time_step += 1

		return new_state, reward, done, info

	def render(self, mode='human', close=False, render_all_frames=False, show_SC=False):
		if render_all_frames:
			for _, frame in enumerate(self.render_frames):
				self.render_one_frame(frame, show_SC=show_SC)
		else:
			self.render_one_frame(np.array([self.vehicle.X[0], self.vehicle.X[1], self.vehicle.heading]), show_SC=show_SC)

	def render_one_frame(self, state, show_SC=False):
		##################
		self.gfx.clear_canvas()
    	##################
		# Extract geometry
		length=self.vehicle.length
		width=self.vehicle.width
		center_pos = state[0:2]
		heading = state[2]

		# Vertices:
		verticesCCF = [np.array([width/2,  length/2 ]),
						np.array([-width/2, length/2 ]),
						np.array([-width/2, -length/2]),
						np.array([width/2,  -length/2])]

		angle = heading-np.pi/2
		R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])

		verticesWCF = []
		for vertex in verticesCCF:
			verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
		# Sides
		sides = [[verticesWCF[-1], verticesWCF[0]]]
		for i in range(len(verticesWCF)-1):
			sides.append([verticesWCF[i], verticesWCF[i+1]])

		self.gfx.draw_sides(sides)
		# Draw heading and center of vehicle
		self.gfx.draw_center_and_headings_simple(heading, center_pos)

		# Draw the goal and goal limit
		self.gfx.draw_goal_state(np.array([self.goal_x, self.goal_y]), threshold=self.goal_threshold)

		# Draw all static obstacles
		for obj in self.objects:
			self.gfx.draw_sides(obj.sides)

		if show_SC:
			self.gfx.draw_static_circogram_data(self, self.SC, self.vehicle)
		##################
		self.gfx.update_display()
		##################
class ClosedField_v22(gym.Env): # THE MILKMAN
	"""Custom Environment that follows gym interface.
	- Adds knowledge of previously chosen actions, and rewards smooth driving.
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, v_max=8, v_min=-2,
	       alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150,
		   episode_s=60, environment_selection="four_walls"):
		super(ClosedField_v22, self).__init__()
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		self.v_max = v_max
		self.v_min = v_min
		self.alpha_max = alpha_max
		self.tau_steering = tau_steering
		self.tau_throttle = tau_throttle
		self.horizon = horizon
		self.edge = edge
		self.selection = -1
		self.episode_seconds = episode_s
		self.environment_selection = environment_selection
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the goal_x, goal_y (IN CCF), speed and steering angle
		self.observation_space = spaces.Box(low=np.array([0, 0, -self.v_min, -alpha_max, -1, -1,
						    								0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0]),
											high=np.array(
											[200, 200, 	self.v_max, alpha_max, 1, 1,
	    												self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon]),
											dtype=np.float32)
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1080, 1920)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7) # Also initializes the display
		self.objects = []

	def add_objects(self, vertices):
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))

	def generate_new_goal_state(self):
		prev_selection = self.selection
		selection = -1
		while selection == prev_selection:
			selection = np.random.randint(0, 4) # [low, high)
		if selection == 0: # WEST
			goal = np.array([25, 75])
		elif selection == 1: # NORTH
			# ADD LID - NOTE was too hard, for now... :(
			#vertices = np.array([[120, 65], [144, 65]])
			#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
			goal = np.array([132, 25])
			#self.vehicle.heading=-np.pi/2
		elif selection == 2: # EAST
			goal = np.array([245, 75])
		elif selection == 3: # SOUTH
			goal = np.array([132, 135])
			# ADD LID
			#vertices = np.array([[120, 85], [144, 85]])
			#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
			#self.vehicle.heading=np.pi/2
		self.goal_x, self.goal_y = goal
		return goal

	def four_walls_envionment(self):
		""" Environment with four walls """
		# Volkswagen
		self.vehicle = Vehicle(np.array([132, 75]), length=4, width=2,
								heading=-np.pi/2, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
		self.outer_rim = Object(np.array([0, 0]), vertices=vertices)
		self.objects = [self.outer_rim]
		# Add the base walls
		vertices = np.array([[35, 90], [35, 60], [40, 60], [40, 90]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[115, 35], [115, 40], [145, 40], [145, 35]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[230, 60], [230, 90], [235, 90], [235, 60]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[115, 125], [115, 130], [145, 130], [145, 125]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		# Goal state
		goal = self.generate_new_goal_state()
		self.goal_stack = deque([goal])
		self.goal_x, self.goal_y = self.goal_stack.popleft()
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		#######################################
	
	def Naples_street(self):
		""" Environment with Slim streets """
		# Volkswagen
		self.vehicle = Vehicle(np.array([132, 140]), length=4, width=2,
								heading=-np.pi/2, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
		self.outer_rim = Object(np.array([0, 0]), vertices=vertices)
		self.objects = [self.outer_rim]
		# Add the base walls
		vertices = np.array([[125, 25], [125, 150], [120, 150], [120, 25]]) # left wall
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[120, 50], [120, 55], [5, 55], [5, 50]]) # Plaza_1 wall
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[140, 5], [140, 68], [145, 68], [145, 5]]) # right wall 1
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[140, 80], [140, 150], [145, 150], [145, 80]]) # right wall 2
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[170, 60], [175, 60], [175, 80], [170, 80]]) # Blocker plate
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))


		# Goal state
		self.goal_stack = deque([np.array([132, 15]), np.array([15, 15]), np.array([132, 15]), np.array([132, 70]), np.array([200, 70])])
		self.goal_x, self.goal_y = self.goal_stack.popleft()
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		#######################################

	def reset(self):
		""" Resets the environment"""
		##### Chose the objects to spawn in #####
		if self.environment_selection=="four_walls":
			self.four_walls_envionment()
		elif self.environment_selection=="naples_street":
			self.Naples_street()
		else:
			self.four_walls_envionment()
		#######################################
		#######################################
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_vel = 0
		steering_angle = 0
		#
		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		real_distances = d2 - d1
		#
		self.previous_throttle = 0
		self.previous_steering = 0
		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_vel, steering_angle, self.previous_throttle, self.previous_steering,
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)

		return new_state

	def step(self, action, add_disturbance=None):
		"""Execute one time step within the environment"""

		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action
		# NOTE Do not need to take new decision every simulation step (0.1 sec)
		times = np.int32(self.decision_dt/self.sim_dt)
		if self.will_render:
			self.render_frames = []

		goal_was_reached = False
		for _ in range(times):
			if add_disturbance: 
				# 1 Add disturbances
				tau_v = self.vehicle.tau_throttle + add_disturbance[0]
				tau_alpha = self.vehicle.tau_steering + add_disturbance[1]
				k_max = self.v_max + add_disturbance[2]
				k_min = self.v_min + add_disturbance[3]
				c_max = self.alpha_max + add_disturbance[4]
				d = self.vehicle.d + add_disturbance[5]
				self.vehicle.one_step_algorithm_3(action=action, dt=self.sim_dt, params=[tau_v, tau_alpha, k_max, k_min, c_max, d])
			else:
				
				self.vehicle.one_step_algorithm_2(alpha_ref, v_ref, dt=self.sim_dt)
			# For rendering in sim time
			xpos, ypos = self.vehicle.position_center
			heading = self.vehicle.heading
			# Check for goal in here!
			self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
			dist = np.linalg.norm(self.goal_CCF)
			if dist < self.goal_threshold:
				goal_was_reached = True # Goal was reached during simulation
			# 
			if self.will_render:
				self.render_frames.append([xpos, ypos, heading])
		
		# After running n simulations steps:
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		# Update goal poses in CCF (as CCF's origin has moved)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		self.vehicle.collision_check(d1, d2)
		real_distances = d2 - d1

		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle, self.previous_throttle, self.previous_steering, 
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)


		######################################
		# Reward function! (Sparse for now?) #
		######################################
		reward = 0
		done = False
		info = "..."

		# Calculate goal distance
		dist = np.linalg.norm(self.goal_CCF)
		# Goal is reached!
		if dist < self.goal_threshold or goal_was_reached:  # some threshold
			reward += 1000 # Goal reached!
			if len(self.goal_stack) == 0:
				done = True
				info = "'Final goal reached!'"
			else:
				self.goal_x, self.goal_y = self.goal_stack.popleft()
				#print('Sub-goal reached!')
				info = "'Sub-goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.01

		if self.vehicle.collided:
			done = True
			reward = -500
			info = "'Collided'"

		# Time is up?
		elif self.time_step > self.episode_seconds*1/self.decision_dt:  # (30 sek)
			done = True
			reward += -5  # Goal not reached :(
			info = "'Time is up!'"
		self.time_step += 1

		# Punish jerk: [-2, 2]
		reward -= np.abs(action[0] - self.previous_throttle)
		reward -= np.abs(action[1] - self.previous_steering)
		self.previous_throttle = action[0]
		self.previous_steering = action[1]
		return new_state, reward, done, info

	def render(self, mode='human', close=False, render_all_frames=False, show_SC=False):
		if render_all_frames:
			for _, frame in enumerate(self.render_frames):
				self.render_one_frame(frame, show_SC=show_SC)
		else:
			self.render_one_frame(np.array([self.vehicle.X[0], self.vehicle.X[1], self.vehicle.heading]), show_SC=show_SC)

	def render_one_frame(self, state, show_SC=False):
		##################
		self.gfx.clear_canvas()
    	##################
		# Extract geometry
		length=self.vehicle.length
		width=self.vehicle.width
		center_pos = state[0:2]
		heading = state[2]

		# Vertices:
		verticesCCF = [np.array([width/2,  length/2 ]),
						np.array([-width/2, length/2 ]),
						np.array([-width/2, -length/2]),
						np.array([width/2,  -length/2])]

		angle = heading-np.pi/2
		R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])

		verticesWCF = []
		for vertex in verticesCCF:
			verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
		# Sides
		sides = [[verticesWCF[-1], verticesWCF[0]]]
		for i in range(len(verticesWCF)-1):
			sides.append([verticesWCF[i], verticesWCF[i+1]])

		self.gfx.draw_sides(sides)
		# Draw heading and center of vehicle
		self.gfx.draw_center_and_headings_simple(heading, center_pos)

		# Draw the goal and goal limit
		self.gfx.draw_goal_state(np.array([self.goal_x, self.goal_y]), threshold=self.goal_threshold)

		# Draw all static obstacles
		for obj in self.objects:
			self.gfx.draw_sides(obj.sides)

		if show_SC:
			self.gfx.draw_static_circogram_data(self, self.SC, self.vehicle)
		##################
		self.gfx.update_display()
		##################


class MPC_environment_v40(gym.Env): # 
	"""Custom Environment that follows gym interface.
	- This environments implement the MPC behaviour we are looking for!
	- Halucinated as well as real gaming
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, v_max=8, v_min=-2,
	       alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150,
		   episode_s=60, mpc=True, boost_N=None, environment_selection="four_walls"):
		super(MPC_environment_v40, self).__init__()
		#
		self.mpc = mpc
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		self.v_max = v_max
		self.v_min = v_min
		self.alpha_max = alpha_max
		self.tau_steering = tau_steering
		self.tau_throttle = tau_throttle
		self.horizon = horizon
		self.edge = edge
		self.episode_seconds = episode_s
		self.boost_N = boost_N # for boosting accuracy of vision box!
		self.environment_selection = environment_selection
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the goal_x, goal_y (IN CCF), speed and steering angle, as well as previous actions
		self.observation_space = spaces.Box(low=np.array([0, 0, -self.v_min, -alpha_max, -1, -1,
						    								0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0]),
											high=np.array(
											[200, 200, 	self.v_max, alpha_max, 1, 1,
	    												self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon]),
											dtype=np.float32)
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1080, 1920)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7, map_img_path="graphics/test_map_2.png") # Also initializes the display
			self.clock = pygame.time.Clock()
			self.fps = 1/sim_dt
		self.objects = []

	def add_objects(self, vertices):
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))

	def generate_new_goal_state(self):
		selection = np.random.randint(0, 4) # [low, high)
		if selection == 0: # WEST
			goal = np.array([25, 75])
		elif selection == 1: # NORTH
			# ADD LID - NOTE was too hard, for now... :(
			#vertices = np.array([[120, 65], [144, 65]])
			#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
			goal = np.array([132, 25])
			self.vehicle.heading=-np.pi/2
		elif selection == 2: # EAST
			goal = np.array([245, 75])
		elif selection == 3: # SOUTH
			goal = np.array([132, 135])
			# ADD LID
			#vertices = np.array([[120, 85], [144, 85]])
			#self.objects.append(Object(np.array([0, 0]), vertices=vertices))
			self.vehicle.heading=np.pi/2
		self.goal_x, self.goal_y = goal
		return goal
	
	def four_walls_envionment(self):
		""" Environment with four walls """
		# Volkswagen
		self.vehicle = Vehicle(np.array([132, 75]), length=4, width=2,
								heading=-np.pi/2, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
		self.outer_rim = Object(np.array([0, 0]), vertices=vertices)
		self.objects = [self.outer_rim]
		# Add the base walls
		vertices = np.array([[35, 90], [35, 60], [40, 60], [40, 90]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[115, 35], [115, 40], [145, 40], [145, 35]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[230, 60], [230, 90], [235, 90], [235, 60]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[115, 125], [115, 130], [145, 130], [145, 125]])
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		# Goal state
		goal = self.generate_new_goal_state()
		self.goal_stack = deque([goal])
		self.goal_x, self.goal_y = self.goal_stack.popleft()
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		#######################################
	
	def Naples_street(self):
		""" Environment with Slim streets """
		# Volkswagen
		self.vehicle = Vehicle(np.array([132, 140]), length=4, width=2,
								heading=-np.pi/2, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
		self.outer_rim = Object(np.array([0, 0]), vertices=vertices)
		self.objects = [self.outer_rim]
		# Add the base walls
		vertices = np.array([[125, 25], [125, 150], [120, 150], [120, 25]]) # left wall
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[120, 50], [120, 55], [5, 55], [5, 50]]) # Plaza_1 wall
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[140, 5], [140, 68], [145, 68], [145, 5]]) # right wall 1
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[140, 80], [140, 150], [145, 150], [145, 80]]) # right wall 2
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))
		vertices = np.array([[170, 60], [175, 60], [175, 80], [170, 80]]) # Blocker plate
		self.objects.append(Object(np.array([0, 0]), vertices=vertices))


		# Goal state
		self.goal_stack = deque([np.array([132, 15]), np.array([15, 15]), np.array([132, 15]), np.array([132, 70]), np.array([200, 70])])
		self.goal_x, self.goal_y = self.goal_stack.popleft()
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		#######################################

	def reset(self):
		""" Resets the environment"""
		##### Chose the objects to spawn in #####
		if self.environment_selection=="four_walls":
			self.four_walls_envionment()
		elif self.environment_selection=="naples_street":
			self.Naples_street()
		else:
			self.four_walls_envionment()
		#######################################
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_vel = 0
		steering_angle = 0
		#
		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, P2, _  = self.SC
		real_distances = d2 - d1
		self.previous_throttle_signal = 0
		self.previous_steering_signal = 0
		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_vel, steering_angle, self.previous_throttle_signal, self.previous_steering_signal, 
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)
		
		return new_state
	
	def update_vision(self, l_action_queue, d2, d2_halu):
		####################################
		# Do we need to update trajectory? #
		####################################
		update_vision = False
		
		# 1 Check if trajectory is expired
		if (l_action_queue) <= 1:
			return True
			#print("Out of actions!")
		
		# Retrieve corresponding SC from trajectory
		""" TODO: This requires tuning! """
		# 2 Outside is drastically different
		diff = d2_halu - d2*0.50
		value = np.sum(diff < 0)
		if value:
			return True
		# 3 Outside gotten inside
		diff = d2-(d2_halu*0.75) # Adding a bit of wiggle room
		value = np.sum(diff < 0) # if only one ray is "true" in the comparison
		if value:
			return True

	def hallucinate(self, trajectory_length, sim_dt, decision_dt, agent, add_noise=True, collision_stop=False, include_collision_state=False, goal_stop = True):
		""" This is where the vehucle hallucinates the future, predicting and avoiding crashes.
		parameters:
		- add_noise: should noise be added to the actions taken by the agent?
		- collision_stop: 
		- add_disturbance = DELTA [tau_v, tau_alpha, k_max, k_min, c_max, d] -> determines what distubances to add to each parameter. If == NONE it wont be any
		"""
		times = np.int32(decision_dt/sim_dt)
		##############################################
		# This only works in simulated environments  #
		##############################################
		if self.boost_N: # can boost accuracy of vision box in simulation
			_, _, _, P2, _ = self.vehicle.static_circogram_2(N=self.boost_N, list_objects_simul=self.objects, d_horizon=self.horizon)
		else: # just use the Current SC
			_, _, _, P2, _ = self.SC
		self.viz_box = Object(np.array([0, 0]), vertices=P2)

		# Generate the trajectory to follow
		halu_car = deepcopy(self.vehicle)
		sim_trajectory = deque()
		decision_trajectory = deque()
		action_queue = deque()
		halu_d2s = deque()
		states = deque()
		collided = False
		#
		previous_throttle_signal = self.previous_throttle_signal
		previous_steering_signal = self.previous_steering_signal
		for _ in range(trajectory_length):
			# Need to multiply by actual direction, as if not - it will always be a positive value.
			try:
				normed_velocity = np.linalg.norm(halu_car.X[2:])*halu_car.actual_direction
			except:
				normed_velocity = 0
			steering_angle = halu_car.alpha

			# Update goal poses in CCF (as CCF's origin has moved)
			goal_CCF = halu_car.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
			############################
			# Get the "new" static circogram
			N = 36
			horizon = 1000
			SC = halu_car.static_circogram_2(N, [self.viz_box], horizon)
			d1_, d2_, _, _, _ = SC
			real_distances = d2_ - d1_
			halu_d2s.append(d2_)
			# Generate new state
			state = np.array([goal_CCF[0], goal_CCF[1], normed_velocity, steering_angle, previous_throttle_signal, previous_steering_signal,
				real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
				real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
				real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
				real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
				real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
				real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
				],
				dtype=np.float32)
			states.append(state)
			##############################
			# LOOK for path terminations #
			##############################
			current_pos = halu_car.position_center
			if collision_stop:
				# The first check sees if we are inside the line, with our hull!
				# The second check sees if the trajectory line crosses the walls!
				collided = halu_car.collision_check(d1_, d2_) 
				if len(states) > 1:
					collided = collided or halu_car.path_collision(current_pos, prev_pos, self.viz_box)
				if collided:
					if include_collision_state:
						return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
					else:
						# Remove unwanted content
						not_smart_move = action_queue.pop() 
						collision_state = states.pop()
						# For visualisation purposes; remove trajectory going into the collision.
						bad_state = decision_trajectory.pop()
						for _ in range(times):
							bad_sub_states = sim_trajectory.pop()
						# Terminate the hallucination early
						return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
			#################
			# Goal reached! #
			#################
			# Stop when current state is a goal state
			if goal_stop:
				dist = np.linalg.norm(goal_CCF)
				if dist < self.goal_threshold:  # some threshold
					return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
			############################################
			# If no termination, choose **one** action #
			############################################
			act = agent.choose_action(state, add_noise=add_noise)
			action_queue.append(act)
			############################
			previous_throttle_signal = act[0]
			previous_steering_signal = act[1]
			prev_pos = current_pos

			
			#################################################################
			# To avoid numerical instability: run multiple small timesteps! #
			#################################################################
			# Translate from signals to actions
			throttle_signal = act[0]
			if throttle_signal >= 0:
				v_ref_t = self.v_max*throttle_signal
			else:
				v_ref_t = -self.v_min*throttle_signal
			steering_signal = act[1]
			alpha_ref_t = self.alpha_max*steering_signal
			#
			for _ in range(times):
				#
				halu_car.one_step_algorithm_2(alpha_ref=alpha_ref_t, v_ref=v_ref_t, dt=sim_dt)
				sim_trajectory.append(halu_car.position_center)
				#
			decision_trajectory.append(halu_car.position_center)
			#
		return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
	
	def step(self, action, add_disturbance=None):
		"""Execute one (decision) time step within the environment.
			- Unlike the halucinated situation, this is actual moving and learning.
			- In theory, The vehicle shouldn't collide with these steps; as trajectories are not allowed to collide!
				- In practice however, collisions could happen, and are accounted for.
		"""
		# Call upon the vehicle step action
		times = np.int32(self.decision_dt/self.sim_dt)
		if self.will_render:
			self.real_trajectory = []
		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal
		# The check if goal was reached during simulation
		goal_was_reached = False
		# NOTE this avoids numerical instability, by using sim_dt to simulate actions taken each decicion_dt
		for _ in range(times):
			if add_disturbance: 
				# 1 Add disturbances
				tau_v = self.vehicle.tau_throttle + add_disturbance[0]
				tau_alpha = self.vehicle.tau_steering + add_disturbance[1]
				k_max = self.v_max + add_disturbance[2]
				k_min = self.v_min + add_disturbance[3]
				c_max = self.alpha_max + add_disturbance[4]
				d = self.vehicle.d + add_disturbance[5]
				self.vehicle.one_step_algorithm_3(action=action, dt=self.sim_dt, params=[tau_v, tau_alpha, k_max, k_min, c_max, d])
			else:
				
				self.vehicle.one_step_algorithm_2(alpha_ref, v_ref, dt=self.sim_dt)
			
			# Check for goal in here!
			self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
			dist = np.linalg.norm(self.goal_CCF)
			if dist < self.goal_threshold:
				goal_was_reached = True # Goal was reached during simulation
			# For render purposes only
			if self.will_render:
				xpos, ypos = self.vehicle.position_center
				heading = self.vehicle.heading
				self.real_trajectory.append([xpos, ypos, heading])
		
		# After running n simulations steps:
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		# Update goal poses in CCF (as CCF's origin has moved)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		self.vehicle.collision_check(d1, d2)
		real_distances = d2 - d1

		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle, self.previous_throttle_signal, self.previous_steering_signal, 
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)


		######################################
		# Reward function! (Sparse for now?) #
		######################################
		reward = 0
		done = False
		info = "..."

		# Calculate goal distance
		dist = np.linalg.norm(self.goal_CCF)
		# Goal is reached!
		if dist < self.goal_threshold or goal_was_reached:  # some threshold
			reward += 1000 # Goal reached!
			if len(self.goal_stack) == 0:
				done = True
				info = "'Final goal reached!'"
			else:
				self.goal_x, self.goal_y = self.goal_stack.popleft()
				#print('Sub-goal reached!')
				info = "'Sub-goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.01

		if self.vehicle.collided:
			done = True
			reward = -500
			info = "'Collided'"

		# Time is up?
		elif self.time_step > self.episode_seconds*1/self.decision_dt:  # (30 sek)

			done = True
			reward += -5  # Goal not reached :(
			info = "'Time is up!'"
		self.time_step += 1

		# Punish jerk: [-2, 2]
		reward -= np.abs(action[0] - self.previous_throttle_signal)
		reward -= np.abs(action[1] - self.previous_steering_signal)
		self.previous_throttle_signal = action[0]
		self.previous_steering_signal = action[1]
		return new_state, reward, done, info
		

	def render(self, decision_trajectory, sim_trajectory, mode='human', display_vision_box=False):
		
		for _, state in enumerate(self.real_trajectory):

			##################
			self.gfx.clear_canvas()
			##################
			# Extract geometry
			length=self.vehicle.length
			width=self.vehicle.width
			center_pos = state[0:2]
			heading = state[2]

			# Vertices:
			verticesCCF = [np.array([width/2,  length/2 ]),
							np.array([-width/2, length/2 ]),
							np.array([-width/2, -length/2]),
							np.array([width/2,  -length/2])]

			angle = heading-np.pi/2
			R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
								[np.sin(angle), np.cos(angle)]])

			verticesWCF = []
			for vertex in verticesCCF:
				verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
			# Sides
			sides = [[verticesWCF[-1], verticesWCF[0]]]
			for i in range(len(verticesWCF)-1):
				sides.append([verticesWCF[i], verticesWCF[i+1]])

			self.gfx.draw_sides(sides)
			# Draw heading and center of vehicle
			self.gfx.draw_center_and_headings_simple(heading, center_pos)

			# Draw the goal and goal limit
			self.gfx.draw_goal_state(np.array([self.goal_x, self.goal_y]), threshold=self.goal_threshold)

			# Draw all static obstacles
			for obj in self.objects:
				self.gfx.draw_sides(obj.sides)

			# Draw all remaining points in trajectory
			for point in decision_trajectory:
				# Draw the simulates steps in between!
				for sim_point in sim_trajectory:
					self.gfx.draw_goal_state((sim_point[0], sim_point[1]), width=1)
				self.gfx.draw_goal_state((point[0], point[1]), width=3)
			
			# Draw vision box
			if display_vision_box:
				self.gfx.draw_one_object(self.viz_box, color=(255, 0, 0), width=4)

			self.clock.tick(self.fps) # fps
			self.gfx.display_fps(self.clock.get_fps(), font_size=32, color="red", where=(0,0))
			self.gfx.update_display()



class ClosedField_v23_dyna(gym.Env): # THE MILKMAN
	"""Custom Environment that follows gym interface.
	- Adds knowledge of previously chosen actions, and rewards smooth driving.
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, v_max=8, v_min=-2,
	       alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150,
		   episode_s=60, reverse_ok=False, total_goals=5):
		super(ClosedField_v23_dyna, self).__init__()
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		self.v_max = v_max
		self.v_min = v_min
		self.alpha_max = alpha_max
		self.tau_steering = tau_steering
		self.tau_throttle = tau_throttle
		self.horizon = horizon
		self.edge = edge
		self.selection = -1
		self.episode_seconds = episode_s
		self.reverse_ok = reverse_ok
		self.total_goals = total_goals
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the goal_x, goal_y (IN CCF), speed and steering angle
		self.observation_space = spaces.Box(low=np.array([0, 0, -self.v_min, -alpha_max, -1, -1,
						    								0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0]),
											high=np.array(
											[200, 200, 	self.v_max, alpha_max, 1, 1,
	    												self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon]),
											dtype=np.float32)
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1080, 1920)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7) # Also initializes the display
		self.objects = []

	def add_static_objects(self, vertices):
		obj = Object(np.array([0, 0]), vertices=vertices)
		self.objects.append(obj)
		self.static_obstacles.append(obj)

	def generate_new_goal_stack(self, gen_goals=5):
		self.goal_stack = deque()
		for _ in range(gen_goals): # 5 goals
			x_pos = np.random.randint(15, 260)
			y_pos = np.random.randint(15, 140)
			goal = np.array([x_pos, y_pos])
			self.goal_stack.append(goal)
		return self.goal_stack

	def dynamic_dojo(self):
		""" Environment with four walls """
		# Volkswagen
		self.vehicle = Vehicle(np.array([132, 75]), length=4, width=2,
								heading=-np.pi/2, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
		self.outer_rim = Object(np.array([0, 0]), vertices=vertices)
		
		#############################################################################################################################
		# Add other vehicles
		car1 = Vehicle(np.array([25, 35]),  length=8, width=4, heading=0,     tau_steering=1, tau_throttle=0.4, dt=0.1) 
		car2 = Vehicle(np.array([250, 35]), length=8, width=4, heading=np.pi, tau_steering=1, tau_throttle=0.4, dt=0.1) 
		car3 = Vehicle(np.array([25, 95]),  length=8, width=4, heading=0,     tau_steering=1, tau_throttle=0.4, dt=0.1) 
		car4 = Vehicle(np.array([250, 95]), length=8, width=4, heading=np.pi, tau_steering=1, tau_throttle=0.4, dt=0.1)
		car5 = Vehicle(np.array([25, 130]), length=8, width=4, heading=0, tau_steering=1, tau_throttle=0.4, dt=0.1)
		car6 = Vehicle(np.array([250, 130]), length=8, width=4, heading=np.pi, tau_steering=1, tau_throttle=0.4, dt=0.1)


		self.dynamic_obstacles = [car1, car2, car3, car4, car5, car6]
		self.static_obstacles = [self.outer_rim]
		self.objects = [car1, car2, car3, car4, car5, car6, self.outer_rim]
		# Spawn drivers
		alpha_max = 1.0 # Volkswagen
		v_max = 10 # 6 
		v_min = -4 
		self.limos = []
		for car in self.dynamic_obstacles:
			agent = Agent(v_max, v_min, alpha_max)
			# Make it a limo!
			limo = Limo(vehicle=car, driver=agent)
			self.limos.append(limo)
		# initial refs
		self.alpha_refs = np.zeros(len(self.dynamic_obstacles))
		self.v_refs = np.ones(len(self.dynamic_obstacles)) * 2
		#############################################################################################################################
		# Goal state
		self.generate_new_goal_stack(gen_goals=self.total_goals)
		self.goal_x, self.goal_y = self.goal_stack.popleft()
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		#######################################

	def reset(self):
		""" Resets the environment"""
		self.dynamic_dojo()
		#######################################
		#######################################
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_vel = 0
		steering_angle = 0
		#
		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		real_distances = d2 - d1
		#
		self.previous_throttle = 0
		self.previous_steering = 0
		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_vel, steering_angle, self.previous_throttle, self.previous_steering,
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)

		return new_state

	def step(self, action):
		"""Execute one time step within the environment"""
		times = np.int32(self.decision_dt/self.sim_dt)
		###################################################################################################################
		# First, all dynamic obstcles
		# Generate circogram
		N = 18
		horizon = 500
		if self.will_render:
			self.render_limo_frames = np.zeros((len(self.dynamic_obstacles), times, 4, 2))
		for n, car in enumerate(self.dynamic_obstacles):
			# Circograms!
			static_circogram = car.static_circogram_2(N, self.objects[0:n]+self.objects[n+1:], horizon)
			dynamic_circogram = car.dynamic_cicogram_2(static_circogram, self.alpha_refs[n], self.v_refs[n], seconds=3)
			#d1, d2, _, _, _ = static_circogram
			#car.collision_check(d1, d2)
			#
			limo = self.limos[n]
			self.v_refs[n], self.alpha_refs[n] = limo.driver.determined_driver(dynamic_circogram, static_circogram, self.v_refs[n], self.alpha_refs[n],
									risk_threshold = 0.2, stop_threshold = 4,  dist_wait=10, verbose=False)
			
			# Run one step
			for t in range(times):
				limo.vehicle.one_step_algorithm_2(alpha_ref=self.alpha_refs[n], v_ref=self.v_refs[n], dt=self.sim_dt)
				if self.will_render:
					self.render_limo_frames[n, t] = limo.vehicle.vertices
		###################################################################################################################
		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal
		# Call upon the vehicle step action
		# NOTE Do not need to take new decision every simulation step (0.1 sec)
		if self.will_render:
			self.render_frames = []

		goal_was_reached = False
		for _ in range(times):
			self.vehicle.one_step_algorithm(alpha_ref, v_ref)
			# For rendering in sim time
			xpos, ypos = self.vehicle.position_center
			heading = self.vehicle.heading
			# Check for goal in here!
			self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
			dist = np.linalg.norm(self.goal_CCF)
			if dist < self.goal_threshold:
				goal_was_reached = True # Goal was reached during simulation
			# 
			if self.will_render:
				self.render_frames.append([xpos, ypos, heading])
		
		# After running n simulations steps:
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		# Update goal poses in CCF (as CCF's origin has moved)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		self.vehicle.collision_check(d1, d2)
		real_distances = d2 - d1

		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle, self.previous_throttle, self.previous_steering,
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)


		######################################
		# Reward function! (Sparse for now?) #
		######################################
		self.time_step += 1
		reward = 0
		done = False
		info = "..."

		# Calculate goal distance
		dist = np.linalg.norm(self.goal_CCF)
		# Goal is reached!
		if dist < self.goal_threshold or goal_was_reached: #  and normed_velocity < 1
			reward += 1000 # Goal reached!
			if len(self.goal_stack) == 0:
				done = True
				info = "'Final goal reached!'"
			else:
				self.goal_x, self.goal_y = self.goal_stack.popleft()
				print('Sub-goal reached!')
				info = "'Sub-goal reached!'"
				self.time_step=0 # RESET! 

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.01

		if self.vehicle.collided:
			done = True
			reward = -500
			info = "'Collided'"

		# Time is up?
		elif self.time_step > self.episode_seconds/self.decision_dt:  # (30 sek)
			done = True
			reward += -5  # Goal not reached :(
			info = "'Time is up!'"
		

		# Punish jerk: [-2, 2]
		reward -= np.abs(action[0] - self.previous_throttle)
		reward -= np.abs(action[1] - self.previous_steering)
		self.previous_throttle = action[0]
		self.previous_steering = action[1]
		
		# What type of throttle do we want?
		if not self.reverse_ok:  # Punish reversing
			if action[0]<0: 
				reward -= 0.1
		else:
			if action[0]<0: 
				reward += 0.1

		return new_state, reward, done, info

	def render(self, mode='human', close=False, render_all_frames=False, show_SC=False):
		for t, frame in enumerate(self.render_frames):
			self.render_one_frame(frame, t, show_SC=show_SC, )

	def render_one_frame(self, state, t, show_SC=False, ):
		##################
		self.gfx.clear_canvas()
    	##################
		
		for n in range(len(self.dynamic_obstacles)):
			vertices = self.render_limo_frames[n, t]
			# Sides
			sides = [[vertices[-1], vertices[0]]]
			for i in range(len(vertices)-1):
				sides.append([vertices[i], vertices[i+1]])

			self.gfx.draw_sides(sides)

		# Extract geometry
		length=self.vehicle.length
		width=self.vehicle.width
		center_pos = state[0:2]
		heading = state[2]

		# Vertices:
		verticesCCF = [np.array([width/2,  length/2 ]),
						np.array([-width/2, length/2 ]),
						np.array([-width/2, -length/2]),
						np.array([width/2,  -length/2])]

		angle = heading-np.pi/2
		R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])

		verticesWCF = []
		for vertex in verticesCCF:
			verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
		# Sides
		sides = [[verticesWCF[-1], verticesWCF[0]]]
		for i in range(len(verticesWCF)-1):
			sides.append([verticesWCF[i], verticesWCF[i+1]])

		self.gfx.draw_sides(sides)
		# Draw heading and center of vehicle
		self.gfx.draw_center_and_headings_simple(heading, center_pos)

		# Draw the goal and goal limit
		self.gfx.draw_goal_state(np.array([self.goal_x, self.goal_y]), threshold=self.goal_threshold)

		# Draw all static obstacles
		for obj in self.static_obstacles:
			self.gfx.draw_sides(obj.sides)
		
		if show_SC:
			self.gfx.draw_static_circogram_data(self, self.SC, self.vehicle)
		##################
		self.gfx.update_display()
		##################


class MPC_environment_v41(gym.Env):
	"""Custom Environment that follows gym interface.
	- This environments implement the MPC behaviour we are looking for!
	- Halucinated as well as real gaming
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, sim_dt=0.1, decision_dt=0.5, render=False, v_max=8, v_min=-2,
	       alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150,
		   episode_s=60, mpc=True, boost_N=None):
		super(MPC_environment_v41, self).__init__()
		#
		self.mpc = mpc
		#
		self.sim_dt = sim_dt
		self.decision_dt = decision_dt
		self.v_max = v_max
		self.v_min = v_min
		self.alpha_max = alpha_max
		self.tau_steering = tau_steering
		self.tau_throttle = tau_throttle
		self.horizon = horizon
		self.edge = edge
		self.episode_seconds = episode_s
		self.boost_N = boost_N # for boosting accuracy of vision box!
		# Actions is to give input signal for throttle and steering.
		self.action_space = spaces.Box(low=np.array([-1, -1]), 
										high=np.array([1, 1]),
										dtype=np.float32)
		# Observations space is the goal_x, goal_y (IN CCF), speed and steering angle, as well as previous actions
		self.observation_space = spaces.Box(low=np.array([0, 0, -self.v_min, -alpha_max, -1, -1,
						    								0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0,
															0, 0, 0, 0, 0, 0]),
											high=np.array(
											[200, 200, 	self.v_max, alpha_max, 1, 1,
	    												self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon,
														self.horizon, self.horizon, self.horizon, self.horizon, self.horizon, self.horizon]),
											dtype=np.float32)
		#
		self.will_render=render
		if self.will_render: # For displaying
			MAP_DIMENSIONS = (1080, 1920)
			self.gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7, map_img_path="graphics/test_map_2.png") # Also initializes the display
			self.clock = pygame.time.Clock()
			self.fps = 1/sim_dt
		self.objects = []

	def add_static_objects(self, vertices):
		obj = Object(np.array([0, 0]), vertices=vertices)
		self.objects.append(obj)
		self.static_obstacles.append(obj)

	def generate_new_goal_stack(self):
		self.goal_stack = deque()
		for _ in range(5): # 5 goals
			x_pos = np.random.randint(15, 260)
			y_pos = np.random.randint(15, 140)
			goal = np.array([x_pos, y_pos])
			self.goal_stack.append(goal)
		return self.goal_stack

	def dynamic_dojo(self):
		""" Environment with four walls """
		# Volkswagen
		self.vehicle = Vehicle(np.array([132, 75]), length=4, width=2,
								heading=-np.pi/2, tau_steering=self.tau_steering, tau_throttle=self.tau_throttle, dt=self.sim_dt)
		#######################################
		# Spawn in the outer wall & obstacles #
		#######################################
		vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
		self.outer_rim = Object(np.array([0, 0]), vertices=vertices)
		
		#############################################################################################################################
		# Add other vehicles
		car1 = Vehicle(np.array([25, 35]),  length=8, width=4, heading=0,     tau_steering=1, tau_throttle=0.4, dt=0.1) 
		car2 = Vehicle(np.array([250, 35]), length=8, width=4, heading=np.pi, tau_steering=1, tau_throttle=0.4, dt=0.1) 
		car3 = Vehicle(np.array([25, 95]),  length=8, width=4, heading=0,     tau_steering=1, tau_throttle=0.4, dt=0.1) 
		car4 = Vehicle(np.array([250, 95]), length=8, width=4, heading=np.pi, tau_steering=1, tau_throttle=0.4, dt=0.1)
		car5 = Vehicle(np.array([25, 130]), length=8, width=4, heading=0, tau_steering=1, tau_throttle=0.4, dt=0.1)
		car6 = Vehicle(np.array([250, 130]), length=8, width=4, heading=np.pi, tau_steering=1, tau_throttle=0.4, dt=0.1)


		self.dynamic_obstacles = [car1, car2, car3, car4, car5, car6]
		self.static_obstacles = [self.outer_rim]
		self.objects = [car1, car2, car3, car4, car5, car6, self.outer_rim]
		# Spawn drivers
		alpha_max = 1.0 # Volkswagen
		v_max = 10 # 6 
		v_min = -4 
		self.limos = []
		for car in self.dynamic_obstacles:
			agent = Agent(v_max, v_min, alpha_max)
			# Make it a limo!
			limo = Limo(vehicle=car, driver=agent)
			self.limos.append(limo)
		# initial refs
		self.alpha_refs = np.zeros(len(self.dynamic_obstacles))
		self.v_refs = np.ones(len(self.dynamic_obstacles)) * 2
		#############################################################################################################################
		# Goal state
		self.generate_new_goal_stack()
		self.goal_x, self.goal_y = self.goal_stack.popleft()
		# Last but not least, turn goal states to CCF! (Has to be done after each step as well)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
		#######################################

	def reset(self):
		""" Resets the environment"""
		self.dynamic_dojo()
		#######################################
		# Determines if the time is up
		self.time_step = 0
		#
		self.goal_threshold = self.vehicle.length*1.2 # Give it some more wiggle room
		#
		normed_vel = 0
		steering_angle = 0
		#
		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, P2, _  = self.SC
		real_distances = d2 - d1
		self.previous_throttle_signal = 0
		self.previous_steering_signal = 0
		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_vel, steering_angle, self.previous_throttle_signal, self.previous_steering_signal, 
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)
		
		return new_state
	
	def update_vision(self, l_action_queue, d2, d2_halu, out_=0.50, in_=0.75):
		####################################
		# Do we need to update trajectory? #
		####################################
		update_vision = False
		
		# 1 Check if trajectory is expired
		if (l_action_queue) <= 1:
			return True
			#print("Out of actions!")
		
		# Retrieve corresponding SC from trajectory
		""" TODO: This requires tuning! """
		# 2 Outside is drastically different
		diff = d2_halu - d2*out_
		value = np.sum(diff < 0)
		if value:
			return True
		# 3 Outside gotten inside
		diff = d2-(d2_halu*in_) # Adding a bit of wiggle room
		value = np.sum(diff < 0) # if only one ray is "true" in the comparison
		if value:
			return True

	def hallucinate(self, trajectory_length, sim_dt, decision_dt, agent, add_noise=True, collision_stop=False, include_collision_state=False):
		""" This is where the vehucle hallucinates the future, predicting and avoiding crashes.
		parameters:
		- add_noise: should noise be added to the actions taken by the agent?
		- collision_stop: 
		- 
		"""
		times = np.int32(decision_dt/sim_dt)
		##############################################
		# This only works in simulated environments  #
		##############################################
		if self.boost_N: # can boost accuracy of vision box in simulation
			_, _, _, P2, _ = self.vehicle.static_circogram_2(N=self.boost_N, list_objects_simul=self.objects, d_horizon=self.horizon)
		else: # just use the Current SC
			_, _, _, P2, _ = self.SC
		self.viz_box = Object(np.array([0, 0]), vertices=P2)

		# Generate the trajectory to follow
		halu_car = deepcopy(self.vehicle)
		sim_trajectory = deque()
		decision_trajectory = deque()
		action_queue = deque()
		halu_d2s = deque()
		states = deque()
		collided = False
		#
		previous_throttle_signal = self.previous_throttle_signal
		previous_steering_signal = self.previous_steering_signal
		for _ in range(trajectory_length):
			# Need to multiply by actual direction, as if not - it will always be a positive value.
			try:
				normed_velocity = np.linalg.norm(halu_car.X[2:])*halu_car.actual_direction
			except:
				normed_velocity = 0
			steering_angle = halu_car.alpha

			# Update goal poses in CCF (as CCF's origin has moved)
			goal_CCF = halu_car.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
			############################
			# Get the "new" static circogram
			N = 36
			horizon = 1000
			SC = halu_car.static_circogram_2(N, [self.viz_box], horizon)
			d1_, d2_, _, _, _ = SC
			real_distances = d2_ - d1_
			halu_d2s.append(d2_)
			# Generate new state
			state = np.array([goal_CCF[0], goal_CCF[1], normed_velocity, steering_angle, previous_throttle_signal, previous_steering_signal,
				real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
				real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
				real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
				real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
				real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
				real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
				],
				dtype=np.float32)
			states.append(state)
			##############################
			# LOOK for path terminations #
			##############################
			current_pos = halu_car.position_center
			if collision_stop:
				# The first check sees if we are inside the line, with our hull!
				# The second check sees if the trajectory line crosses the walls!
				collided = halu_car.collision_check(d1_, d2_) 
				if len(states) > 1:
					collided = collided or halu_car.path_collision(current_pos, prev_pos, self.viz_box)
				if collided:
					if include_collision_state:
						return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
					else:
						# Remove unwanted content
						not_smart_move = action_queue.pop() 
						collision_state = states.pop()
						# For visualisation purposes; remove trajectory going into the collision.
						bad_state = decision_trajectory.pop()
						for _ in range(times):
							bad_sub_states = sim_trajectory.pop()
						# Terminate the hallucination early
						return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
			#################
			# Goal reached! #
			#################
			dist = np.linalg.norm(goal_CCF)
			if dist < self.goal_threshold:  # some threshold
				return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
			############################################
			# If no termination, choose **one** action #
			############################################
			act = agent.choose_action(state, add_noise=add_noise)
			action_queue.append(act)
			############################
			previous_throttle_signal = act[0]
			previous_steering_signal = act[1]
			prev_pos = current_pos
			# Translate action signals to steering signals
			""" TODO: add this to its own function """
			throttle_signal = act[0]
			if throttle_signal >= 0:
				v_ref_t = self.v_max*throttle_signal
			else:
				v_ref_t = -self.v_min*throttle_signal
			steering_signal = act[1]
			alpha_ref_t = self.alpha_max*steering_signal
			#################################################################
			# To avoid numerical instability: run multiple small timesteps! #
			#################################################################
			for _ in range(times):
				halu_car.one_step_algorithm_2(alpha_ref=alpha_ref_t, v_ref=v_ref_t, dt=sim_dt)
				sim_trajectory.append(halu_car.position_center)
			decision_trajectory.append(halu_car.position_center)
		
		return action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided
	
	def step(self, action):
		"""Execute one (decision) time step within the environment.
			- Unlike the halucinated situation, this is actual moving and learning.
			- In theory, The vehicle shouldn't collide with these steps; as trajectories are not allowed to collide!
				- In practice however, collisions could happen, and are accounted for.
		"""
		"""Execute one time step within the environment"""
		times = np.int32(self.decision_dt/self.sim_dt)
		###################################################################################################################
		# First, all dynamic obstcles
		# Generate circogram
		N = 18
		horizon = 500
		if self.will_render:
			self.render_limo_frames = np.zeros((len(self.dynamic_obstacles), times, 4, 2))
		for n, car in enumerate(self.dynamic_obstacles):
			# Circograms!
			static_circogram = car.static_circogram_2(N, self.objects[0:n]+self.objects[n+1:], horizon)
			dynamic_circogram = car.dynamic_cicogram_2(static_circogram, self.alpha_refs[n], self.v_refs[n], seconds=3)
			#d1, d2, _, _, _ = static_circogram
			#car.collision_check(d1, d2)
			#
			limo = self.limos[n]
			self.v_refs[n], self.alpha_refs[n] = limo.driver.determined_driver(dynamic_circogram, static_circogram, self.v_refs[n], self.alpha_refs[n],
									risk_threshold = 0.2, stop_threshold = 4,  dist_wait=10, verbose=False)
			
			# Run one step
			for t in range(times):
				limo.vehicle.one_step_algorithm_2(alpha_ref=self.alpha_refs[n], v_ref=self.v_refs[n], dt=self.sim_dt)
				if self.will_render:
					self.render_limo_frames[n, t] = limo.vehicle.vertices
		###################################################################################################################
		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action
		if self.will_render:
			self.real_trajectory = []
		# NOTE this avoids numerical instability, by using sim_dt to simulate actions taken each decicion_dt
		goal_was_reached = False
		for _ in range(times):
			self.vehicle.one_step_algorithm_2(alpha_ref, v_ref, dt=self.sim_dt)
			# Check for goal in here!
			self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))
			dist = np.linalg.norm(self.goal_CCF)
			if dist < self.goal_threshold:
				goal_was_reached = True # Goal was reached during simulation

			# For render purposes only
			if self.will_render:
				xpos, ypos = self.vehicle.position_center
				heading = self.vehicle.heading
				self.real_trajectory.append([xpos, ypos, heading])
		
		# After running n simulations steps:
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

		# Need to multiply by actual direction, as if not - it will always be a positive value.
		normed_velocity = np.linalg.norm(self.vehicle.X[2:])*self.vehicle.actual_direction
		steering_angle = self.vehicle.alpha

		# Update goal poses in CCF (as CCF's origin has moved)
		self.goal_CCF = self.vehicle.WCFtoCCF(np.array([self.goal_x, self.goal_y]))

		# Get the new static circogram
		self.SC = self.vehicle.static_circogram_2(N=36, list_objects_simul=self.objects, d_horizon=self.horizon)
		d1, d2, _, _, _  = self.SC
		self.vehicle.collision_check(d1, d2)
		real_distances = d2 - d1

		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_velocity, steering_angle, self.previous_throttle_signal, self.previous_steering_signal, 
			real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
			real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
			real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
			real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
			real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
			real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
			],
			dtype=np.float32)


		######################################
		# Reward function! (Sparse for now?) #
		######################################
		self.time_step += 1
		reward = 0
		done = False
		info = "..."

		# Calculate goal distance
		dist = np.linalg.norm(self.goal_CCF)
		# Goal is reached!
		if dist < self.goal_threshold or goal_was_reached: #  and normed_velocity < 1
			reward += 1000 # Goal reached!
			if len(self.goal_stack) == 0:
				done = True
				info = "'Final goal reached!'"
			else:
				self.goal_x, self.goal_y = self.goal_stack.popleft()
				print('Sub-goal reached!')
				info = "'Sub-goal reached!'"
				self.time_step=0 # RESET! 

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.01

		if self.vehicle.collided:
			done = True
			reward = -500
			info = "'Collided'"

		# Time is up?
		elif self.time_step > self.episode_seconds/self.decision_dt:  # (30 sek)
			done = True
			reward += -5  # Goal not reached :(
			info = "'Time is up!'"
		

		# Punish jerk: [-2, 2]
		reward -= np.abs(action[0] - self.previous_throttle_signal)
		reward -= np.abs(action[1] - self.previous_steering_signal)
		self.previous_throttle_signal = action[0]
		self.previous_steering_signal = action[1]
		
		# Last but not least, punish reversing slightly
		if action[0]<0: 
			reward -= 0.1

		return new_state, reward, done, info

	def render(self, decision_trajectory, sim_trajectory, mode='human', display_vision_box=False):
		
		for t, state in enumerate(self.real_trajectory):

			##################
			self.gfx.clear_canvas()
			##################
			# Extract geometry
			length=self.vehicle.length
			width=self.vehicle.width
			center_pos = state[0:2]
			heading = state[2]

			# Vertices:
			verticesCCF = [np.array([width/2,  length/2 ]),
							np.array([-width/2, length/2 ]),
							np.array([-width/2, -length/2]),
							np.array([width/2,  -length/2])]

			angle = heading-np.pi/2
			R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
								[np.sin(angle), np.cos(angle)]])

			verticesWCF = []
			for vertex in verticesCCF:
				verticesWCF.append(R_W_V@vertex + np.asarray(center_pos))
			# Sides
			sides = [[verticesWCF[-1], verticesWCF[0]]]
			for i in range(len(verticesWCF)-1):
				sides.append([verticesWCF[i], verticesWCF[i+1]])

			self.gfx.draw_sides(sides)
			# Draw heading and center of vehicle
			self.gfx.draw_center_and_headings_simple(heading, center_pos)

			# Draw the goal and goal limit
			self.gfx.draw_goal_state(np.array([self.goal_x, self.goal_y]), threshold=self.goal_threshold)

			# Draw all vehicles
			for n in range(len(self.dynamic_obstacles)):
				vertices = self.render_limo_frames[n, t]
				# Sides
				sides = [[vertices[-1], vertices[0]]]
				for i in range(len(vertices)-1):
					sides.append([vertices[i], vertices[i+1]])

				self.gfx.draw_sides(sides)
			
			# Draw all static obstacles
			for obj in self.static_obstacles:
				self.gfx.draw_sides(obj.sides)


			# Draw all remaining points in trajectory
			for point in decision_trajectory:
				# Draw the simulates steps in between!
				for sim_point in sim_trajectory:
					self.gfx.draw_goal_state((sim_point[0], sim_point[1]), width=1)
				self.gfx.draw_goal_state((point[0], point[1]), width=3)
			
			# Draw vision box
			if display_vision_box:
				self.gfx.draw_one_object(self.viz_box, color=(255, 0, 0), width=4)

			self.clock.tick(self.fps) # fps
			self.gfx.display_fps(self.clock.get_fps(), font_size=32, color="red", where=(0,0))
			self.gfx.update_display()
