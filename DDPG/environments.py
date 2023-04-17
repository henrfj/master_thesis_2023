import gym
from gym import spaces
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
# My own libraries
#import a_star_utils as autils
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/henri/Desktop/Simple_simulator')
from Vehicle import Vehicle
from Agent import Agent
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

	def __init__(self, sim_dt=0.1, SC_dt=0.5, render=False, v_max=8, v_min=-2,
	       alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150,
		   episode_s=60, mpc=True):
		super(MPC_environment_v40, self).__init__()
		#
		self.mpc = mpc
		#
		self.sim_dt = sim_dt
		self.SC_dt = SC_dt
		self.v_max = v_max
		self.v_min = v_min
		self.alpha_max = alpha_max
		self.tau_steering = tau_steering
		self.tau_throttle = tau_throttle
		self.horizon = horizon
		self.edge = edge
		self.episode_seconds = episode_s

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
		# Goal state
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
		d1, d2, _, P2, _  = self.SC
		real_distances = d2 - d1
		previous_steering_signal = 0
		previous_throttle_signal = 0
		# Generate new state
		new_state = np.array([self.goal_CCF[0], self.goal_CCF[1], normed_vel, steering_angle, previous_steering_signal, previous_throttle_signal,
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
		"""Execute an actual decision step, in the environment"""

		# Translate action signals to steering signals
		throttle_signal = action[0]
		if throttle_signal >= 0:
			v_ref = self.v_max*throttle_signal
		else:
			v_ref = -self.v_min*throttle_signal
		steering_signal = action[1]
		alpha_ref = self.alpha_max*steering_signal

		# Call upon the vehicle step action)
		self.vehicle.one_step_algorithm_2(alpha_ref, v_ref, dt=self.sim_dt)
		# For rendering in sim time
		xpos, ypos = self.vehicle.position_center
		heading = self.vehicle.heading

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
			reward += 400  # Goal reached!
			done = True
			info = "'Goal reached!'"

		else:  # punish for further distance ( hill climb? )
			# NOTE hill climber only gives flat negative reward...
			reward += -dist*0.05

		if self.vehicle.collided:
			done = True
			reward = -1000
			info = "'Collided'"

		# Time is up?
		elif self.time_step > self.episode_seconds*1/self.decision_dt:  # (30 sek)
			done = True
			reward += -1000  # Goal not reached :(
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
