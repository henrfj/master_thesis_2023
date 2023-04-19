from ddpg_torch import MPC_Agent, Agent
import gym
import numpy as np
from environments import OpenField_v00, OpenField_v01, OpenField_v10, ClosedField_v20, ClosedField_v21, MPC_environment_v40
import pygame
import sys


def visualize_v00(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, folder="DDPG/checkpoints/v00"):
    # Start a new environment
    env = OpenField_v00(sim_dt=sim_dt, decision_dt=decision_dt, render=True)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[3], tau=0.1, env=env, 
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    # Load a pre-trained model
    agent.load_models(Verbose=False)
    while repeat:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=render_all_frames)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def visualize_v01(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, folder="DDPG/checkpoints/v01"):
    # Start a new environment
    env = OpenField_v01(sim_dt=sim_dt, decision_dt=decision_dt, render=True)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[5], tau=0.1, env=env, 
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    # Load a pre-trained model
    agent.load_models(Verbose=False)
    while repeat:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=render_all_frames)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def visualize_v10(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, folder="DDPG/checkpoints/v10"):
    # Start a new environment
    env = OpenField_v10(sim_dt=sim_dt, decision_dt=decision_dt, render=True)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.1, env=env, 
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    # Load a pre-trained model
    agent.load_models(Verbose=False)
    while repeat:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=render_all_frames)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def milk_man_challenge_v0(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, env_selected="v10", folder="DDPG/checkpoints/v10",
                          vmax=30, v_min=-15, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7 ):
    """
    In this challenge the agent is made to follow sequentially generated goal poses.
    - In this v0, there is not time limit - so each goal stays until the agent can reach it.
    """
    # Start a new environment
    if env_selected=="v10":
        env = OpenField_v10(sim_dt=sim_dt, decision_dt=decision_dt, render=True)
        # Start a new agent
        agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.1, env=env, 
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    else:
        print("That env is not yet implemented for milkman challenge:", env)
        return ""

    # Load a pre-trained model
    agent.load_models(Verbose=False)
    # Start the environment with a "reset"
    obs = env.reset(vmax=vmax, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle) # restart env

    while repeat:
        # 
        done = False
        score = 0
        clock = pygame.time.Clock()
        env.time_step = 0
        #
        new_goal_x, new_goal_y = env.generate_new_goal_state_2(0, 150)
        env.goal_x = new_goal_x
        env.goal_y = new_goal_y

        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=render_all_frames)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def visualize_v20(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, env_selected="v20",  folder="DDPG/checkpoints/v20",
                          v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7):
    if env_selected=="v20":
        # Start new env
        env = ClosedField_v20(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                                   v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                     horizon=200, edge=150)

         # Start a new agent
        agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    elif env_selected=="v21":
        env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                                   v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                     horizon=200)
        # Start a new agent
        agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
                    batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    else:
        raise ValueError("Not implemented for that env")
    
    
    # Load a pre-trained model
    agent.load_models(Verbose=False)
    while repeat:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=True)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def milk_man_challenge_v1(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, env_selected="v20", folder="DDPG/checkpoints/v20",
                          v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7):
    """
    In this challenge the agent is made to follow sequentially generated goal poses.
    """
    # Start a new environment
    if env_selected=="v20":
        env = ClosedField_v20(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                               v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                 horizon=200, edge=150)
        # Start a new agent
        agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    else:
        print("That env is not yet implemented for milkman challenge v1:", env)
        return ""

    # Load a pre-trained model
    agent.load_models(Verbose=False)
    # Start the environment with a "reset"
    obs = env.reset() # restart env

    while repeat:
        # 
        done = False
        score = 0
        clock = pygame.time.Clock()
        env.time_step = 0
        #
        new_goal_x, new_goal_y = env.generate_new_goal_state(10, 150)
        env.goal_x = new_goal_x
        env.goal_y = new_goal_y

        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=render_all_frames)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def tunnel_challenge(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, env_selected="v20", folder="DDPG/checkpoints/v20",
                          v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7):
    """
    In this challenge the agent is made to follow sequentially generated goal poses, in a tunnel environment.
    """
    # Start a new environment
    if env_selected=="v20":
        env = ClosedField_v20(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                               v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                 horizon=200, edge=150)
        # Start a new agent
        agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[14], tau=0.1, env=env, 
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    else:
        print("That env is not yet implemented for tunnel challenge:", env)
        return ""

    # Load a pre-trained model
    agent.load_models(Verbose=False)
    # Start the environment with a "reset"
    obs = env.reset() # restart env

    ###############################################
    # Generate new objects & pre-determined goals #
    ###############################################
    env.objects = []
    vertices = np.array([[5, 5], [5, 150], [270, 150], [270, 5]])
    env.add_objects(vertices)
    vertices = np.array([[30, 35], [30, 40], [125, 40], [125, 35]])
    env.add_objects(vertices)
    vertices = np.array([[30, 65], [30, 70], [125, 70], [125, 65]])
    env.add_objects(vertices)
    vertices = np.array([[30, 40], [30, 65], [35, 65], [35, 40]])
    env.add_objects(vertices)
    #################################
    goal_poses = np.array([[15, 15], [15, 140], [260, 140], [260, 15], [45, 52]])
    #################################

    while repeat:
        # 
        done = False
        score = 0
        clock = pygame.time.Clock()
        env.time_step = 0
        #
        idx= np.random.randint(0, 5)
        new_goal_x, new_goal_y = goal_poses[idx]
        env.goal_x = new_goal_x
        env.goal_y = new_goal_y

        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=render_all_frames)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def milk_man_challenge_v2(repeat=False, sim_dt=0.1,decision_dt=0.1, render_all_frames=True, env_selected="v21", folder="DDPG/checkpoints/v21",
                          v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7):
    """
    In this challenge the agent is made to follow sequentially generated goal poses.
    Obstacles will always appear in the path of the goal state.
    """
    # Start a new environment
    if env_selected=="v21":
        env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                                   v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                     horizon=200)
        # Start a new agent
        agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[14], tau=0.1, env=env, 
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    else:
        print("That env is not yet implemented for milkman challenge v2:", env)
        return ""

    # Load a pre-trained model
    agent.load_models(Verbose=False)
    # Start the environment with a "reset"
    obs = env.reset() # restart env

    while repeat:
        # 
        done = False
        score = 0
        clock = pygame.time.Clock()
        env.time_step = 0
        #
        env.generate_new_goal_state()

        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=render_all_frames)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        if info=="'Collided'":
            return
        #env.close() # Automatically when env is garbage-collected.

############################################################## NEW technology! ##############################################################

def visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5):
    """
        - loadfolder="DDPG/checkpoints/v22"; holds a good starting point, based on a basic DDPG algorithm
        - loadfolder="DDPG/checkpoints/v40"; is a MPC-specifically trained agent to be visualized

    """
    
    # Initialization
    env = MPC_environment_v40(sim_dt=sim_dt, decision_dt=decision_dt, render=True, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=60, mpc=True, boost_N=False)
    env.reset()
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)
    np.random.seed(0)
    ###############################################################################################################
    # Sets certain parameters
    trajectory_length = np.int32(3.0/decision_dt) # to get 3 second trajectories
    ###############################################################################################################
    """ One episode"""
    while True: # Keeps repeating until exited
        obs = env.reset()
        done = False
        update_vision=True # need to make initial update
        """ One decision_dt """
        while not done:
            ##############
            # Get new SC #
            ##############
            real_circogram = env.SC
            d1, d2, _, P2, _ = real_circogram
            #
            if update_vision:
                action_queue, decision_trajectory, sim_trajectory, halu_d2s, _, _ = \
                    env.hallucinate(trajectory_length, sim_dt, decision_dt, agent)
                # TODO: add rejection to collision trajectories, even in display
                
                    
            ####################################
            # Do we need to update trajectory? #
            ####################################
            d2_halu = halu_d2s.popleft() # Retrieve believed/halucinated SC from trajectory
            update_vision = env.update_vision(len(action_queue), d2, d2_halu)

            #############################
            # Select and execute action #
            #############################
            act = action_queue.popleft()
            """ NOTE during env.step, the SC for the new step is calculated! """
            new_state, reward, done, info = env.step(act)
            # Remember the transition
            agent.remember(obs, act, reward, new_state, int(done))
            # Learn from replay buffer, given batch size
            agent.learn()
            
            ########################
            # End of state actions #
            ########################
            obs = new_state
            env.render(decision_trajectory, sim_trajectory, display_vision_box=True)
            # TODO: remove visited points of the trajectory, as we go...


if __name__ == "__main__":
    #################### WCF
    #visualize_v00(repeat=True, sim_dt=0.05,decision_dt=0.5, render_all_frames=True, folder="DDPG/checkpoints/v00")
    #visualize_v01(repeat=True, sim_dt=0.05,decision_dt=0.1, render_all_frames=True, folder="DDPG/checkpoints/v01")
    #visualize_v01(repeat=True, sim_dt=0.05,decision_dt=0.1, render_all_frames=True, folder="DDPG/checkpoints/v01_1") # Added reversing
    #################### CCF
    #visualize_v10(repeat=True, sim_dt=0.05,decision_dt=0.1, render_all_frames=True, folder="DDPG/checkpoints/v10")  
    #milk_man_challenge_v0(repeat=True, sim_dt=0.05,decision_dt=0.1, render_all_frames=True, env_selected="v10", folder="DDPG/checkpoints/v10_1")
    #################### With obstacles! 
    #visualize_v20(repeat=True, sim_dt=0.05, decision_dt=0.1, render_all_frames=True, env_selected="v20", folder="DDPG/checkpoints/v20")
    #visualize_v20(repeat=True, sim_dt=0.05, decision_dt=0.1, render_all_frames=True, env_selected="v20", folder="DDPG/checkpoints/v20_2",
    #              v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7)
    #milk_man_challenge_v1(repeat=True, sim_dt=0.05, decision_dt=0.1, render_all_frames=True, env_selected="v20", folder="DDPG/checkpoints/v20_2",
    #                      v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7)
    #visualize_v20(repeat=True, sim_dt=0.05, decision_dt=0.5, render_all_frames=True, env_selected="v21", folder="DDPG/checkpoints/v21_2",
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5)

    # MPC!
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22",
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5)
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v40", # Trained from scratch
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5)
    visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v40_22", # Trained from v22
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5)
