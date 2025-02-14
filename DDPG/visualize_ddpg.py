#from .ddpg_torch import MPC_Agent, Agent
from .ddpg_torch import MPC_Agent, Agent
import gymnasium as gym
import numpy as np
from .environments import OpenField_v00, OpenField_v01, OpenField_v10, ClosedField_v20, ClosedField_v21, ClosedField_v22, ClosedField_v23_dyna, MPC_environment_v40, MPC_environment_v41
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

def visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="...",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
                  ):

    env = ClosedField_v22(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                                v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                    horizon=200, environment_selection=environment_selection)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
                    batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    
    # Load a pre-trained model
    agent.load_models(Verbose=True)
    render_all_frames=True
    while True:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            act = agent.choose_action(obs, add_noise=False)
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

def visualize_v23(sim_dt=0.05, decision_dt=0.5, folder="...",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, user_controlled=False):

    env = ClosedField_v23_dyna(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                                v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                    horizon=200, episode_s=200)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
                    batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    
    # Load a pre-trained model
    agent.load_models(Verbose=True)
    while True:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            if user_controlled:
                act = np.array([0, 0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # Press x button
                        #exit()
                        pygame.quit()
                        sys.exit()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    #print("UP")
                    act[0] = 1
                if keys[pygame.K_DOWN]:
                    #print("DOWN")
                    act[0] = -1
                if keys[pygame.K_LEFT]:
                    #print("LEFT")
                    act[1] = -1
                if keys[pygame.K_RIGHT]:
                    #print("RIGHT")
                    act[1] = 1
            else:
                act = agent.choose_action(obs, add_noise=False)

            new_state, reward, done, info = env.step(act)
            obs = new_state
            score += reward

            

            env.render(render_all_frames=True)
            clock.tick(1/sim_dt)

        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.


############################################################## NEW technology! ##############################################################

def visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
                  trajectory_time=3.0, user_controlled=False, add_noise=False, collision_stop=True, include_collision_state=False,
                  goal_stop=True, add_disturbance=None):
    """
        - loadfolder="DDPG/checkpoints/v22"; holds a good starting point, based on a basic DDPG algorithm
        - loadfolder="DDPG/checkpoints/v40"; is a MPC-specifically trained agent to be visualized
    """
    
    # Initialization
    env = MPC_environment_v40(sim_dt=sim_dt, decision_dt=decision_dt, render=True, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=100, mpc=True, boost_N=False, environment_selection=environment_selection)
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)
    #np.random.seed(0)
    ###############################################################################################################
    # Sets certain parameters
    trajectory_length = np.int32(trajectory_time/decision_dt) # to get 3 second trajectories
    ###############################################################################################################
    """ One episode"""
    while True: # Keeps repeating until exited
        current_state = env.reset()
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
                action_queue, decision_trajectory, sim_trajectory, halu_d2s, halu_states, collided = \
                    env.hallucinate(trajectory_length, sim_dt, decision_dt, agent, add_noise=add_noise, collision_stop=collision_stop, include_collision_state=include_collision_state,
                                    goal_stop=goal_stop)
                # TODO: add rejection to collision trajectories, even in display

 
            ####################################
            # Do we need to update trajectory? #
            ####################################
            d2_halu = halu_d2s.popleft() # Retrieve believed/halucinated SC from trajectory
            update_vision = env.update_vision(len(action_queue), d2, d2_halu)


            #############################
            # Select and execute action #
            #############################
            if len(action_queue) == 0:
                # This happens when trajectory is only 1 step, and it lead to collision... :(
                print("No actions")
                act = [0, 0]
            else:
                act = action_queue.popleft()
            """ NOTE during env.step, the SC for the new step is calculated! """

            if user_controlled:
                act = np.array([0, 0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # Press x button
                        #exit()
                        pygame.quit()
                        sys.exit()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    #print("UP")
                    act[0] = 1
                if keys[pygame.K_DOWN]:
                    #print("DOWN")
                    act[0] = -1
                if keys[pygame.K_LEFT]:
                    #print("LEFT")
                    act[1] = -1
                if keys[pygame.K_RIGHT]:
                    #print("RIGHT")
                    act[1] = 1

            next_state, _, done, info = env.step(act, add_disturbance=add_disturbance)

        
            ########################
            # End of state actions #
            ########################
            current_state = next_state
            env.render(decision_trajectory, sim_trajectory, display_vision_box=True)
            # TODO: remove visited points of the trajectory, as we go...
        print("Episode finished with code;", info)

def visualize_v41(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22",
                    v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, trajectory_time=3.0,
                    user_controlled=False, add_noise=False, collision_stop=True, include_collision_state=False, repeat_forever=False):
    
    """
        - loadfolder="DDPG/checkpoints/v22"; holds a good starting point, based on a basic DDPG algorithm
        - loadfolder="DDPG/checkpoints/v40"; is a MPC-specifically trained agent to be visualized
    """
    
    # Initialization
    env = MPC_environment_v41(sim_dt=sim_dt, decision_dt=decision_dt, render=True, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=100, mpc=True, boost_N=False)
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)
    #np.random.seed(0)
    ###############################################################################################################
    # Sets certain parameters
    trajectory_length = np.int32(trajectory_time/decision_dt) # to get 3 second trajectories
    ###############################################################################################################
    """ One episode"""
    while True: # Keeps repeating until exited
        current_state = env.reset()
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
                action_queue, decision_trajectory, sim_trajectory, halu_d2s, _, collided = \
                    env.hallucinate(trajectory_length, sim_dt, decision_dt, agent, add_noise=add_noise, collision_stop=collision_stop, include_collision_state=include_collision_state)
                # TODO: add rejection to collision trajectories, even in display
                
 
            ####################################
            # Do we need to update trajectory? #
            ####################################
            d2_halu = halu_d2s.popleft() # Retrieve believed/halucinated SC from trajectory
            update_vision = env.update_vision(len(action_queue), d2, d2_halu, out_=0.9, in_=0.2) # Update very time!


            #############################
            # Select and execute action #
            #############################
            if len(action_queue) == 0:
                # This happens when trajectory is only 1 step, and it lead to collision... :(
                print("No actions")
                act = [0, 0]
            else:
                act = action_queue.popleft()
            # NOTE: often the last action in trajectory is right in front of collision, so it eventually leads to collision
            #if len(action_queue) == 0:
            #    print("Last action: stop!")
            #    act = [0, 0]

            if user_controlled:
                act = np.array([0, 0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # Press x button
                        #exit()
                        pygame.quit()
                        sys.exit()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    #print("UP")
                    act[0] = 1
                if keys[pygame.K_DOWN]:
                    #print("DOWN")
                    act[0] = -1
                if keys[pygame.K_LEFT]:
                    #print("LEFT")
                    act[1] = -1
                if keys[pygame.K_RIGHT]:
                    #print("RIGHT")
                    act[1] = 1

            next_state, _, done, _ = env.step(act)
            if repeat_forever and len(env.goal_stack)==1:
                env.generate_new_goal_stack()
        
            ########################
            # End of state actions #
            ########################
            current_state = next_state
            env.render(decision_trajectory, sim_trajectory, display_vision_box=True)
            





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
    #visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22", # VERY GOOD
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls")
    #visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22_fw", # FLOPPED in training
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls")
    # NAPLES v22
    #visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22_naples", # Noisy training
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="naples_street")
    #visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22_naples_nn", # No noise in training! Very smooth!
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="naples_street")

    #visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22_fw_plotty", # Is used for plotting, and training v40's
    #          v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls")


    # MPC!
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22", # just using v22
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls", user_controlled=False,
    #              add_disturbance=None)
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v40_22", # Trained from v22; smooth driver
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5)
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v40_fw", # Trained from v22; in four walls environment; smooth solver
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls", trajectory_time = 10.0, include_collision_state=True)
    
    
    # GOOD STUFF: illustrates the strenghts of v40, while using v22 directly.
    #visualize_v40(sim_dt=0.05, decision_dt=0.25, save_folder="None", loadfolder="DDPG/checkpoints/v22_naples_nn", # Using v22_nn; in naples environment - GOOD
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="naples_street", trajectory_time = 10.0,
    #                add_noise=False, collision_stop=True, include_collision_state=False)
    # Shorter planning horizon
    visualize_v40(sim_dt=0.02, decision_dt=0.25, save_folder="None", loadfolder="DDPG/checkpoints/v22_naples_nn", # Using v22_nn; in naples environment - GOOD
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="naples_street", trajectory_time = 2.0,
                    add_noise=False, collision_stop=True, include_collision_state=False)
    
    #############################################
    # V40 self trained
    
    # IRL trainer of v40 - from scratch!!!!!!!!!!!!!!! # Suprise surprise! IT WORKS!!! BUT 856 Collisions in training!
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="None", loadfolder="DDPG/checkpoints/v40_IRL_fw", 
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls", trajectory_time = 5.0, 
    #              add_noise=False, collision_stop=True, include_collision_state=False)
    
    # Score looks terrible - it looks terrible
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="None", loadfolder="DDPG/checkpoints/v40_fw_plotty", 
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls", trajectory_time = 10.0, 
    #              add_noise=False, collision_stop=True, include_collision_state=False)
    
    # Score looks terrible
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="None", loadfolder="DDPG/checkpoints/v40_naples_plotty", 
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="naples_street", trajectory_time = 10.0, 
    #              add_noise=False, collision_stop=True, include_collision_state=False)
    
    #############################################
    # Dynamic! Does collide from time to times - but how often?
    #visualize_v23(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v23",
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, user_controlled=False)
    
    #visualize_v23(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v23_reverser",
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, user_controlled=False)
    
    # Dynamic v41 environment
    # TODO: use collision_test, to also (Instead?) trigger "update vision" if the trajectory is intersected.
    #visualize_v41(sim_dt=0.05, decision_dt=0.5, save_folder="None", loadfolder="DDPG/checkpoints/v23", 
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, trajectory_time = 3.0, 
    #              add_noise=False, collision_stop=False, include_collision_state=True, repeat_forever=True)
    
    
    #############################################
    # Adding disturbance => DELTA [tau_v, tau_alpha, k_max, k_min, c_max, d]
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22", # just using v22
    #                v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls", user_controlled=False,
    #                add_noise=False, collision_stop=True, include_collision_state=False,
    #                goal_stop=True, add_disturbance=[-0.1, -0.1, -1, -1, 0.5, 1], trajectory_time=10.0)

    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22", # just using v22
    #                v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls", user_controlled=False,
    #                add_noise=False, collision_stop=True, include_collision_state=False,
    #                goal_stop=True, add_disturbance=[-0.2, -0.2, -5, -5, 0.5, 2])
    
    ### TRAINED FOR disturbance6 ###
    #visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v40_v22_fw_dist", # just using v22
    #                v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls", user_controlled=False,
    #                add_noise=False, collision_stop=True, include_collision_state=False,
    #                goal_stop=True, add_disturbance=[-0.2, -0.2, -5, -5, 0.5, 2])
    
    pass

