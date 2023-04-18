from ddpg_torch import Agent, MPC_Agent
import gym
import numpy as np
import os
from collections import deque
from copy import deepcopy
# My own env!
from environments import OpenField_v00, OpenField_v01, OpenField_v10, ClosedField_v20, ClosedField_v21, ClosedField_v22, MPC_environment_v40
from utils import plotLearning

def open_field_v00_training(episodes=5000, sim_dt=0.1, decision_dt=0.1):
    #env = gym.make('LunarLanderContinuous-v2')
    env = OpenField_v00(sim_dt=sim_dt, decision_dt=decision_dt, render=False)

    #best_score = env.reward_range[0] # Bound of performance
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[3], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir="DDPG/checkpoints/v00")

    #agent.load_models()
    np.random.seed(0)

    score_history = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()



        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)

    print(os.getcwd())
    filename = 'DDPG/plots/openfield_v00.png'
    plotLearning(score_history, filename, window=100)

def open_field_v01_training(episodes=5000, sim_dt=0.1,decision_dt=0.1):
    env = OpenField_v01(sim_dt=sim_dt, decision_dt=decision_dt, render=False, vmax=8)

    #best_score = env.reward_range[0] # Bound of performance
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[5], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir="DDPG/checkpoints/v01_1")

    #agent.load_models()
    np.random.seed(0)

    score_history = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()



        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)

    print(os.getcwd())
    filename = 'DDPG/plots/openfield_v01.png'
    plotLearning(score_history, filename, window=100)

def open_field_v10_training(episodes=5000, sim_dt=0.1,decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v10", filename = 'DDPG/plots/openfield_v10.png'):
    env = OpenField_v10(sim_dt=sim_dt, decision_dt=decision_dt, render=False, vmax=8)

    #best_score = env.reward_range[0] # Bound of performance
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=chkpt_dir)

    #agent.load_models()
    np.random.seed(0)

    score_history = []
    for i in range(episodes):
        obs = env.reset(vmax=30, v_min=-15, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7)
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()



        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)

    plotLearning(score_history, filename, window=100)

def v20_training(episodes=5000, sim_dt=0.1,decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v20", filename = 'DDPG/plots/openfield_v20.png'):
    env = ClosedField_v20(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7, horizon=200, edge=150)
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=chkpt_dir)
    np.random.seed(0)

    score_history = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)

    plotLearning(score_history, filename, window=100)

def v21_training(episodes=5000, sim_dt=0.1,decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v21", filename = 'DDPG/plots/openfield_v21.png'):
    env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, horizon=200)
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=chkpt_dir)
    np.random.seed(0)

    score_history = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)

    plotLearning(score_history, filename, window=100)

def v22_training(episodes=5000, sim_dt=0.1,decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v22", filename = 'DDPG/plots/openfield_v22.png'):
    """
    Here, we added knowledge of previous action, and punish for jerk.
    """
    env = ClosedField_v22(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, horizon=200)
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=chkpt_dir)
    np.random.seed(42)

    score_history = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)

    plotLearning(score_history, filename, window=100)


def v40_MPC_training(episodes=5000, sim_dt=0.05, decision_dt=0.5, plotting = 'DDPG/plots/mpc_v40.png', save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22_5"):
    ###############################################################################################################
    # Parameters
    times = np.int32(decision_dt/sim_dt)
    v_max=20
    v_min=-4
    alpha_max=0.5
    tau_steering=0.5
    tau_throttle=0.5
    # Initialization
    env = MPC_environment_v40(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=60, mpc=True)
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
    # Book-keeping during training
    score_history = []
    actual_collisions_during_training = 0
    ###############################################################################################################
    """ One episode"""
    for i in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        update_vision=True # need to make initial update
        rejected_trajectories = 0
        """ One decision_dt """
        while not done:
            ##############
            # Get new SC #
            ##############
            real_circogram = env.SC
            d1, d2, _, P2, _ = real_circogram

            # Do not allow colliding trajectories
            while update_vision: 
                if update_vision:
                    action_queue, decision_trajectory, sim_trajectory, halu_d2s, states, collided = \
                        env.hallucinate(P2, trajectory_length, sim_dt, decision_dt, agent)
                    
                    # What if the vehicle did collide in its planned trajectory? (meaning Collided == True)
                    if collided: # need to learn from that experience
                        rejected_trajectories += 1
                        # 1 Add all halucinated knowledge to memory
                        disc = 0
                        for i in range(len(states)-1, 0, -1):
                            next_state = states[i]
                            current_state = states[i-1]
                            action = action_queue.pop()
                            R = -40 * agent.gamma**disc # Discounted collision reward
                            disc += 1 # Discount growing back in time.
                            # Only the last state is the "done" staet, where collision happend
                            if i==len(states)-1:
                                done = True
                            else:
                                done = False
                            agent.remember_halu(current_state, action, R, next_state, int(done))
                        # 2 Learn from hallucinated memory
                        agent.learn(halu=True)
                    else:
                        # Now we are happy
                        update_vision = False
                        

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
            
            ################
            # Book-keeping #
            ################
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        ####################################
        ## BOOK-Keeping from the training ##
        ####################################
        score_history.append(score)
        if info == "'Collided'":
            actual_collisions_during_training +=1
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)
        print("Rejected trajectories:", rejected_trajectories)
        ####################################

    plotLearning(score_history, plotting, window=100)
    print("Actual collisions during training:", actual_collisions_during_training)

if __name__ =="__main__":
    # IN WCF
    #open_field_v00_training(episodes=2000, sim_dt=0.1, decision_dt=0.1)
    #open_field_v01_training(episodes=5000, sim_dt=0.1, decision_dt=0.1)
    # IN CCF
    #open_field_v10_training(episodes=5000, sim_dt=0.1, decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v10_1", filename = 'DDPG/plots/openfield_v10_1.png')
    # Obstacles
    #v20_training(episodes=10000, sim_dt=0.1, decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v20_2", filename = 'DDPG/plots/openfield_v20_2.png')
    #v21_training(episodes=50000, sim_dt=0.1, decision_dt=0.5, chkpt_dir="DDPG/checkpoints/v21_2", filename = 'DDPG/plots/openfield_v21_2.png')
    # JERK ADDED
    #v22_training(episodes=50000, sim_dt=0.05, decision_dt=0.5, chkpt_dir="DDPG/checkpoints/v22", filename = 'DDPG/plots/openfield_v22.png')
    # MPC
    v40_MPC_training(episodes=50000, sim_dt=0.05, decision_dt=0.5, plotting = 'DDPG/plots/mpc_v40.png', save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22")