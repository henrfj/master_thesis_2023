from ddpg_torch import Agent, MPC_Agent
import gym
import numpy as np
import os
# My own env!
from environments import OpenField_v00, OpenField_v01, OpenField_v10, ClosedField_v20, ClosedField_v21, MPC_environment_v40
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
    env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, horizon=200)
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
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


def v40_MPC_training(episodes=5000, sim_dt=0.1, SC_dt=0.5, plotting = 'DDPG/plots/mpc_v40.png', save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v21_2"):
    env = MPC_environment_v40(sim_dt=0.1, SC_dt=0.5, render=False, v_max=8, v_min=-2,
	       alpha_max=0.8, tau_steering=0.4, tau_throttle=0.4, horizon=200, edge=150,
		   episode_s=60, mpc=True)
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        # TODO: specify where to load models from!
        agent.load_models(load_directory=loadfolder)
    
    np.random.seed(0)
    #####################################################
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

    plotLearning(score_history, plotting, window=100)


if __name__ =="__main__":
    # IN WCF
    #open_field_v00_training(episodes=2000, sim_dt=0.1, decision_dt=0.1)
    #open_field_v01_training(episodes=5000, sim_dt=0.1, decision_dt=0.1)
    # IN CCF
    #open_field_v10_training(episodes=5000, sim_dt=0.1, decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v10_1", filename = 'DDPG/plots/openfield_v10_1.png')
    # Obstacles
    #v20_training(episodes=10000, sim_dt=0.1, decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v20_2", filename = 'DDPG/plots/openfield_v20_2.png')
    #v21_training(episodes=50000, sim_dt=0.1, decision_dt=0.5, chkpt_dir="DDPG/checkpoints/v21_2", filename = 'DDPG/plots/openfield_v21_2.png')
    v22_training(episodes=50000, sim_dt=0.1, decision_dt=0.1, chkpt_dir="DDPG/checkpoints/v22", filename = 'DDPG/plots/openfield_v22.png') # Added jerk control
    # MPC