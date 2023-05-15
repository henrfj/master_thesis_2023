from ddpg_torch import MPC_Agent, Agent
import gym
import numpy as np
from environments import ClosedField_v22, MPC_environment_v40
import sys
from tqdm import tqdm

def data_v22(episodes = 10000, sim_dt=0.05, decision_dt=0.5, folder="...",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
                  store_run_data=None):

    # Data:
    collision_history = np.zeros((episodes,))
    score_history = np.zeros((episodes,))


    env = ClosedField_v22(sim_dt=sim_dt, decision_dt=decision_dt, render=False,
                                v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                    horizon=200, environment_selection=environment_selection)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
                    batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    
    # Load a pre-trained model
    agent.load_models(Verbose=True)

    for e in tqdm(range(episodes)):
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        while not done:
            act = agent.choose_action(obs, add_noise=False)
            new_state, reward, done, info = env.step(act)
            obs = new_state
            score += reward
        
        if info == "'Collided'":
            collision_history[e] = 1
        score_history[e] = score
        
    np.savetxt('DDPG/rundata/'+store_run_data+'_ch.txt', collision_history, fmt='%d')
    np.savetxt('DDPG/rundata/'+store_run_data+'_sh.txt', np.asarray(score_history))


def data_v40(episodes = 10000, sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
                  trajectory_time=3.0, add_noise=False, collision_stop=True, include_collision_state=False, store_run_data=None):
    """
        - loadfolder="DDPG/checkpoints/v22"; holds a good starting point, based on a basic DDPG algorithm
        - loadfolder="DDPG/checkpoints/v40"; is a MPC-specifically trained agent to be visualized
    """
    # Data:
    collision_history = np.zeros((episodes,))
    score_history = np.zeros((episodes,))
    
    # Initialization
    env = MPC_environment_v40(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=100, mpc=True, boost_N=False, environment_selection=environment_selection)
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)

    ###############################################################################################################
    # Sets certain parameters
    trajectory_length = np.int32(trajectory_time/decision_dt) # to get 3 second trajectories
    ###############################################################################################################
    """ One episode"""
    for e in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        update_vision=True # need to make initial update
        score = 0
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
            next_state, reward, done, info = env.step(act)

        
            ########################
            # End of state actions #
            ########################
            current_state = next_state
            score += reward

        if info == "'Collided'":
            collision_history[e] = 1
        score_history[e] = score

    np.savetxt('DDPG/rundata/'+store_run_data+'_ch.txt', collision_history, fmt='%d')
    np.savetxt('DDPG/rundata/'+store_run_data+'_sh.txt', np.asarray(score_history))

if __name__ =="__main__":
    #######
    # v22 #
    #######
    #data_v22(episodes = 10000, sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22",
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
    #              store_run_data="v22_fw_data")


    #data_v22(episodes = 10000, sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22_naples_nn",
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="naples_street",
    #              store_run_data="v22_naples_data")
    
    #######
    # v40 #
    #######

    #data_v40(episodes = 10000, sim_dt=0.05, decision_dt=0.5, save_folder="", loadfolder="DDPG/checkpoints/v22",
    #              v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
    #              trajectory_time=3.0, add_noise=False, collision_stop=True, include_collision_state=False, store_run_data="v40_v22_fw_data")

    data_v40(episodes = 10000, sim_dt=0.05, decision_dt=0.5, save_folder="", loadfolder="DDPG/checkpoints/v22_naples_nn",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="naples_street",
                  trajectory_time=3.0, add_noise=False, collision_stop=True, include_collision_state=False, store_run_data="v40_v22_naples_data")

    pass