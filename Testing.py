from Vehicle import Vehicle
from limo import Limo
from visualization import *
import numpy as np
import sys
from DDPG.ddpg_torch import Agent as DDPG_Agent
from copy import deepcopy
from DDPG.environments import ClosedField_v21
import time
from collections import deque

""" ... """
def test_numerical_stability():
    """
    # PARAMETERS:
    """   
    #sim_dt=0.25
    sim_dt = 0.5
    #sim_dt = 0.0333333333333333333333333333 #(1/30)
    decision_dt = 0.5
    env="v21"
    folder="DDPG/checkpoints/v21_2"
    v_max=20
    v_min=-4
    alpha_max=0.5
    tau_steering=0.5
    tau_throttle=0.5
     
    # Start a new agent
    agent = DDPG_Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    # Load the pre-trained model
    agent.load_models(Verbose=False)

    # Graphics
    MAP_DIMENSIONS = (1080, 1920)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7, map_img_path="graphics/test_map_2.png")
    # Environment
    env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200)
    env.reset()
    # vehicle
    car1 = env.vehicle
    objects = env.objects
    goal_x, goal_y = env.goal_x, env.goal_y

    # Sets certain parameters
    collision = False
    trajectory_length = 10
    clock = pygame.time.Clock()
    fps = 1/sim_dt
    update_vision=True # need to make initial update

    while True:
        # Default steering parameters
        alpha_ref = 0
        v_ref = 0 

        if not collision:
            ###############
            # USER INPUTS #
            ###############

            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    pygame.quit()
                    sys.exit()
        
            ###########################
            # Vision box & Trajectory #
            ###########################
            if update_vision:
                print("Update time!")
                # Generate circogram
                N = 36
                horizon = 1000
                real_circogram = car1.static_circogram_2(N, objects, horizon)
                d1, d2, _, P2, _ = real_circogram
                collision = car1.collision_check(d1, d2)
                if collision:
                    print("Collision!")
                    continue
                viz_box = Object(np.array([0, 0]), vertices=P2)
                update_vision = False

                # Generate the trajectory to follow
                halu_car = deepcopy(car1)
                trajectory = deque()
                action_queue = deque()
                for _ in range(trajectory_length):
                    # Need to multiply by actual direction, as if not - it will always be a positive value.
                    try:
                        normed_velocity = np.linalg.norm(halu_car.X[2:])*halu_car.actual_direction
                    except:
                        normed_velocity = 0
                    steering_angle = halu_car.alpha

                    # Update goal poses in CCF (as CCF's origin has moved)
                    goal_CCF = halu_car.WCFtoCCF(np.array([goal_x, goal_y]))

                    # Get the "new" static circogram
                    N = 36
                    horizon = 1000
                    SC = halu_car.static_circogram_2(N, [viz_box], horizon)
                    d1_, d2_, _, _, _ = SC
                    real_distances = d2_ - d1_

                    # Generate new state
                    state = np.array([goal_CCF[0], goal_CCF[1], normed_velocity, steering_angle,
		            	real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
		            	real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
		            	real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
		            	real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
		            	real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
		            	real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
		            	],
		            	dtype=np.float32)

                    act = agent.choose_action(state)
                    # Translate action signals to steering signals
                    throttle_signal = act[0]
                    if throttle_signal >= 0:
                        v_ref_t = v_max*throttle_signal
                    else:
                        v_ref_t = -v_min*throttle_signal
                    steering_signal = act[1]
                    alpha_ref_t = alpha_max*steering_signal
                
                    # Run one cycle
                    halu_car.one_step_algorithm_2(alpha_ref=alpha_ref_t, v_ref=v_ref_t, dt=decision_dt)
                    trajectory.append(halu_car.position_center)
                    action_queue.append([v_ref_t, alpha_ref_t])
                    # Runtime
                    trajectory_run_time = time.time()
                    counter = 0
            ####################################
            # Do we need to update trajectory? #
            ####################################
            counter += 1
            # Use simple timer, or check if trajectory is expired
            #if counter == trajectory_length//2: # 2
            #    update_vision = True  
            if (time.time() - trajectory_run_time) > decision_dt*trajectory_length: # Time should be up
                update_vision = True
                print("Time out")
            elif (len(action_queue)) <= 1:
                update_vision = True
                print("Out of actions!")
            #elif  # check for big changes in vision / visual differences expected. requires storing all SCs from trajectory generation.

            #################################
            # Select the action, from queue #
            #################################
            v_ref, alpha_ref = action_queue.popleft()
            trajectory.popleft() # Also remove a point of the trajectory
            ##############
            # Visualize! #
            ##############
            times = np.int32(decision_dt/sim_dt)
            for _ in range(times):
                gfx.clear_canvas()
                gfx.draw_all_objects(objects+[car1]) 
                gfx.draw_one_object(viz_box, color=(255, 0, 0), width=4)

                gfx.draw_headings([car1], scale=True)
                gfx.draw_goal_state((goal_x, goal_y), threshold=10)
                for point in trajectory:
                    gfx.draw_goal_state((point[0], point[1]), width=2)

                #gfx.draw_centers(cars)
                #gfx.draw_static_circogram_data(real_circogram, car1, color=(0, 0, 255))
                #gfx.draw_static_circogram_data(halucinated_circogram, car1, color=(0, 255, 0))
                clock.tick(fps) # fps
                gfx.display_fps(clock.get_fps(), font_size=32, color="red", where=(0,0))
                gfx.update_display()

                # Run x number of simulation cycles.
                car1.one_step_algorithm_2(alpha_ref=alpha_ref, v_ref=v_ref, dt=sim_dt)

        else:
            exit()

def test_user_input():
    # Create a visualizer
    dt = 0.0333333333333333333333333333 #(1/30)
    gfx, objects, cars = map_lanes_single(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True, dt=dt)
    car1 = cars[0]

    # Spawn a driver
    alpha_max = 0.5
    v_max = 12 # 8
    v_min = -4 # -4
    agent = Agent(v_max, v_min, alpha_max)
    # Make it a limo!
    limo = Limo(vehicle=car1, driver=agent)

    # Sets certain parameters
    collision = False
    clock = pygame.time.Clock()
    fps = 1/dt

    while True:
        # Default steering parameters
        alpha_ref = 0
        v_ref = 0 

        if not collision:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    #exit()
                    pygame.quit()
                    sys.exit()
                # checking if keydown event happened or not
                #if event.type == pygame.KEYDOWN: # A key is pressed.
                #    if event.key == pygame.K_UP:
                #        print("UP")
                #        v_ref += v_max
                #    if event.key == pygame.K_DOWN:
                #        print("DOWN")
                #        v_ref += v_min
                #    if event.key == pygame.K_LEFT:
                #        print("LEFT")
                #        alpha_ref -= alpha_max
                #    if event.key == pygame.K_RIGHT:
                #        print("RIGHT")
                #        alpha_ref += alpha_max
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                #print("UP")
                v_ref += v_max
            if keys[pygame.K_DOWN]:
                #print("DOWN")
                v_ref += v_min
            if keys[pygame.K_LEFT]:
                #print("LEFT")
                alpha_ref -= alpha_max
            if keys[pygame.K_RIGHT]:
                #print("RIGHT")
                alpha_ref += alpha_max
                
            ##############
            # Visualize! #
            ##############   
            gfx.clear_canvas()
            gfx.draw_all_objects(objects) 
            
            # Generate circogram
            N = 36
            horizon = 1000
            #
            static_circogram = car1.static_circogram_2(N, objects[1:], horizon)
            #
            d1, d2, _, P2, _ = static_circogram
            collision = car1.collision_check(d1, d2)
            if collision:
                print("Collision!")
                continue

            gfx.draw_headings(cars, scale=True)
            gfx.draw_centers(cars)
            gfx.draw_static_circogram_data(static_circogram, car1)
            # Run one cycle
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)

            ##############
            # Vision box #
            ##############
            #sides = [[P2[-1], P2[0]]]
            #for i in range(len(P2)-1):
            #    sides.append([P2[i], P2[i+1]])
            #gfx.draw_sides(sides, color=(255, 0, 0), width=12)

            
        else:
            v_ref = 0
            alpha_ref = 0

        clock.tick(fps) # fps
        gfx.display_fps(clock.get_fps(), font_size=32, color="red", where=(0,0))
        gfx.update_display()
        
def test_visual_box():
    # Create a visualizer
    dt = 0.0333333333333333333333333333 #(1/30)
    gfx, objects, cars = map_lanes_single(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True, dt=dt)
    car1 = cars[0]

    # Spawn a driver
    alpha_max = 0.5
    v_max = 12 # 8
    v_min = -4 # -4
    agent = Agent(v_max, v_min, alpha_max)
    # Make it a limo!
    limo = Limo(vehicle=car1, driver=agent)

    # Sets certain parameters
    collision = False
    clock = pygame.time.Clock()
    fps = 1/dt

    while True:
        # Default steering parameters
        alpha_ref = 0
        v_ref = 0 

        if not collision:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    #exit()
                    pygame.quit()
                    sys.exit()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                #print("UP")
                v_ref += v_max
            if keys[pygame.K_DOWN]:
                #print("DOWN")
                v_ref += v_min
            if keys[pygame.K_LEFT]:
                #print("LEFT")
                alpha_ref -= alpha_max
            if keys[pygame.K_RIGHT]:
                #print("RIGHT")
                alpha_ref += alpha_max
                
            ##############
            # Visualize! #
            ##############   
            gfx.clear_canvas()
            gfx.draw_all_objects(objects) 
            
            # Generate circogram
            N = 36
            horizon = 1000
            #
            static_circogram = car1.static_circogram_2(N, objects[1:], horizon)
            #
            d1, d2, _, P2, _ = static_circogram
            collision = car1.collision_check(d1, d2)
            if collision:
                print("Collision!")
                continue

            gfx.draw_headings(cars, scale=True)
            gfx.draw_centers(cars)
            gfx.draw_static_circogram_data(static_circogram, car1)
            # Run one cycle
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)

            ##############
            # Vision box #
            ##############
            viz_box = Object(np.array([0, 0]), vertices=P2)
            gfx.draw_one_object(viz_box, color=(255, 0, 0), width=4)

            
        else:
            v_ref = 0
            alpha_ref = 0

        clock.tick(fps) # fps
        gfx.display_fps(clock.get_fps(), font_size=32, color="red", where=(0,0))
        gfx.update_display()

def test_visual_box_update_rule():
    # Create a visualizer
    dt = 0.0333333333333333333333333333 #(1/30)
    gfx, objects, cars = map_lanes_single(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True, dt=dt)
    car1 = cars[0]

    # Spawn a driver
    alpha_max = 0.5
    v_max = 12 # 8
    v_min = -4 # -4
    agent = Agent(v_max, v_min, alpha_max)
    # Make it a limo!
    limo = Limo(vehicle=car1, driver=agent)

    # Sets certain parameters
    collision = False
    clock = pygame.time.Clock()
    fps = 1/dt
    update_vision=True # need to make initial update
    goal_x, goal_y = (170, 95)
    while True:
        # Default steering parameters
        alpha_ref = 0
        v_ref = 0 
        

        if not collision:
            ##########
            # INPUTS #
            ##########

            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    #exit()
                    pygame.quit()
                    sys.exit()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                #print("UP")
                v_ref += v_max
            if keys[pygame.K_DOWN]:
                #print("DOWN")
                v_ref += v_min
            if keys[pygame.K_LEFT]:
                #print("LEFT")
                alpha_ref -= alpha_max
            if keys[pygame.K_RIGHT]:
                #print("RIGHT")
                alpha_ref += alpha_max
                
            
            # Generate circogram
            N = 36
            horizon = 1000
            real_circogram = car1.static_circogram_2(N, objects[1:], horizon)
            d1, d2, _, P2, _ = real_circogram
            collision = car1.collision_check(d1, d2)
            if collision:
                print("Collision!")
                continue
            ##############
            # Vision box #
            ##############
            if update_vision:
                viz_box = Object(np.array([0, 0]), vertices=P2)
                update_vision = False

            # Hallucinated vision.
            N = 36
            horizon = 1000
            halucinated_circogram = car1.static_circogram_2(N, [viz_box], horizon)
            _, d2_halu, _, _, _ = halucinated_circogram
            # 1
            diff = d2_halu - d2*0.80
            value = np.sum(diff < 0)
            if value:
                print("Outside is drastically different!")
                update_vision = True
            # 2
            diff = d2-(d2_halu*0.80) # Adding a bit of wiggle room
            value = np.sum(diff < 0) # if only one ray is "true" in the comparison
            if value:
                print("Outside gotten inside!")
                update_vision = True

            ##############
            # Visualize! #
            ##############   
            gfx.clear_canvas()
            gfx.draw_all_objects(objects) 
            gfx.draw_one_object(viz_box, color=(255, 0, 0), width=4)
            gfx.draw_goal_state((goal_x, goal_y), threshold=10)
            gfx.draw_headings(cars, scale=True)
            gfx.draw_centers(cars)
            gfx.draw_static_circogram_data(real_circogram, car1, color=(0, 0, 255))
            gfx.draw_static_circogram_data(halucinated_circogram, car1, color=(0, 255, 0))
            
            
            # Run one cycle
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)

            
            

            
        else:
            v_ref = 0
            alpha_ref = 0

        clock.tick(fps) # fps
        gfx.display_fps(clock.get_fps(), font_size=32, color="red", where=(0,0))
        gfx.update_display()

def test_pre_trained_MPC_agent():
    """
    # PARAMETERS:
    """   
    #sim_dt=0.25
    sim_dt = 0.1
    #sim_dt = 0.0333333333333333333333333333 #(1/30)
    decision_dt = 0.1
    env="v21"
    folder="DDPG/checkpoints/v21_2"
    v_max=20
    v_min=-4
    alpha_max=0.5
    tau_steering=0.5
    tau_throttle=0.5
     
    # Start a new agent
    agent = DDPG_Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    # Load the pre-trained model
    agent.load_models(Verbose=False)

    # Graphics
    MAP_DIMENSIONS = (1080, 1920)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7, map_img_path="graphics/test_map_2.png")
    # Environment
    env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200)
    env.reset()
    # vehicle
    car1 = env.vehicle
    objects = env.objects
    goal_x, goal_y = env.goal_x, env.goal_y

    # Sets certain parameters
    collision = False
    trajectory_length = 30
    clock = pygame.time.Clock()
    fps = 1/sim_dt
    update_vision=True # need to make initial update

    while True:
        # Default steering parameters
        alpha_ref = 0
        v_ref = 0 

        if not collision:
            ###############
            # USER INPUTS #
            ###############

            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    pygame.quit()
                    sys.exit()
        
            ###########################
            # Vision box & Trajectory #
            ###########################
            if update_vision:
                print("Update time!")
                # Generate circogram
                N = 40 # 36
                horizon = 1000
                real_circogram = car1.static_circogram_2(N, objects, horizon)
                d1, d2, _, P2, _ = real_circogram
                collision = car1.collision_check(d1, d2)
                if collision:
                    print("Collision!")
                    continue
                viz_box = Object(np.array([0, 0]), vertices=P2)
                update_vision = False

                # Generate the trajectory to follow
                halu_car = deepcopy(car1)
                trajectory = deque()
                action_queue = deque()
                for _ in range(trajectory_length):
                    # Need to multiply by actual direction, as if not - it will always be a positive value.
                    try:
                        normed_velocity = np.linalg.norm(halu_car.X[2:])*halu_car.actual_direction
                    except:
                        normed_velocity = 0
                    steering_angle = halu_car.alpha

                    # Update goal poses in CCF (as CCF's origin has moved)
                    goal_CCF = halu_car.WCFtoCCF(np.array([goal_x, goal_y]))

                    # Get the "new" static circogram
                    N = 36
                    horizon = 1000
                    SC = halu_car.static_circogram_2(N, [viz_box], horizon)
                    d1_, d2_, _, _, _ = SC
                    real_distances = d2_ - d1_

                    # Generate new state
                    state = np.array([goal_CCF[0], goal_CCF[1], normed_velocity, steering_angle,
		            	real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
		            	real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
		            	real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
		            	real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
		            	real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
		            	real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
		            	],
		            	dtype=np.float32)

                    act = agent.choose_action(state)
                    # Translate action signals to steering signals
                    throttle_signal = act[0]
                    if throttle_signal >= 0:
                        v_ref_t = v_max*throttle_signal
                    else:
                        v_ref_t = -v_min*throttle_signal
                    steering_signal = act[1]
                    alpha_ref_t = alpha_max*steering_signal
                
                    # Run one cycle
                    halu_car.one_step_algorithm_2(alpha_ref=alpha_ref_t, v_ref=v_ref_t, dt=decision_dt)
                    trajectory.append(halu_car.position_center)
                    action_queue.append([v_ref_t, alpha_ref_t])
                    # Runtime
                    trajectory_run_time = time.time()
                    counter = 0
            ####################################
            # Do we need to update trajectory? #
            ####################################
            counter += 1
            # Use simple timer, or check if trajectory is expired
            #if counter == trajectory_length//2: # 2
            #    update_vision = True  
            if (time.time() - trajectory_run_time) > decision_dt*trajectory_length: # Time should be up
                update_vision = True
                print("Time out")
            elif (len(action_queue)) <= 1:
                update_vision = True
                print("Out of actions!")
            #elif  # check for big changes in vision / visual differences expected. requires storing all SCs from trajectory generation.

            #################################
            # Select the action, from queue #
            #################################
            v_ref, alpha_ref = action_queue.popleft()
            trajectory.popleft() # Also remove a point of the trajectory
            ##############
            # Visualize! #
            ##############
            times = np.int32(decision_dt/sim_dt)
            for _ in range(times):
                gfx.clear_canvas()
                gfx.draw_all_objects(objects+[car1]) 
                gfx.draw_one_object(viz_box, color=(255, 0, 0), width=4)

                gfx.draw_headings([car1], scale=True)
                gfx.draw_goal_state((goal_x, goal_y), threshold=10)
                for point in trajectory:
                    gfx.draw_goal_state((point[0], point[1]), width=2)

                #gfx.draw_centers(cars)
                #gfx.draw_static_circogram_data(real_circogram, car1, color=(0, 0, 255))
                #gfx.draw_static_circogram_data(halucinated_circogram, car1, color=(0, 255, 0))
                clock.tick(fps) # fps
                gfx.display_fps(clock.get_fps(), font_size=32, color="red", where=(0,0))
                gfx.update_display()

                # Run x number of simulation cycles.
                car1.one_step_algorithm_2(alpha_ref=alpha_ref, v_ref=v_ref, dt=sim_dt)

        else:
            exit()

def test_pre_trained_with_update_vbox():
    """
    # PARAMETERS:
    """   
    #sim_dt=0.25
    sim_dt = 0.1
    #sim_dt = 0.0333333333333333333333333333 #(1/30)
    decision_dt = 0.1
    env="v21"
    folder="DDPG/checkpoints/v21_2"
    v_max=20
    v_min=-4
    alpha_max=0.5
    tau_steering=0.5
    tau_throttle=0.5
     
    # Start a new agent
    agent = DDPG_Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    # Load the pre-trained model
    agent.load_models(Verbose=False)

    # Graphics
    MAP_DIMENSIONS = (1080, 1920)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7, map_img_path="graphics/test_map_2.png")
    # Environment
    env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200)
    env.reset()
    # vehicle
    car1 = env.vehicle
    objects = env.objects
    goal_x, goal_y = env.goal_x, env.goal_y

    # Sets certain parameters
    collision = False
    trajectory_length = 30
    clock = pygame.time.Clock()
    fps = 1/sim_dt
    update_vision=True # need to make initial update

    while True:
        # Default steering parameters
        alpha_ref = 0
        v_ref = 0 

        if not collision:
            ###############
            # USER INPUTS #
            ###############

            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    pygame.quit()
                    sys.exit()
        
            ###########################
            # Vision box & Trajectory #
            ###########################
            # Generate circogram (each decision dt!)
            N = 36
            horizon = 1000
            real_circogram = car1.static_circogram_2(N, objects, horizon)
            d1, d2, _, P2, _ = real_circogram
            collision = car1.collision_check(d1, d2)
            if collision:
                print("Collision!")
                continue

            if update_vision:
                print("Update time!")
                viz_box = Object(np.array([0, 0]), vertices=P2)
                update_vision = False

                # Generate the trajectory to follow
                halu_car = deepcopy(car1)
                trajectory = deque()
                action_queue = deque()
                halu_d2s = deque()
                for _ in range(trajectory_length):
                    # Need to multiply by actual direction, as if not - it will always be a positive value.
                    try:
                        normed_velocity = np.linalg.norm(halu_car.X[2:])*halu_car.actual_direction
                    except:
                        normed_velocity = 0
                    steering_angle = halu_car.alpha

                    # Update goal poses in CCF (as CCF's origin has moved)
                    goal_CCF = halu_car.WCFtoCCF(np.array([goal_x, goal_y]))

                    # Get the "new" static circogram
                    N = 36
                    horizon = 1000
                    SC = halu_car.static_circogram_2(N, [viz_box], horizon)
                    d1_, d2_, _, _, _ = SC
                    real_distances = d2_ - d1_
                    halu_d2s.append(d2_)
                    # Generate new state
                    state = np.array([goal_CCF[0], goal_CCF[1], normed_velocity, steering_angle,
		            	real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
		            	real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
		            	real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
		            	real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
		            	real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
		            	real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
		            	],
		            	dtype=np.float32)

                    act = agent.choose_action(state)
                    # Translate action signals to steering signals
                    throttle_signal = act[0]
                    if throttle_signal >= 0:
                        v_ref_t = v_max*throttle_signal
                    else:
                        v_ref_t = -v_min*throttle_signal
                    steering_signal = act[1]
                    alpha_ref_t = alpha_max*steering_signal
                
                    # Run one cycle
                    halu_car.one_step_algorithm_2(alpha_ref=alpha_ref_t, v_ref=v_ref_t, dt=decision_dt)
                    trajectory.append(halu_car.position_center)
                    action_queue.append([v_ref_t, alpha_ref_t])
                    # Runtime
                    trajectory_run_time = time.time()
                    counter = 0
            ####################################
            # Do we need to update trajectory? #
            ####################################
            counter += 1
            # Use simple timer, or check if trajectory is expired
            #if counter == trajectory_length//2: # 2
            #    update_vision = True  
            if (time.time() - trajectory_run_time) > decision_dt*trajectory_length: # Time should be up
                update_vision = True
                print("Time out")
            elif (len(action_queue)) <= 1:
                update_vision = True
                print("Out of actions!")
            # Retrieve corresponding SC from trajectory
            d2_halu = halu_d2s.popleft()
            # 1 Outside is drastically different
            diff = d2_halu - d2*0.80
            value = np.sum(diff < 0)
            if value:
                update_vision = True
            # 2 Outside gotten inside
            diff = d2-(d2_halu*0.80) # Adding a bit of wiggle room
            value = np.sum(diff < 0) # if only one ray is "true" in the comparison
            if value:
                update_vision = True

            #################################
            # Select the action, from queue #
            #################################
            v_ref, alpha_ref = action_queue.popleft()
            trajectory.popleft() # Also remove a point of the trajectory, for visuals.
            ##############
            # Visualize! #
            ##############
            times = np.int32(decision_dt/sim_dt)
            for _ in range(times):
                gfx.clear_canvas()
                gfx.draw_all_objects(objects+[car1]) 
                gfx.draw_one_object(viz_box, color=(255, 0, 0), width=4)

                gfx.draw_headings([car1], scale=True)
                gfx.draw_goal_state((goal_x, goal_y), threshold=10)
                for point in trajectory:
                    gfx.draw_goal_state((point[0], point[1]), width=2)

                #gfx.draw_centers(cars)
                #gfx.draw_static_circogram_data(real_circogram, car1, color=(0, 0, 255))
                #gfx.draw_static_circogram_data(halucinated_circogram, car1, color=(0, 255, 0))
                clock.tick(fps) # fps
                gfx.display_fps(clock.get_fps(), font_size=32, color="red", where=(0,0))
                gfx.update_display()

                # Run x number of simulation cycles.
                car1.one_step_algorithm_2(alpha_ref=alpha_ref, v_ref=v_ref, dt=sim_dt)

        else:
            exit()
    
def test_milkman_full():
    """
    # PARAMETERS:
    """   
    #sim_dt=0.25
    sim_dt = 0.1
    #sim_dt = 0.0333333333333333333333333333 #(1/30)
    decision_dt = 0.1
    env="v21"
    folder="DDPG/checkpoints/v21_2"
    v_max=20
    v_min=-4
    alpha_max=0.5
    tau_steering=0.5
    tau_throttle=0.5
     
    # Start a new agent
    agent = DDPG_Agent(alpha=0.000025, beta=0.00025, input_dims=[40], tau=0.1, env=env, 
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    # Load the pre-trained model
    agent.load_models(Verbose=False)

    # Graphics
    MAP_DIMENSIONS = (1080, 1920)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=7, map_img_path="graphics/test_map_2.png")
    # Environment
    env = ClosedField_v21(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200)
    env.reset()
    # vehicle
    car1 = env.vehicle
    objects = env.objects
    goal_x, goal_y = env.goal_x, env.goal_y

    # Sets certain parameters
    collision = False
    trajectory_length = 30
    clock = pygame.time.Clock()
    fps = 1/sim_dt
    update_vision=True # need to make initial update

    while True:
        # Default steering parameters
        alpha_ref = 0
        v_ref = 0 

        if not collision:
            ###############
            # USER INPUTS #
            ###############

            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    pygame.quit()
                    sys.exit()
        
            ###########################
            # Vision box & Trajectory #
            ###########################
            # Generate circogram (each decision dt!)
            # NOTE this is done each decision_dt only because we need to dynamically look for updates to vision box.
            # NOTE so in case of static environment, this only needs to be done after we are run out of trajectory moves.
            # NOTE, also adds the benefit of allowing collision check each decision dt!
            N = 36
            horizon = 1000
            real_circogram = car1.static_circogram_2(N, objects, horizon)
            d1, d2, _, P2, _ = real_circogram
            collision = car1.collision_check(d1, d2)
            if collision:
                print("Collision!")
                continue

            if update_vision:
                #N = 36
                #horizon = 1000
                #real_circogram = car1.static_circogram_2(N, objects, horizon)
                #d1, d2, _, P2, _ = real_circogram
                #collision = car1.collision_check(d1, d2)
                #if collision:
                #    print("Collision!")
                #    continue

                #print("Update time!")
                viz_box = Object(np.array([0, 0]), vertices=P2)
                update_vision = False

                # Generate the trajectory to follow
                halu_car = deepcopy(car1)
                trajectory = deque()
                action_queue = deque()
                halu_d2s = deque()
                for _ in range(trajectory_length):
                    # Need to multiply by actual direction, as if not - it will always be a positive value.
                    try:
                        normed_velocity = np.linalg.norm(halu_car.X[2:])*halu_car.actual_direction
                    except:
                        normed_velocity = 0
                    steering_angle = halu_car.alpha

                    # Update goal poses in CCF (as CCF's origin has moved)
                    goal_CCF = halu_car.WCFtoCCF(np.array([goal_x, goal_y]))

                    # Get the "new" static circogram
                    N = 36
                    horizon = 1000
                    SC = halu_car.static_circogram_2(N, [viz_box], horizon)
                    d1_, d2_, _, _, _ = SC
                    real_distances = d2_ - d1_
                    halu_d2s.append(d2_)
                    # Generate new state
                    state = np.array([goal_CCF[0], goal_CCF[1], normed_velocity, steering_angle,
		            	real_distances[0], real_distances[1], real_distances[2], real_distances[3], real_distances[4], real_distances[5],
		            	real_distances[6], real_distances[7], real_distances[8], real_distances[9], real_distances[10], real_distances[11],
		            	real_distances[12], real_distances[13], real_distances[14], real_distances[15], real_distances[16], real_distances[17],
		            	real_distances[18], real_distances[19], real_distances[20], real_distances[21], real_distances[22], real_distances[23],
		            	real_distances[24], real_distances[25], real_distances[26], real_distances[27], real_distances[28], real_distances[29],
		            	real_distances[30], real_distances[31], real_distances[32], real_distances[33], real_distances[34], real_distances[35]
		            	],
		            	dtype=np.float32)

                    act = agent.choose_action(state)
                    # Translate action signals to steering signals
                    throttle_signal = act[0]
                    if throttle_signal >= 0:
                        v_ref_t = v_max*throttle_signal
                    else:
                        v_ref_t = -v_min*throttle_signal
                    steering_signal = act[1]
                    alpha_ref_t = alpha_max*steering_signal
                
                    # Run one cycle
                    halu_car.one_step_algorithm_2(alpha_ref=alpha_ref_t, v_ref=v_ref_t, dt=decision_dt)
                    trajectory.append(halu_car.position_center)
                    action_queue.append([v_ref_t, alpha_ref_t])
                    # Runtime
                    trajectory_run_time = time.time()
                    counter = 0
            ####################################
            # Do we need to update trajectory? #
            ####################################
            counter += 1
            # Use simple timer, or check if trajectory is expired
            #if counter == trajectory_length//2: # 2
            #    update_vision = True  
            if (time.time() - trajectory_run_time) > decision_dt*trajectory_length: # Time should be up
                update_vision = True
                #print("Time out")
            elif (len(action_queue)) <= 1:
                update_vision = True
                #print("Out of actions!")
               
            # Retrieve corresponding SC from trajectory
            d2_halu = halu_d2s.popleft()
            # 1 Outside is drastically different
            diff = d2_halu - d2*0.50
            value = np.sum(diff < 0)
            if value:
                update_vision = True
            # 2 Outside gotten inside
            diff = d2-(d2_halu*0.75) # Adding a bit of wiggle room
            value = np.sum(diff < 0) # if only one ray is "true" in the comparison
            if value:
                update_vision = True
            
            #################################
            # Select the action, from queue #
            #################################
            v_ref, alpha_ref = action_queue.popleft()
            trajectory.popleft() # Also remove a point of the trajectory, for visuals.
            ##############
            # Visualize! #
            ##############
            times = np.int32(decision_dt/sim_dt)
            for _ in range(times): # if sim_dt == decision_dt, this only runs once.
                gfx.clear_canvas()
                gfx.draw_all_objects([car1]+objects) # +objects
                gfx.draw_one_object(viz_box, color=(255, 0, 0), width=4)

                gfx.draw_headings([car1], scale=True)
                gfx.draw_goal_state((goal_x, goal_y), threshold=10)
                for point in trajectory:
                    gfx.draw_goal_state((point[0], point[1]), width=2)

                #gfx.draw_centers(cars)
                #gfx.draw_static_circogram_data(real_circogram, car1, color=(0, 0, 255))
                #gfx.draw_static_circogram_data(halucinated_circogram, car1, color=(0, 255, 0))
                clock.tick(fps) # fps
                gfx.display_fps(clock.get_fps(), font_size=32, color="red", where=(0,0))
                gfx.update_display()

                # Run x number of simulation cycles.
                car1.one_step_algorithm_2(alpha_ref=alpha_ref, v_ref=v_ref, dt=sim_dt)

            #################
            # Goal reached? #
            #################
            # Calculate goal distance
            carx, cary = car1.position_center
            dist = np.linalg.norm(np.array([carx, cary]) - np.array([goal_x, goal_y]))
            # Goal is reached!
            if dist < 10:
                #env.generate_new_goal_state()
                #goal_x, goal_y = env.goal_x, env.goal_y
                return

        else:
            exit()
    

if __name__ == "__main__":
    #test_user_input()
    #test_visual_box()
    #test_visual_box_update_rule()
    """
    What we learned: 
        - The model might not be numerically stable; as we get different results for running one step of 0.5, or 10 steps of 0.05 seconds...
        - if sim_dt == decision_dt, we are good!
    """
    #test_pre_trained_MPC_agent() # # vbox does not update
    #test_pre_trained_with_update_vbox() 
    while True:
        test_milkman_full() # Buggy as hell currently...
    
    #test_numerical_stability()