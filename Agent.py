import numpy as np
import matplotlib.pyplot as plt
from alpha_star import *

class Agent:

    def __init__(self, v_max : float = 10, v_min : float = 0, alpha_max : float = 1.2, var_vel : float = 0.5, var_alpha : float = 0.2) -> None:
        self.v_max = v_max
        self.v_min = v_min
        self.alpha_max = alpha_max
        self.var_vel = var_vel
        self.var_alpha = var_alpha
        self.state = 0

        # needed for brownian_DC_action
        self.risky_distance = None
        self.risky_index = None
        # Debugging
        self.prev_dist = None
        self.new_dists = []
        self.states = []
        self.v_refs = []
        self.wait_counter = 0

        # NEW DC tech
        self.prev_DC_risks = None

    def long_term_plan(map, current_state : tuple, goal_state : tuple, scale_percent=5, dilation=10) -> np.array:
        # 1 Prepare map for planning
        inv_map = (255-np.copy(map)) # invert map, so that values represent cost
        dilated = cv2.dilate(inv_map, None, iterations=dilation)
        width = int(dilated.shape[1] * scale_percent / 100)
        height = int(dilated.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(dilated, dim, interpolation = cv2.INTER_AREA)
        # 2 path planning om resized, dilated map
        small_path = a_star(resized, current_state, goal_state)
        # 3 Resize path to original size
        path_resized = np.array(small_path)*100/scale_percent # Scale back up
        return path_resized

    @staticmethod
    def point_distance(point1, point2):
        """ Euclidean distance between two points """
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def choose_path_point(path, current_pos, max_dist=20):
        chosen_index = None
        for point in path:
            ...

    @staticmethod
    def approach_a_point(current_pos : np.array, other_pos : np.array) -> tuple:
        v_ref = 0
        alpha_ref = 0
        return v_ref, alpha_ref
    
    def brownian_action(self, v_ref, alpha_ref, r_factor=0.02):
        """
        Random walk algorithm. Takes current steering input, and changes it
        """
        # Random driver:
        if np.random.choice(a=[0,1], p=[1-r_factor, r_factor]): # Random motion has occured
            choices = np.random.randint(0, 2, 2) # Increase or decrease?

            if choices[0]:
                v_rand = -self.var_vel
            else: 
                v_rand = self.var_vel

            if choices[1]:
                alpha_rand = -self.var_alpha
            else:
                alpha_rand = self.var_alpha
            # Add reflection
            if (v_ref+v_rand) > self.v_max or (v_ref+v_rand) <= self.v_min:
                v_ref = v_ref - v_rand
            else:
                v_ref = v_ref + v_rand
            
            if (alpha_ref + alpha_rand) > self.alpha_max or (alpha_ref + alpha_rand) <= -self.alpha_max:
                alpha_ref = alpha_ref - alpha_rand
            else:
                alpha_ref = alpha_ref + alpha_rand
        return v_ref, alpha_ref

    def old_brownian_dc_action(self, dynamic_circogram, static_circogram, v_ref, alpha_ref, risk_threshold=0.8, stop_threshold = 5, r_factor=0.02):
        """ Collision avoidance protocol that kicks into action,
            whenever the risk from dynamic circogram (dc) is too high.
        """
        # State machine:
        # {"brownian": 0, "dc_avoid" : 1, "waiting" : 2}
        # 
        angles = dynamic_circogram[:,0]
        risks = dynamic_circogram[:,2]
        d1, d2, _, _, _ = static_circogram
        max_index = np.argmax(risks)
        ####################################################
        # STATE STOP/Waiting
        if  (self.state == 2): # *2!
            # State remains
            v_ref = 0
            new_distance = np.abs(d2[self.risky_index] - d1[self.risky_index])
            # State change?
            if new_distance > self.risky_distance*2: # The only way out of state 2
                self.wait_counter +=1
                if self.wait_counter > 30: # To avoid nocie slips
                    self.state = 0 # Brownian again, base.
                    alpha_ref = self.alpha_max # Turn sharp right
                    v_ref = self.v_max/4 # Calm start
            else:
                self.wait_counter = 0
            
        ####################################################
        # STATE DC avoidance
        elif (self.state == 1):
            # State change?
            if (risks[max_index] >= risk_threshold*stop_threshold):
                # Stop!
                v_ref = 0
                # Parameters 
                self.state = 2
                self.risky_distance = np.abs(d2[max_index] - d1[max_index]) # distance along risky ray
                self.risky_index = max_index
                print("CAR IS STOPPING...")
            elif risks[max_index] < risk_threshold:
                # Return to brownian
                self.state = 0
                alpha_ref = 0
                #v_ref = self.v_max/2 # keep speed as it is!
            else:
                # Default state 1 behaviour
                v_ref, alpha_ref = self.dc_avoidance_action(angles, risks, v_ref, alpha_ref)
        ####################################################
        # STATE brownian
        elif (self.state==0):
            # State change?
            if (risks[max_index] >= risk_threshold*stop_threshold):
                # Stop!
                v_ref = 0
                # Parameters 
                self.state = 2
                self.risky_distance = np.abs(d2[max_index] - d1[max_index]) # distance along risky ray
                self.risky_index = max_index
                print("CAR IS STOPPING...")
            elif risks[max_index] >= risk_threshold:
                # Avoid collision!
                self.state = 1
            else:
                # Default state 0 behaviour
                v_ref, alpha_ref = self.brownian_action(v_ref=v_ref, alpha_ref=alpha_ref, r_factor=r_factor)
        
        else:
            raise Exception("Not a valid state: "+str(self.state))
        ####################################################
        return v_ref, alpha_ref

    def new_brownian_dc_action(self, dynamic_circogram, v_ref, alpha_ref, risk_threshold=0.8, stop_threshold = 5, r_factor=0.02):
        """ Collision avoidance protocol that kicks into action,
            whenever the risk from dynamic circogram (dc) is too high.

            Also has reversing!
        """
        # State machine:
        # {"brownian": 0, "dc_avoid" : 1, "waiting" : 2}
        # 
        angles = dynamic_circogram[:,0]
        risks = dynamic_circogram[:,2]
        max_index = np.argmax(risks)
        ####################################################
        if (self.state == 2):
            self.wait_counter += 1
            if self.wait_counter > 10: # About 1 second, depending on dt...
                self.state = 1

        ####################################################
        # STATE DC avoidance
        elif (self.state == 1):
            # State change?
            if (risks[max_index] >= risk_threshold*stop_threshold):
                # Reverse!
                #self.stopping_speed = v_ref # Save what speed was before stopping
                v_ref = -(v_ref/np.abs(v_ref))*self.v_max/3
                alpha_ref = self.alpha_max # Sharp turn!
                # Parameters 
                self.state = 2 # Never go into stopping state; just start reversing
                self.wait_counter = 0
            elif risks[max_index] < risk_threshold:
                # Return to brownian
                self.state = 0
                alpha_ref = 0
                #v_ref = self.v_max/2 # keep speed as it is!
            else:
                # Default state 1 behaviour
                v_ref, alpha_ref = self.dc_avoidance_action(angles, risks, v_ref, alpha_ref)
        ####################################################
        # STATE brownian
        elif (self.state==0):
            # State change?
            if (risks[max_index] >= risk_threshold*stop_threshold):
                # Reverse!
                #self.stopping_speed = v_ref # Save what speed was before stopping
                v_ref = -(v_ref/np.abs(v_ref))*self.v_max/3
                alpha_ref = self.alpha_max # Sharp turn!
                # Parameters 
                self.state = 2 # Never go into stopping state; just start reversing
                self.wait_counter = 0
            elif risks[max_index] >= risk_threshold:
                # Avoid collision!
                self.state = 1
            else:
                # Default state 0 behaviour
                v_ref, alpha_ref = self.brownian_action(v_ref=v_ref, alpha_ref=alpha_ref, r_factor=r_factor)
        
        else:
            raise Exception("Not a valid state: "+str(self.state))
        ####################################################
        return v_ref, alpha_ref

    def experimental_driving_action(self, dynamic_circogram, v_ref, alpha_ref, risk_threshold=0.8, stop_threshold = 5, r_factor=0.02):
        """ Collision avoidance, and keeping realaively high speeds
        """
        # State machine:
        # {"brownian": 0, "dc_avoid" : 1, "waiting" : 2}
        # 
        angles = dynamic_circogram[:,0]
        risks = dynamic_circogram[:,2]
        max_index = np.argmax(risks)
        ####################################################
        # Stop and reverse
        if (self.state == 2):
            self.wait_counter += 1
            if self.wait_counter > 10: # About 2 second, depending on dt...
                self.state = 0
                # Reverse the sign of old_v_ref.
                if self.old_v_ref>=0: # If we are now reversing, we tak sharp turn
                    v_ref = self.v_min/3
                    # 50% / 50% of what turn to take
                    num = np.random.choice([0, 1])
                    if num:
                        alpha_ref = self.alpha_max # Sharp turn!
                    else:
                        alpha_ref = -self.alpha_max
                else: # Forward now
                    v_ref = self.v_max/3
                    alpha_ref = 0

        ####################################################
        # STATE DC avoidance
        elif (self.state == 1):
            # State change?
            if (risks[max_index] >= risk_threshold*stop_threshold):
                # STOP
                self.old_v_ref = v_ref # keep sign
                v_ref = 0
                # Parameters 
                self.state = 2 # Never go into stopping state; just start reversing
                self.wait_counter = 0
            elif risks[max_index] < risk_threshold:
                # Return to brownian
                self.state = 0
                alpha_ref = 0
                #v_ref = self.v_max/2 # keep speed as it is!
            else:
                # Default state 1 behaviour
                v_ref, alpha_ref = self.dc_avoidance_action(angles, risks, v_ref, alpha_ref)
        ####################################################
        # STATE brownian
        elif (self.state==0):
            # State change?
            if (risks[max_index] >= risk_threshold*stop_threshold):
                # STOP
                self.old_v_ref = v_ref # keep sign
                v_ref = 0
                # Parameters 
                self.state = 2 # Never go into stopping state; just start reversing
                self.wait_counter = 0
            elif risks[max_index] >= risk_threshold:
                # Avoid collision!
                self.state = 1
            else:
                # Default state 0 behaviour
                v_ref, alpha_ref = self.brownian_action(v_ref=v_ref, alpha_ref=alpha_ref, r_factor=r_factor)
        
        else:
            raise Exception("Not a valid state: "+str(self.state))
        ####################################################
        return v_ref, alpha_ref

    def dc_avoidance_action(self, angles, risks, v_ref, alpha_ref):
        """
        Take a single action to avoid colliding, after detecting too high risks.
        """
        # TODO: Sometimes the max risk is not in same direction as car is driving.
            # This should not really matter; only risks in driving direction really matters...
            # This is fixed in dynamic circogram generation :)

        max_index = np.argmax(risks)
        max_risk = risks[max_index] # Risks are primarily in the region 0->1 (sometimes (above!))
        max_angle = angles[max_index]

        # 1 The turning angle:
        if v_ref>= 0: #Accelerating forward
            if np.pi >= max_angle >= np.pi/2: # left turn needed
                #print("TURN LEFT!")
                # The closer the risk is to straight in front, the sharper the turn
                #alpha_ref = -self.alpha_max/1.2
                alpha_ref = -self.alpha_max * (1-(max_angle-np.pi/2)/(np.pi))
                alpha_ref *= 0.7


            elif np.pi/2 > max_angle >= 0: # Right turn needed (max_angle < pi/2)
                #print("TURN RIGHT!")
                # The closer the risk is to straight in front, the sharper the turn
                alpha_ref = self.alpha_max * (0.5+(max_angle)/(np.pi))
                alpha_ref *= 0.7

            else:
                # That is weird... Max angle should be somewhere in driving direction.
                pass # Weird behaviour... # see TODO

        else: # Accelerating backwards!
            if 3*np.pi/2 >= max_angle >= np.pi: # left turn needed
                #print("TURN LEFT!")
                # The closer the risk is to straight in front, the sharper the turn
                #alpha_ref = -self.alpha_max/1.2
                alpha_ref = -self.alpha_max/1.2

            elif 2*np.pi >= max_angle >= 3*np.pi/2: # Right turn needed (max_angle < pi/2)
                #print("TURN RIGHT!")
                # The closer the risk is to straight in front, the sharper the turn
                alpha_ref = self.alpha_max/1.2

            else:
                # That is weird... Max angle should be somewhere in driving direction.
                pass # Weird behaviour... # see TODO


        # 2 v_ref: The higher the risk, the **slower** the driving.
        if v_ref >=0: # Forward
            v_ref = self.v_max*(1/(1+max_risk)) 
        else:
            v_ref = self.v_min*(1/(1+max_risk))
        return v_ref, alpha_ref

    def determined_dc_avoidance_action(self, angles, risks, v_ref, alpha_ref, stop_threshold):
        """
        Take a single action to avoid colliding, after detecting too high risks.
        - Prefers right hand turns
        - Smoother turns
        """
        max_index = np.argmax(risks)
        max_risk = risks[max_index] # Risks are primarily in the region 0->1 (sometimes (above!))
        max_angle = angles[max_index]

        # 1 The turning angle:
        if v_ref>= 0: #Accelerating forward
            # NOTE prefers the right hand turn.
            if np.pi >= max_angle >= np.pi/2+0.1: # left turn needed
                #print("TURN LEFT!")
                # The closer the risk is to straight in front, the sharper the turn
                alpha_ref = -self.alpha_max
                #alpha_ref = -self.alpha_max * (1-(max_angle-np.pi/2)/(np.pi))
                #alpha_ref *= 1.1


            elif np.pi/2+0.1 > max_angle >= 0: # Right turn needed (max_angle < pi/2)
                #print("TURN RIGHT!")
                # The closer the risk is to straight in front, the sharper the turn
                alpha_ref = self.alpha_max
                #alpha_ref = self.alpha_max * (0.5+(max_angle)/(np.pi))
                #alpha_ref *= 1.1

            else:
                # That is weird... Max angle should be somewhere in driving direction.
                pass # Weird behaviour... # see TODO

        else: # Accelerating backwards!
            if 3*np.pi/2 >= max_angle >= np.pi: # left turn needed
                # NOTE turns towards the obstacle.
                alpha_ref = -self.alpha_max

            elif 2*np.pi >= max_angle >= 3*np.pi/2: # Right turn needed
                # NOTE turns towards the obstacle.
                alpha_ref = self.alpha_max

            else:
                # That is weird... Max angle should be somewhere in driving direction.
                pass # Weird behaviour... # see TODO


        # 2 v_ref: The higher the risk, the **slower** the driving.
        if v_ref >=0: # Forward
            #v_ref = self.v_max*(1/(1+max_risk))
            v_ref = self.v_max*(1-max_risk/(stop_threshold))
        else:
            v_ref = self.v_min*(1/(1+max_risk)) # Keep backing
            #v_ref = self.v_min*(1-max_risk/stop_threshold)
        return v_ref, alpha_ref

    def determined_driver(self, dynamic_circogram, static_circogram, v_ref, alpha_ref, risk_threshold=0.8, stop_threshold = 2, dist_wait=10, verbose=False):
        """ Collision avoidance, and keeping realaively high speeds
        """
        # State machine:
        # {"Determined": 0, "dc_avoid" : 1, "stop, reverse" : 2}
        # 
        d1, d2, _, _, _ = static_circogram
        angles = dynamic_circogram[:,0]
        risks = dynamic_circogram[:,2]
        max_index = np.argmax(risks)
        ####################################################
        # Stop and reverse
        # Reverse until: path is cleared OR backing risk is too high again
        if (self.state == 2):
            # NOTE we stop to make sure we are stationary
            # TODO: add a 50% of waiting a bit before proceeding?
            #if self.stop_counter<20:
            #    v_ref = 0
            #    alpha_ref = 0
            #    self.stop_counter += 1
            #    return v_ref, alpha_ref

            # TODO: problem when car is actually stuck - will move forward, as risk == 0, then see danger, stop - but slide into the obstacle... :(

            # State remains
            v_ref = self.v_min
            if risks[max_index] >= risk_threshold:
                # Drive towards angle
                v_ref, alpha_ref = self.determined_dc_avoidance_action(angles, risks, self.v_min, alpha_ref, stop_threshold)
            else:
                alpha_ref = self.backing_angle

            # State change?
            new_distance = np.abs(d2[self.risky_index] - d1[self.risky_index])
            if new_distance > self.risky_distance*2: # One way out of state 2
                self.dist_counter +=1
                if self.dist_counter > dist_wait: # To be sure
                    v_ref = 0
                    self.state = 1
                    if verbose:
                        print("Risky dist was cleared!")    
            else:
                self.dist_counter = 0

            if risks[max_index] > stop_threshold: # Reverse risk is greater again.
                v_ref = 0 # Stop again
                self.state = 1
                if verbose:
                    print("Reversing was more dangerous!")

        ####################################################
        # STATE DC avoidance
        elif (self.state == 1):
            # State change?
            if (risks[max_index] >= stop_threshold):
                # STOP
                v_ref = 0
                self.state = 2
                self.risky_distance = np.abs(d2[max_index] - d1[max_index]) # distance along risky ray
                self.risky_index = max_index
                self.dist_counter = 0
                ## 50% / 50% of what turn to stop and wait or not
                #num = np.random.choice([0, 1])
                #if num==0:
                #    self.stop_counter = 30 # Done waiting
                #elif num==1:
                #    self.stop_counter = 0 # Wait
                
                # 50% / 50% of what turn to take
                num = np.random.choice([0, 1])
                if num==0:
                    self.backing_angle = self.alpha_max/1.5 
                elif num==1:
                    self.backing_angle = -self.alpha_max/1.5
                else:
                    self.backing_angle = 0 #Straight
                
                if verbose:
                    print("CAR IS STOPPING...")
            elif risks[max_index] < risk_threshold:
                # Return to determined
                self.state = 0
                alpha_ref = 0
                v_ref = self.v_max
            else:
                # Default state 1 behaviour: prefers rht
                v_ref, alpha_ref = self.determined_dc_avoidance_action(angles, risks, v_ref, alpha_ref, stop_threshold)
        ####################################################
        # STATE Determined
        elif (self.state==0):
            # State change?
            if risks[max_index] >= risk_threshold:
                # Avoid collision!!
                self.state = 1
            else:
                # Default state 0 behaviour
                v_ref = self.v_max
                alpha_ref = 0
        else:
            raise Exception("Not a valid state: "+str(self.state))
        ####################################################
        return v_ref, alpha_ref

    def determined_driver_new_DC(self, new_DC, static_circogram, v_ref, alpha_ref, risk_threshold=0.8, stop_threshold = 2, dist_wait=10, verbose=False):
            """ Collision avoidance, and keeping realaively high speeds
            """
            # State machine:
            # {"Determined": 0, "dc_avoid" : 1, "stop, reverse" : 2}
            d1, d2, _, _, angles = static_circogram
            new_risks = new_DC[:,2]
            #
            #if  (self.prev_DC_risks is None) or self.state == 2:
            #    self.prev_DC_risks = new_risks
            #    risks = new_risks
            #    print("BOOM!")
            #else:
            #    differentiated_risks = (new_risks - self.prev_DC_risks)/0.1
            #    risks = differentiated_risks
            try:
                differentiated_risks = (new_risks - self.prev_DC_risks)/0.1
            except TypeError:
                differentiated_risks = new_risks
            self.prev_DC_risks = new_risks
            risks = differentiated_risks

            max_index = np.argmax(risks)
            ####################################################
            # Stop and reverse
            # Reverse until: path is cleared OR backing risk is too high again
            if (self.state == 2):
                # State remains
                v_ref = self.v_min
                if risks[max_index] >= risk_threshold:
                    # Drive towards angle
                    v_ref, alpha_ref = self.determined_dc_avoidance_action(angles, risks, self.v_min, alpha_ref)
                else:
                    alpha_ref = self.backing_angle

                # State change?
                new_distance = np.abs(d2[self.risky_index] - d1[self.risky_index])
                if new_distance > self.risky_distance*2: # One way out of state 2
                    self.dist_counter +=1
                    if self.dist_counter > dist_wait: # To be sure
                        v_ref = 0
                        self.state = 1
                        if verbose:
                            print("Risky dist was cleared!")    
                else:
                    self.dist_counter = 0

                if risks[max_index] > stop_threshold: # Reverse risk is greater again.
                    v_ref = 0 # Stop again
                    self.state = 1
                    if verbose:
                        print("Reversing was more dangerous!")

            ####################################################
            # STATE DC avoidance
            elif (self.state == 1):
                # State change?
                if (risks[max_index] >= stop_threshold):
                    # STOP
                    v_ref = 0
                    self.state = 2
                    self.risky_distance = np.abs(d2[max_index] - d1[max_index]) # distance along risky ray
                    self.risky_index = max_index
                    self.dist_counter = 0
                    ## 50% / 50% of what turn to stop and wait or not
                    #num = np.random.choice([0, 1])
                    #if num==0:
                    #    self.stop_counter = 30 # Done waiting
                    #elif num==1:
                    #    self.stop_counter = 0 # Wait
                    
                    # 50% / 50% of what turn to take
                    num = np.random.choice([0, 1])
                    if num==0:
                        self.backing_angle = self.alpha_max/1.5 
                    elif num==1:
                        self.backing_angle = -self.alpha_max/1.5
                    else:
                        self.backing_angle = 0 #Straight
                    
                    if verbose:
                        print("CAR IS STOPPING...")
                elif risks[max_index] < risk_threshold:
                    # Return to determined
                    self.state = 0
                    alpha_ref = 0
                    v_ref = self.v_max
                else:
                    # Default state 1 behaviour: prefers rht
                    v_ref, alpha_ref = self.determined_dc_avoidance_action(angles, risks, v_ref, alpha_ref)
            ####################################################
            # STATE Determined
            elif (self.state==0):
                # State change?
                if risks[max_index] >= risk_threshold:
                    # Avoid collision!!
                    self.state = 1
                else:
                    # Default state 0 behaviour
                    v_ref = self.v_max
                    alpha_ref = 0
            else:
                raise Exception("Not a valid state: "+str(self.state))
            ####################################################
            return v_ref, alpha_ref, risks

if __name__ == "__main__":
    pass