from numpy import sign
from Object import *
from Agent import Agent
import matplotlib.pyplot as plt
from copy import deepcopy
from numba import jit

class Vehicle(Object):
    def __init__(self, center: np.array, length: float, width: float, heading: float, 
                    dt : float = 0.1, alpha_max : float = 1.2, v_max : float = 8, 
                    gamma : float = 5e-7, tau_throttle : float = 1, tau_steering : float = 1):

        """ Vehicle location and vertices """
        self.heading = heading # psi
        self.verticesVCF = self.init_verticesVCF(length, width)
        self.position_center = center
        # initialize the originVCF as a CCT point, hence the minus!
        self.originVCF = self.CCFtoWCF(np.array([0.0, -length / 2]))
        self.vertices = self.vertices_VCTtoWCF(self.verticesVCF)
        super().__init__(center, self.vertices)

        """ Vehicle structure """
        self.d = length # Keep it simple!
        self.length = length
        self.width = width

        """ Current state """
        self.X = np.array([[self.originVCF[0]], [self.originVCF[1]], [0], [0]]) # State vector
        self.alpha = 0  # Wheel angle. Throttle level can be derived from X[2:,:]
        self.omega = 0  # Current turning rate


        """Dynamics parameters"""
        self.dt = dt
        self.tau_throttle = tau_throttle # Time constant throttle (Accelerating, not breaking)
        self.tau_steering = tau_steering # Time constant Steering (Turning)
        self.K_a = self.dt / self.tau_steering # Gain parameter
        self.K_v = self.dt / self.tau_throttle # Gain parameter
        # Limits:
        self.v_max = v_max      
        self.alpha_max = alpha_max
        self.k_max = 3.0        # Velocity signal parameter
        self.c_max = 0.5        # Steering signal parameter

        """ Other parameters"""
        self.gamma = gamma
        self.c = self.alpha_to_inverse_curve()
        self.focus_direction = None # {1, 0, -1}
        self.actual_direction = None # {1, 0, -1}

        self.collided = False

    # Print vehicle infos
    def print_vehicle(self):
        print("Center:", self.position_center)
        print("Vertices:", self.vertices)
        print("Length:", self.length)
        print("lines:", self.lines)
        print("width:", self.width)
        print("originVCF:", self.originVCF)


    def init_verticesVCF(self, length: float, width: float) -> List[np.array]:
        verticesVCF = [np.array([width/2, 0]),
                       np.array([width/2, length]),
                       np.array([-width/2, length]),
                       np.array([-width/2, 0])]
        return verticesVCF

    def vertices_VCTtoWCF(self, verticesVCF) -> List[np.array]:

        verticesWCF = []

        for vertex in verticesVCF:
            verticesWCF.append(self.VCTtoWCF(vertex))
        return verticesWCF

    def rot_matrix(self, angle: float) -> np.array:
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    """Transformation from Center Coordinate Frame to World Coordinate Frame """
    def CCFtoWCF(self, point: np.array) -> np.array:
        #trans_point = np.matmul(self.rot_matrix(self.heading - np.pi/2), point) + self.position_center
        angle = self.heading-np.pi/2
        R_W_C = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        return R_W_C@point + self.position_center
        
    def WCFtoCCF(self, point : np.array) -> np.array:
        # This is same as for CCFtoWCF
        angle = self.heading-np.pi/2
        R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

        return R.T@point - R.T@self.position_center

    """Transformation from Vehicle Coordinate Frame to World Coordinate Frame """
    def VCTtoWCF(self, point: np.array) -> np.array:
        #trans_point = np.matmul(self.rot_matrix(self.heading - np.pi/2), point) + self.originVCF
        angle = self.heading-np.pi/2
        R_W_V = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        return R_W_V@point + self.originVCF
    
    def WCF_rotate_CCF(self, vector_c : np.array):
        """ Inverse of CCFtoWCF"""
        angle = self.heading-np.pi/2
        R_W_C = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

        return R_W_C.T@vector_c 

    def WCF_rotate_VCF(self, vector_wcf : np.array):
        """ Inverse of CCFtoWCF"""
        angle = self.heading-np.pi/2
        R_W_C = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        return R_W_C.T@vector_wcf 

    """Transformation from Vehicle Coordinate Frame to World Coordinate Frame """
    def CCTtoVCF(self, point: np.array) -> np.array:
        # TODO
        return ...


    ####################################
    ### Methods for the motion model ###
    ####################################
    def alpha_to_inverse_curve(self):
        # Mapping from steering angle alpha, to inverse curve radius c = 1/R.
        return np.tan(self.alpha)/self.d

    def one_step_algorithm(self, alpha_ref, v_ref):
        """
        Running one step of the update algorithm.
        """
        ####################################
        # Step 0: Are we reversing or not? #
        ####################################
        # TODO: All bugs connected to directions.
        # >>>>>> Sometimes, it actually runs backwards even if v_ref > 0
        # >>>>>> Then, we get ~infinite values in the update function below.
        if v_ref > 0:
            self.focus_direction = 1
        elif v_ref < 0:
            self.focus_direction = -1
        else: self.focus_direction = 0


        # TODO: rather set based on v_ref, instead of actual v - might just be sliding.
        v_k = self.X[2:,:] # Current speed from state matrix
        v_k_VCF = self.WCF_rotate_VCF(v_k) # rotate to VCF
        if v_k_VCF[1, 0]>0: # Driving down y axis or not.
            self.actual_direction=1
        elif v_k_VCF[1, 0]<0:
            self.actual_direction=-1
        else:
            self.actual_direction=0
    
        #####################################
        # Step 1: update u, alpha, find B_k #
        #####################################

        self.alpha = self.alpha + self.K_a * (alpha_ref - self.alpha)
        u_k = self.K_v * (v_ref-self.actual_direction*np.linalg.norm(v_k))

        # Small value problem.
        if np.abs(u_k) < self.gamma:
            # Done to avoid mini-vibrations: close to infinite acceleration
            # Alternatively filter the inputs/add dampening in the first order response.
            u_k = 0 

        B_k = np.array([[0],
                        [0], 
                        [np.cos(self.heading)], 
                        [np.sin(self.heading)]])
        ######################
        # Step 2: Choose A_k #
        ######################
        if np.abs(self.omega) > self.gamma:
            A_k = np.array([[1, 0, np.sin(self.omega*self.dt)/self.omega, -(1-np.cos(self.omega*self.dt))/self.omega],
                          [0, 1, (1-np.cos(self.omega*self.dt))/self.omega, np.sin(self.omega*self.dt)/self.omega],
                          [0, 0, np.cos(self.omega*self.dt), -np.sin(self.omega*self.dt)], 
                          [0, 0, np.sin(self.omega*self.dt), np.cos(self.omega*self.dt)]], dtype=np.ndarray)
        else: # In case omega ~ 0
            A_k = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]])

        # Step 3: Update state matrix
        self.X = A_k@self.X + B_k*u_k

        # Step 4: Update omega nad psi (based on theta)
        theta_next = np.linalg.norm(v_k)*self.dt*np.tan(self.alpha) / self.d # Turn angle
        self.heading = self.heading + theta_next # Heading in WCF
        self.omega = np.linalg.norm(v_k)*np.tan(self.alpha) / self.d

        # Step 5: Update position_center, vertices and sides accordingly.
        self.originVCF[0], self.originVCF[1] = self.X[0], self.X[1]
        self.position_center = self.VCTtoWCF(np.array([0, self.length/2]))

        self.vertices = self.vertices_VCTtoWCF(self.verticesVCF)
        self.sides = [[self.vertices[0], self.vertices[1]],
                      [self.vertices[1], self.vertices[2]],
                      [self.vertices[2], self.vertices[3]],
                      [self.vertices[3], self.vertices[0]]]
        # Bug: Needs to update the sides as well!
        self.lines = self.eval_lines(self.sides)

    def one_step_algorithm_2(self, alpha_ref, v_ref, dt):
        """
        Running one step of the update algorithm.
        """
        ####################################
        # Step 0: Are we reversing or not? #
        ####################################
        # TODO: All bugs connected to directions.
        # >>>>>> Sometimes, it actually runs backwards even if v_ref > 0
        # >>>>>> Then, we get ~infinite values in the update function below.
        if v_ref > 0:
            self.focus_direction = 1
        elif v_ref < 0:
            self.focus_direction = -1
        else: self.focus_direction = 0


        # TODO: rather set based on v_ref, instead of actual v - might just be sliding.
        v_k = self.X[2:,:] # Current speed from state matrix
        v_k_VCF = self.WCF_rotate_VCF(v_k) # rotate to VCF
        if v_k_VCF[1, 0]>0: # Driving down y axis or not.
            self.actual_direction=1
        elif v_k_VCF[1, 0]<0:
            self.actual_direction=-1
        else:
            self.actual_direction=0
    
        #####################################
        # Step 1: update u, alpha, find B_k #
        #####################################

        self.alpha = self.alpha + self.K_a * (alpha_ref - self.alpha)
        u_k = self.K_v * (v_ref-self.actual_direction*np.linalg.norm(v_k))

        # Small value problem.
        if np.abs(u_k) < self.gamma:
            # Done to avoid mini-vibrations: close to infinite acceleration
            # Alternatively filter the inputs/add dampening in the first order response.
            u_k = 0 

        B_k = np.array([[0],
                        [0], 
                        [np.cos(self.heading)], 
                        [np.sin(self.heading)]])
        ######################
        # Step 2: Choose A_k #
        ######################
        if np.abs(self.omega) > self.gamma:
            A_k = np.array([[1, 0, np.sin(self.omega*dt)/self.omega, -(1-np.cos(self.omega*dt))/self.omega],
                          [0, 1, (1-np.cos(self.omega*dt))/self.omega, np.sin(self.omega*dt)/self.omega],
                          [0, 0, np.cos(self.omega*dt), -np.sin(self.omega*dt)], 
                          [0, 0, np.sin(self.omega*dt), np.cos(self.omega*dt)]], dtype=np.ndarray)
        else: # In case omega ~ 0
            A_k = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]])

        # Step 3: Update state matrix
        self.X = A_k@self.X + B_k*u_k

        # Step 4: Update omega nad psi (based on theta)
        theta_next = np.linalg.norm(v_k)*dt*np.tan(self.alpha) / self.d # Turn angle
        self.heading = self.heading + theta_next # Heading in WCF
        self.omega = np.linalg.norm(v_k)*np.tan(self.alpha) / self.d

        # Step 5: Update position_center, vertices and sides accordingly.
        self.originVCF[0], self.originVCF[1] = self.X[0], self.X[1]
        self.position_center = self.VCTtoWCF(np.array([0, self.length/2]))

        self.vertices = self.vertices_VCTtoWCF(self.verticesVCF)
        self.sides = [[self.vertices[0], self.vertices[1]],
                      [self.vertices[1], self.vertices[2]],
                      [self.vertices[2], self.vertices[3]],
                      [self.vertices[3], self.vertices[0]]]
        # Bug: Needs to update the sides as well!
        self.lines = self.eval_lines(self.sides)

    #######################################################################
    ###################### Nicolo  CIRCOGRAMs #############################
    #######################################################################
    
    # Evaluate line equation in a point
    def eval_value_line_eq(self, line: np.array, point: np.array) -> float:
        """
        PSEUDOCODE

        Evaluate the value of the line equation by substituting a point as x and y
        """
        return line[0]*point[0] + line[1]*point[1] + line[2]

    # METHOD 8: Check the intersection between a line and an entity (it works both for an object and a segment)
    def is_line_intersect_entity(self, line: np.array, vertices: List[np.array]) -> bool:
        """
        PSEUDOCODE

        IF every vertex of the object substituted in the line equation, gives a positive or respectively negative
        result, there is no intersection. Return False
        ELSE there is intersection, return True
        """

        same_side = []
        reference = self.eval_value_line_eq(line, vertices[0]) >= 0

        for vertex in vertices[1:]:
            # We are going to compare every vertex with the initial one, if they are on the same side of the line,
            # we add a True to the list same_side. Same side it is initialized as a list of one element True, because
            # of course the first element is on the same side of itself.
            same_side.append((self.eval_value_line_eq(line, vertex) >= 0) == reference)

        # if all the vertex_values are positive (or negative), there is NO intersection with the ray, return False.
        # Otherwise return True
        return not all(same_side)

    # METHOD 9: find the point of intersection between two lines, knowing that there is intersection
    def find_intersection_line_line(self, line1: np.array, line2: np.array) -> np.array:
        """
        PSEUDOCODE

        return the point of intersection, that can be found by solving the system of the two equations

        """
        a1 = line1[0]
        b1 = line1[1]
        c1 = line1[2]
        a2 = line2[0]
        b2 = line2[1]
        c2 = line2[2]

        den = a1*b2 - a2*b1
        point_inters = np.array([(b1*c2 - b2*c1)/den, (a2*c1 - a1*c2)/den])
        return point_inters

    def is_up(self, point1: np.array, center: np.array) -> bool:
        return point1[1] > center[1]

    def is_right(self, point1: np.array, center: np.array) -> bool:
        return point1[0] > center[0]

    # METHOD 10: find the point of intersection between a line and a object obj and the distance between them
    def find_intersection_ray_object(self, line: np.array, obj: Object, angle: float) -> List:
        """
        PSEUDOCODE
        IF there is intersection with the object (METHOD 8)
            FOR every side of the object
                IF there is intersection of the line with the side (METHOD 8)
                    Find point of intersection (METHOD 9)
            Keep the closest point of intersection to the vehicle by checking distance point-point (METHOD 1)
        Return point
        """
        cent = self.position_center
        inters_points = []
        dist = []

        if self.is_line_intersect_entity(line, obj.vertices):
            for i in range(len(obj.lines)):
                if self.is_line_intersect_entity(line, obj.sides[i]):
                    loc_point = self.find_intersection_line_line(line, obj.lines[i])

                    if (0 < angle < np.pi / 2 and self.is_up(loc_point, cent) and self.is_right(loc_point, cent) or
                        np.pi / 2 < angle < np.pi and self.is_up(loc_point, cent) and not self.is_right(loc_point, cent) or
                        np.pi < angle < 3 / 2 * np.pi and not self.is_up(loc_point, cent) and not self.is_right(loc_point, cent) or
                        3 / 2 * np.pi < angle < 2 * np.pi and not self.is_up(loc_point, cent) and self.is_right(loc_point, cent) or
                        angle == 0 and self.is_right(loc_point, cent) or
                        angle == np.pi / 2 and self.is_up(loc_point, cent) or
                        angle == np.pi and not self.is_right(loc_point, cent) or
                        angle == 3 * np.pi / 2 and not self.is_up(loc_point, cent)):

                        inters_points.append(loc_point)
                        dist.append(self.dist_point_point(loc_point, self.position_center))

        if inters_points:  # if it not empty
            pos_min = dist.index(min(dist))  # find position of the minimum distance in the list
            hit_point = [inters_points[pos_min], dist[pos_min]]
        else:
            hit_point = None
        return hit_point

    # METHOD 11: find the point of intersection between a the circogram ray and the ego vehicle
    def find_intersection_line_ego(self, angle: float, length: float, width: float) -> np.array:
        """
        PSEUDOCODE
        with the angle, evaluate the point via a formula in CCF(center coordinate frame)
        Transform te point from CCF to WCF
        Return the point
        """
        if width * abs(np.sin(angle)) < length * abs(np.cos(angle)):
            x = sign(np.cos(angle)) * width/2
            hit_pointCCF = np.array([x, x * np.tan(angle)])
        else:
            y = sign(np.sin(angle)) * length / 2
            hit_pointCCF = np.array([y / np.tan(angle), y])

        hit_pointWCF = self.CCFtoWCF(hit_pointCCF)

        return hit_pointWCF

    #####################################################################
    #####################################################################'
    ##################
    # MY own attempt #
    ##################
    def static_circogram_2(self, N: int, list_objects_simul, d_horizon: float):
            """
            A slight alteration of the original "static_circogram" function,
             removing a bug where if the vehicle turned, the circogram would stop working properly.
            """
            # Static arrays >> dynamic arrays
            dist_center_P1 = np.zeros(N)
            dist_center_P2 = np.zeros(N)
            P1 = np.zeros((N, 2))
            P2 = np.zeros((N,2))
            angles = np.zeros(N)

            for n in range(N):
                """
                The 1.st ray of the circogram should always be in the heading direction,
                no matter the orientation of the ego-vehicle.
                """
                angle = 2*np.pi/N * n # 
                #angle = 2*np.pi/N * n + np.pi/2 # Now it will go from the the heading (yC), instead of xC.
                angles[n] = angle
                # Find P1 and its distance from the center
                P1[n]=self.find_intersection_line_ego(angle, self.length, self.width)
                dist_center_P1[n]=self.dist_point_point(P1[n], self.position_center)
                
                # Find P2 and its distance from the center
                ray_line = self.eval_line_point_point(self.position_center, P1[n])
                all_intersections = []
                all_distances = []
                for obj in list_objects_simul:
                    for i in range(len(obj.lines)):
                        if self.is_line_intersect_entity(ray_line, obj.sides[i]):
                            loc_point = self.find_intersection_line_line(ray_line, obj.lines[i])
                            distance = self.dist_point_point(loc_point, self.position_center)

                            #if distance < d_horizon: # Cannot be outside of view
                            #    #TODO: all objects passed in here should **already** be close enough to be inside the horizon.
                            #           - Needs to implement a first object pruning: a low-res distance method. 

                            if self.dist_point_point(loc_point, P1[n]) < distance: # Should be on correct side of object!
                                all_intersections.append(loc_point)
                                all_distances.append(self.dist_point_point(loc_point, self.position_center))

                                

                # Choose shortest distance intersection
                if all_intersections:  # if it not empty
                    min_index = all_distances.index(min(all_distances))  # find position of the minimum distance in the list
                    P2[n]=all_intersections[min_index]
                    dist_center_P2[n]=all_distances[min_index]
                else: # No intersections
                    P2[n]=None
                    dist_center_P2[n]=d_horizon

            # return the circogram function, which is a list of N tuple composed by [dist_center_P1, dist_center_P2, P1, P2, agnles]
            circogram = [dist_center_P1, dist_center_P2, P1, P2, angles]
            return circogram

    def dynamic_cicogram_1(self, static_circogram, alpha_ref, v_ref, seconds : int = 1):
        """
        Based on "my method".
        """
        ghost_vehicle = deepcopy(self)
        d1, d2, _, _, angles = static_circogram
        real_distances = d2 - d1 # works with static numpy arrays

        # Simulate ghost movement
        sim_steps = int(seconds / self.dt)
        for _ in range(sim_steps):
            ghost_vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)
        ghost_location = ghost_vehicle.position_center # In WCF
        transition_vector = ghost_location - self.position_center # In WCF
        transition_vector = ghost_vehicle.WCF_rotate_CCF(transition_vector) # in CCF

        # if v_ref>= 0: #Accelerating forward
        #     # Delete all rear risks...
        #     risks[len(risks)//2:]=0
        # else: # Accelerating backwards!
        #     # Delete all forward risks...
        #     risks[:len(risks)//2]=0

        dynamic_circogram = np.zeros((len(angles), 3))
        # TODO: dirction is set by reference speed. Is that good?
        if self.focus_direction == 1: # Forward
            for i in range(0, len(angles)//2):
                angle = angles[i]
                #if angle<(np.pi/6) or angle>(np.pi - np.pi/6):
                #    continue
                # 1 Create a unit vector along the ray (angle)
                r = np.array([np.cos(angle),np.sin(angle)])
                if np.linalg.norm(r)>1.01 or np.linalg.norm(r)<0.99:
                    raise Exception("r is not a unit vector:", r, np.linalg.norm(r))

                # 2 Dot product => Produce the "movement along ray": t
                t = np.dot(r, transition_vector)

                # 3 Calculate "RISK", along ray. t/real_dist
                risk = t / real_distances[i] # Along this ray
                #risk = t / np.linalg.norm(transition_vector)

                # 4 TTC= 1/Risk
                #TTC = 1 / risk
                dynamic_circogram[i] = np.array([angle, real_distances[i], risk])
        elif self.focus_direction == -1: # Backwards
            for i in range(len(angles)//2, len(angles)):
                angle = angles[i]
                #if angle<(np.pi+np.pi/6) or angle>(2*np.pi - np.pi/6):
                #    continue
                # 1 Create a unit vector along the ray (angle)
                r = np.array([np.cos(angle),np.sin(angle)])
                if np.linalg.norm(r)>1.01 or np.linalg.norm(r)<0.99:
                    raise Exception("r is not a unit vector:", r, np.linalg.norm(r))

                # 2 Dot product => Produce the "movement along ray": t
                t = np.dot(r, transition_vector)

                # 3 Calculate "RISK", along ray. t/real_dist
                risk = t / real_distances[i] # Along this ray
                #risk = t / np.linalg.norm(transition_vector)

                # 4 TTC= 1/Risk
                #TTC = 1 / risk
                dynamic_circogram[i] = np.array([angle, real_distances[i], risk])
        else: # Stationary
            pass
        return dynamic_circogram

    def dynamic_cicogram_2(self, static_circogram, alpha_ref, v_ref, seconds : int = 1):
        """
        Based on "my method".
        """
        ghost_vehicle = deepcopy(self)
        d1, d2, _, _, angles = static_circogram
        real_distances = d2 - d1 # works with static numpy arrays

        # Simulate ghost movement
        sim_steps = int(seconds / self.dt)
        for _ in range(sim_steps):
            # TODO: WHY does this work??? NOTE that we pass - alpha ref
            # NOTE could also pass 0
            ghost_vehicle.one_step_algorithm(alpha_ref=-alpha_ref, v_ref=v_ref)
        ghost_location = ghost_vehicle.position_center # In WCF
        transition_vector = ghost_location - self.position_center # In WCF
        transition_vector = ghost_vehicle.WCF_rotate_CCF(transition_vector) # in CCF

        dynamic_circogram = np.zeros((len(angles), 3))
        # TODO: dirction is set by reference speed. Is that good?
        for i, angle in enumerate(angles):
            # 1 Create a unit vector along the ray (angle)
            r = np.array([np.cos(angle),np.sin(angle)])

            # 2 Dot product => Produce the "movement along ray": t
            t = np.dot(r, transition_vector)

            # 3 Calculate "RISK", along ray. t/real_dist
            #risk = t / np.linalg.norm(transition_vector) # Only gives cos(angle) => angle closest to transition vecotr
            #risk = t # Does exact same as above, only cos(angle)*|t_vector|, which is same for every angle
            risk = t / real_distances[i] # Along this ray

            # 4 TTC= 1/Risk
            #TTC = 1 / risk
            dynamic_circogram[i] = np.array([angle, real_distances[i], risk])

        return dynamic_circogram

    def dynamic_circogram_3(self, dynamic_circograms : list, h : float):
        """ Numerical differentiate over consecutive DCs, to see where risks grows the most"""
        # TODO: This is not noice-tolerant; only relies on single measurement.
        dc1 = dynamic_circograms[0]
        dc2 = dynamic_circograms[1]
        #
        risks_1 = dc1[:,2]
        risks_2 = dc2[:, 2]
        #
        differentiated_risks = (risks_2 - risks_1)/h
        return differentiated_risks
        
    # Collision check for the ego vehicle with the other objects
    def collision_check(self, d1, d2):
        d1 = np.array(d1)
        diff = (d2 - d1) >= 0
        self.collided = not all(diff)
        return self.collided #might be unnecessary

if __name__ == "__main__":
    # Vehicle
    dt = 0.1 
    alpha_ref = 0
    v_ref = -1
    car = Vehicle(np.array([25, 25]), length=4, width=2, heading=np.pi/2, dt=0.1)
    # Driver
    alpha_max = 0.8
    v_max = 5
    v_min = -2
    var_alpha= 0.2 # 0.3
    var_vel= 0.2 # 0.5
    agent = Agent(v_max=v_max, v_min=v_min, alpha_max=alpha_max, var_alpha=var_alpha, var_vel=var_vel)
    # Bookkeeping
    steps = 1000
    states = np.zeros((4, 1, steps))
    alphas = np.zeros((steps,))
    directions = np.ones((steps,))
    v_refs = np.ones((steps,))*v_ref
    alpha_refs = np.ones((steps,))*alpha_ref
    psis = np.zeros((steps,))
    # Run for steps
    from time import time
    t0 = time()
    for i in range(steps):
        # Check for new random motion every second.
        if (i*car.dt).is_integer: # Checks only for new commands on whole seconds
            v_ref, alpha_ref = agent.brownian_action(v_ref, alpha_ref, r_factor=0.02)
        
        # Book-keeping
        alphas[i] = car.alpha
        directions[i] = car.actual_direction
        alpha_refs[i] = alpha_ref
        v_refs[i]= v_ref
        states[:, :, i] = car.X
        psis[i] = car.heading

        # Run one cycle
        car.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)
    
    # Plotting
    time_axis = np.linspace(0, steps*dt-dt, steps)
    Xs = states[0, :].T
    Ys = states[1, :].T
    Vs = states[2:, :]
    print(Vs.shape)
    abs_speeds = np.linalg.norm(Vs.T, axis=2)
    for i in range(len(abs_speeds)):
        abs_speeds[i] = abs_speeds[i]*directions[i]

    print("Time:", time()-t0)


    plt.title("Speeds")
    plt.plot(time_axis, abs_speeds, label="Real")
    plt.plot(time_axis, v_refs, label="Reference")
    plt.xlabel("Time[s]")
    plt.ylabel("Speed [m/s]")
    plt.legend()
    plt.show()

    plt.title("Steering angles (alpha)")
    plt.xlabel("Time[s]")
    plt.ylabel("Steering angle [rad]")
    plt.plot(time_axis, alphas, label="Real")
    plt.plot(time_axis, alpha_refs, label="Reference")
    plt.legend()
    plt.show()

    plt.plot(time_axis, Xs)
    plt.title("X pos over time") 
    plt.xlabel("Time[s]")
    plt.ylabel("X pos[m]")
    plt.show()

    plt.plot(time_axis, Ys)
    plt.title("Y pos over time") 
    plt.xlabel("Time[s]")
    plt.ylabel("Y pos[m]")
    plt.show()

    plt.plot(Xs, Ys)
    plt.title("Trajectory in the plane") 
    plt.xlabel("X pos [m]")
    plt.ylabel("Y pos [m]")
    plt.show()

    plt.plot(time_axis, psis)
    plt.title("Heading over time") 
    plt.xlabel("Time [s]")
    plt.ylabel("Heading[rad]")
    plt.show()