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