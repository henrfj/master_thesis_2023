import pygame
import math
import numpy as np
from Object import Object
from Vehicle import Vehicle
from limo import Limo
from Agent import Agent
import copy
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import time
from typing import List
from time import time


class Visualization:
    """

    """

    def __init__(self, dimentions, pixels_per_unit, robot_img_path="graphics/small_robot.png", map_img_path="graphics/test_map_2.png") -> None:
        pygame.init()

        # COLORS
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

        # Map
        try:
            self.robot = pygame.image.load(robot_img_path)
        except:
            print("No robot image")
        
        self.map_img = pygame.image.load(map_img_path)

        # Dimensions
        self.height, self.width = dimentions
        self.ppu = pixels_per_unit

        # Window settings
        pygame.display.set_caption("Cordinated turn motion model: Vehicle") # Title
        self.map = pygame.display.set_mode((self.width, self.height))       # Canvas
        self.map.blit(self.map_img, (0, 0))                                 # Clear

    def draw_robot(self, state, heading, alpha, d, robot : Vehicle = None):
        """  For drawing the images on nicely. Inverted coordinates """
        # Extract from state
        x = state[0, 0]
        y = state[1, 0]
        v_x = state[2, 0]
        v_y = state[3, 0]
        v = np.sqrt(v_x**2 + v_y**2)

        rotated = pygame.transform.rotozoom(self.robot, math.degrees(heading), 1) # Rotate robot image to heading
        
        x, y = self.to_pygame(x, y)
        rect = rotated.get_rect(center=(int(x), int(y))) # Bounding rectangle in world coordinates
        self.map.blit(rotated, rect) # Draw roboto onto the map

        heading = -heading # World is flipped aboud x-axis.
        # TODO: Does not rotate about center, but about back axel of vehicle.  
        real_x = x - (d/2)*np.cos(heading)
        real_y = y - (d/2)*np.sin(heading)
        #pygame.draw.circle(self.map, self.red, (real_x, real_y), 3, 0)
        # Draw heading / other useful vectors
        end_x = real_x + 5*v*np.cos(heading)
        end_y = real_y + 5*v*np.sin(heading)
        pygame.draw.line(self.map, self.red, (real_x, real_y), (end_x, end_y), width=3)

        # Draw turn radius
        if np.abs(np.tan(alpha))>1e-7:
            end_x = real_x - d/np.tan(alpha)*np.cos(np.pi/2 + heading)
            end_y = real_y - d/np.tan(alpha)*np.sin(np.pi/2 + heading)
            pygame.draw.line(self.map, self.green, (real_x, real_y), (end_x, end_y), width=3)
        else:
            pass

        # Draw trajectory?
        if robot:
            trajectory_robot = copy.deepcopy(robot) # To use the same parameters
            # Make sure that the state is equal
            trajectory_robot.X = state 
            trajectory_robot.psi = heading
            trajectory_robot.alpha = alpha

            n = 100
            for i in range(n):
                x = trajectory_robot.X[0, 0]
                y = trajectory_robot.X[1, 0]
                traj_x, traj_y = self.to_pygame(x, y)
                pygame.draw.circle(self.map, self.blue, (traj_x, traj_y), 1, 0)
                trajectory_robot.one_step_algorithm(alpha_ref=alpha, v_ref=v)
           
    def to_pygame(self, x, y):
        """Convert coordinates into pygame coordinates (lower-left => top left)."""
        return (x, self.height - y)
    
    def clear_canvas(self):
        self.map.blit(self.map_img, (0,0))
    
    def update_display(self):
        pygame.display.update()

    def draw_static_circogram_data(self, circog, vehicle, color=(0, 0, 255)):
        # Extract useful info:
        _, _, P1, P2, _ = circog
        for i, ego_points in enumerate(P1):
            # Draw ego-hits
            pygame.draw.circle(self.map, self.blue, (ego_points[0]*self.ppu, ego_points[1]*self.ppu), 2, 0)
            #pygame.draw.line(self.map, self.red, (vehicle.position_center[0]*self.ppu, vehicle.position_center[1]*self.ppu), (ego_points[0]*self.ppu, ego_points[1]*self.ppu), width=2)
            
            # Draw lines between P1 and P2
            if P2[i] is not None:
                start_x = vehicle.position_center[0]*self.ppu
                start_y = vehicle.position_center[1]*self.ppu
                end_x = P2[i][0]*self.ppu
                end_y = P2[i][1]*self.ppu 

                pygame.draw.line(self.map, color, (start_x, start_y), (end_x, end_y), width=2)
                pygame.draw.circle(self.map, color, (end_x, end_y), 5, 0)

        # Mark the center vehicle
        #pygame.draw.circle(self.map, self.blue, (vehicle.position_center[0]*self.ppu, vehicle.position_center[1]*self.ppu), 6, 0)
    
    def draw_dynamic_circogram_data(self, dyn_circog, static_circog, risk_threshold=0.3, verbose=True):
        # Extract useful info:
        d1, d2, P1, P2, angles = static_circog
        risks = dyn_circog[:,2]
        max_risk_index = np.argmax(risks)
        for n, angle in enumerate(angles):
            if P2[n] is not None:
                start_x =  P1[n][0]*self.ppu
                start_y =  P1[n][1]*self.ppu
                end_x = P2[n][0]*self.ppu
                end_y = P2[n][1]*self.ppu
                if risks[n]>risk_threshold:
                    if n == max_risk_index:
                        pygame.draw.line(self.map, self.blue, (start_x, start_y), (end_x, end_y), width=3)
                    else: 
                        pygame.draw.line(self.map, self.red, (start_x, start_y), (end_x, end_y), width=3)
                else:
                    if verbose:
                        pygame.draw.line(self.map, self.green, (start_x, start_y), (end_x, end_y), width=3)

    def draw_DC_3_data(self, risks, static_circog, risk_threshold=0.3, verbose=True):
        # Extract useful info:
        d1, d2, P1, P2, angles = static_circog
        max_risk_index = np.argmax(risks)
        for n, angle in enumerate(angles):
            if P2[n] is not None:
                start_x =  P1[n][0]*self.ppu
                start_y =  P1[n][1]*self.ppu
                end_x = P2[n][0]*self.ppu
                end_y = P2[n][1]*self.ppu
                if risks[n]>risk_threshold:
                    if n == max_risk_index:
                        pygame.draw.line(self.map, self.blue, (start_x, start_y), (end_x, end_y), width=3)
                    else: 
                        pygame.draw.line(self.map, self.red, (start_x, start_y), (end_x, end_y), width=3)
                else:
                    if verbose:
                        pygame.draw.line(self.map, self.green, (start_x, start_y), (end_x, end_y), width=3)
    
    def draw_all_objects(self, objects : List[Object], color=(0, 0, 0), width=6):
        # First draw all vehicles
        for obj in objects:
            self.draw_one_object(obj, color, width)

    def draw_one_object(self, object, color=(0, 0, 0), width=6):
        for side in object.sides:
            start_x = side[0][0]*self.ppu
            start_y = side[0][1]*self.ppu
            end_x = side[1][0]*self.ppu
            end_y = side[1][1]*self.ppu
            pygame.draw.line(self.map, color, (start_x, start_y), (end_x, end_y), width=width)


    def draw_sides(self, sides : List[Object], color=(0, 0, 0), width=6):
        # First draw all vehicles
        for side in sides:
            start_x = side[0][0]*self.ppu
            start_y = side[0][1]*self.ppu
            end_x = side[1][0]*self.ppu
            end_y = side[1][1]*self.ppu
            pygame.draw.line(self.map, color, (start_x, start_y), (end_x, end_y), width=width)

    def draw_headings(self, cars : List[Vehicle], scale : bool = True):
        for car in cars:
            if scale:
                # Scale with speed
                v_x = car.X[2, 0]
                v_y = car.X[3, 0]
                v = np.sqrt(v_x**2 + v_y**2)
            else:
                v = 2

            # Draw heading / other useful vectors
            start_x = car.position_center[0]*self.ppu
            start_y = car.position_center[1]*self.ppu
            end_x = start_x + 2*np.cos(car.heading)*self.ppu + 2*v*np.cos(car.heading)*self.ppu
            end_y = start_y + 2*np.sin(car.heading)*self.ppu + 2*v*np.sin(car.heading)*self.ppu
            pygame.draw.line(self.map, self.green, (start_x, start_y), (end_x, end_y), width=4)

    def draw_center_and_headings_simple(self, heading, position_center):
        # Draw heading on single vehicle
        v = 2
        # Draw heading / other useful vectors
        start_x = position_center[0]*self.ppu
        start_y = position_center[1]*self.ppu
        end_x = start_x + 2*np.cos(heading)*self.ppu + 2*v*np.cos(heading)*self.ppu
        end_y = start_y + 2*np.sin(heading)*self.ppu + 2*v*np.sin(heading)*self.ppu
        pygame.draw.line(self.map, self.green, (start_x, start_y), (end_x, end_y), width=4)
        # Draw centers
        pygame.draw.circle(self.map, self.green, (start_x, start_y), 7, 0)

    def draw_goal_state(self, goal_position, threshold=0, width=4):
        center_x = goal_position[0]*self.ppu
        center_y = goal_position[1]*self.ppu
        # Draw exact spot
        pygame.draw.circle(self.map, self.red, (center_x, center_y), width, 0)
        if threshold:
            pygame.draw.circle(self.map, self.red, (center_x, center_y), threshold*self.ppu, 3)


    def draw_goal_path(self, goal_path, current_goal_index, threshold=0):
        for i, goal in enumerate(goal_path):
            center_x = goal[0]*self.ppu
            center_y = goal[1]*self.ppu
            # Draw exact spot
            pygame.draw.circle(self.map, self.red, (center_x, center_y), 4, 0)
            if i == current_goal_index: # Current goal gets a little ring
                pygame.draw.circle(self.map, self.red, (center_x, center_y), threshold*self.ppu, 3)

    def draw_centers(self, cars : List[Vehicle]):
        for car in cars:
            # Retrieve centers
            center_x = car.position_center[0]*self.ppu
            center_y = car.position_center[1]*self.ppu
            pygame.draw.circle(self.map, self.green, (center_x, center_y), 7, 0)

    def display_fps(self, fps, font_size=32, color="white", where=(0,0)):
        "Renders the fonts as passed from display_fps"
        font = pygame.font.SysFont("Arial", 32)
        text_to_show = font.render(str(int(fps)), 0, pygame.Color(color))
        self.map.blit(text_to_show, where)

###################################################################
############################# TESTING #############################
###################################################################
def PointsOnCircum(r,n=100, center=(0,0)):
    circle_points=np.asarray([(np.cos(2*np.pi/n*x)*r,np.sin(2*np.pi/n*x)*r) for x in range(0,n+1)])
    circle_points[:, 0] += center[0]
    circle_points[:, 1] += center[1]
    return circle_points

def still_circogram_test():
    # Spawn in 4 cars    
    car1 = Vehicle(np.array([25, 25]), length=4, width=2, heading=np.pi/2) # np.pi/2
    car2 = Vehicle(np.array([20, 28]), length=4, width=2, heading=np.pi/42)
    car3 = Vehicle(np.array([30, 20]), length=4, width=2, heading=np.pi/8)
    car4 = Vehicle(np.array([30, 30]), length=4, width=2, heading=np.pi/5)
    objects = [car1, car2, car3, car4]
    
    # Testing Circogram
    N = 100
    horizon = 50
    x = np.linspace(0, 2*np.pi, N)
    #
    before_time = time.time()
    circog = car1.static_circogram_2(N, [car2, car3, car4], horizon)
    print("Time to calculate circogram with",N,"rays.", time.time()-before_time)
    d1, d2, P1, P2, angle = circog
    
    """ Plotting """
    x = np.linspace(0, 2*np.pi, N)
    plt.title("Car_1")
    plt.plot(x, d1, 'b+', label='Ego perimeter')
    plt.plot(x, d2, 'r+', label='Objects surrounding')
    plt.legend(loc='best')
    #plt.savefig('figures/Circogram_graph.pdf')
    #plt.savefig('figures/Circogram_Car_1.png', dpi=300)
    plt.show()

    ##############
    # Visualize! #
    ##############
    MAP_DIMENSIONS = (800, 800)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=20, map_img_path="graphics/test_map_2.png") # Also initializes the display
    
    gfx.clear_canvas()
    gfx.draw_all_objects(objects) 
    gfx.draw_static_circogram_data(circog, car1)

    gfx.update_display()
    time.sleep(1000)

def dynamic_circogram_test():
    # Spawn in 4 cars    
    car1 = Vehicle(np.array([22, 15]), length=4, width=2, heading=np.pi/2) # np.pi/2
    car2 = Vehicle(np.array([20, 28]), length=4, width=2, heading=np.pi/42)
    car3 = Vehicle(np.array([30, 20]), length=4, width=2, heading=np.pi/8)
    car4 = Vehicle(np.array([30, 30]), length=4, width=2, heading=np.pi/5)
    objects = [car1, car2, car3, car4]
    # ...And a wall
    wall1 = Object(np.array([20, 20]), vertices=np.array([[5, 5],
                                                          [5, 55], 
                                                          [55, 55],
                                                          [40, 40],
                                                          [55, 5],
                                                          [35, 6]])) # Can now add more than 4 sides to objects!
    objects = [car1, car2, car3, car4]
    cars = [car1, car2, car3, car4]
    
    # Testing Circogram
    N = 72
    horizon = 20
    #
    static_circogram = list(car1.static_circogram_2(N, objects[1:], horizon))
    # Dynamic circogram:
    alpha_ref = 0.4
    v_ref = 1
    dynamic_circogram = car1.dynamic_cicogram_1(static_circogram, alpha_ref=alpha_ref, v_ref=v_ref)
    
    """ Plotting """
    d1, d2, P1, P2, angles = static_circogram
    risks = dynamic_circogram[:, 2]
    real_distances = dynamic_circogram[:, 1]
    #x = np.linspace(0, 2*np.pi, N)
    plt.title("Car_1")
    plt.plot(angles, d1, 'b+', label='Ego perimeter')
    plt.plot(angles, d2, 'r+', label='Objects surrounding')
    plt.plot(angles, risks*100, "r.", label="Risks")
    plt.plot(angles, real_distances, "g.", label = "real distances")
    plt.legend(loc='best')
    plt.xlabel("Angle[rad]")
    plt.ylabel("Distance[units]")
    #plt.savefig('figures/Circogram_graph.pdf')
    #plt.savefig('figures/Circogram_Car_1.png', dpi=300)
    plt.show()

    ##############
    # Visualize! #
    ##############
    MAP_DIMENSIONS = (1200, 1200)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=20, map_img_path="graphics/test_map_2.png") # Also initializes the display
    gfx.clear_canvas()
    gfx.draw_all_objects(objects) 
    gfx.draw_static_circogram_data(static_circogram, car1)
    gfx.draw_centers(cars)
    gfx.draw_headings(cars)
    gfx.update_display()
    time.sleep(1000)

def map_circles_multi(scale=1, height=1080, width=1920, pixels_per_unit=20):
    # Create a visualizer
    MAP_DIMENSIONS = (height*scale, width*scale)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
    # Spawn in the walls:
    outer_circle_points = PointsOnCircum(r=scale*36, n=50, center=(37*scale, 36*scale))
    wall1 = Object(np.array([20, 20]), vertices=outer_circle_points)
    #wall1 = Object(np.array([20, 20]), vertices=np.array([[5, 17], # A tunnel
    #                                                      [55, 17], 
    #                                                      [55, 23],
    #                                                      [5, 23]])) 

    small_circle_points = PointsOnCircum(r=8*scale, n=15, center=(35*scale, 44*scale))
    wall2 = Object(np.array([35, 20]), vertices=small_circle_points)
    small_circle_points = PointsOnCircum(r=8*scale, n=4, center=(23*scale, 33*scale))
    wall3 = Object(np.array([20, 20]), vertices=small_circle_points)
    small_circle_points = PointsOnCircum(r=8*scale, n=6, center=(37*scale, 73*scale))
    wall4 = Object(np.array([20, 20]), vertices=small_circle_points)
    small_circle_points = PointsOnCircum(r=8*scale, n=8, center=(30*scale, 13*scale))
    wall5 = Object(np.array([20, 20]), vertices=small_circle_points)
    small_circle_points = PointsOnCircum(r=8*scale, n=30, center=(55*scale, 30*scale))
    wall6 = Object(np.array([20, 20]), vertices=small_circle_points)
    small_circle_points = PointsOnCircum(r=10*scale, n=4, center=(-5*scale, 30*scale))
    wall7 = Object(np.array([20, 20]), vertices=small_circle_points)

    # Spawn in 2 limo-cars
    car1 = Vehicle(np.array([20*scale, 20*scale]), length=4*scale, width=2*scale, heading=np.pi/2, tau_steering=1, tau_throttle=1) #np.pi/2
    car2 = Vehicle(np.array([60*scale, 20*scale]), length=6*scale, width=3*scale, heading=np.pi, tau_steering=1, tau_throttle=1)
    car3 = Vehicle(np.array([60*scale, 50*scale]), length=4*scale, width=2*scale, heading=0, tau_steering=1, tau_throttle=1)
    car4 = Vehicle(np.array([50*scale, 30*scale]), length=4*scale, width=2*scale, heading=3*np.pi/2, tau_steering=1, tau_throttle=1)

    objects = [car1, car2, car3, car4, wall1, wall2]#, wall3, wall4, wall5, wall6, wall7]
    cars = [car1, car2, car3, car4]
    return gfx, objects, cars

def map_maze(scale=1, height=1080, width=1920, pixels_per_unit=20):
    # Create a visualizer
    MAP_DIMENSIONS = (height*scale, width*scale)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
    
    # Spawn in the walls:
    vertices = np.array([[5, 5], [5, 105], [180, 105], [180, 5]])
    wall1 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[20, 20], [20, 105//2], [180//2, 105//2], [180//2, 20]])
    wall2 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[20, 105//2 + 20], [20, 105-10], [180//2, 105-10], [180//2, 105//2+20]])
    wall3 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[180//2 + 20, 20], [180//2 + 20, 105//2], [180-20, 105//2], [180-20, 20]])
    wall4 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[180//2 + 20, 105//2+20], [180//2 + 20, 105-10], [180-20, 105-10], [180-20, 105//2+20]])
    wall5 = Object(np.array([0, 0]), vertices=vertices)
    # Obstacles
    vertices = np.array([[70, 105//2], [70, 105//2+20], [80, 105//2+20], [80, 105//2]])
    wall6 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[180//2, 90], [180//2+20, 90], [180//2+20, 80], [180//2, 80]])
    wall7 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[40, 5], [40, 20], [45, 20], [45, 5]])
    wall8 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[160, 5], [160, 20], [155, 20], [155, 5]])
    wall9 = Object(np.array([0, 0]), vertices=vertices)

    # Extra spice
    vertices = np.array([[5, 40], [5, 45], [20, 45], [20, 40]])
    wall10 = Object(np.array([0, 0]), vertices=vertices)

    # Spawn in 2 limo-cars
    car1 = Vehicle(np.array([10*scale, 10*scale]), length=4*scale, width=2*scale, heading=0, tau_steering=0.4, tau_throttle=0.4) #np.pi/2
    car2 = Vehicle(np.array([60*scale, 10*scale]), length=4*scale, width=2*scale, heading=np.pi)

    objects = [car1, car2, wall1, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1, car2]
    return gfx, objects, cars

def map_maze_multi(scale=1, height=1080, width=1920, pixels_per_unit=20):
    # Create a visualizer
    MAP_DIMENSIONS = (height*scale, width*scale)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
    
    # Spawn in the walls:
    vertices = np.array([[5, 5], [5, 105], [180, 105], [180, 5]])
    wall1 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[20, 20], [20, 105//2], [180//2, 105//2], [180//2, 20]])
    wall2 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[20, 105//2 + 20], [20, 105-10], [180//2, 105-10], [180//2, 105//2+20]])
    wall3 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[180//2 + 20, 20], [180//2 + 20, 105//2], [180-20, 105//2], [180-20, 20]])
    wall4 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[180//2 + 20, 105//2+20], [180//2 + 20, 105-10], [180-20, 105-10], [180-20, 105//2+20]])
    wall5 = Object(np.array([0, 0]), vertices=vertices)
    # Obstacles
    vertices = np.array([[70, 105//2], [70, 105//2+20], [80, 105//2+20], [80, 105//2]])
    wall6 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[180//2, 90], [180//2+20, 90], [180//2+20, 80], [180//2, 80]])
    wall7 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[40, 5], [40, 20], [45, 20], [45, 5]])
    wall8 = Object(np.array([0, 0]), vertices=vertices)
    vertices = np.array([[160, 5], [160, 20], [155, 20], [155, 5]])
    wall9 = Object(np.array([0, 0]), vertices=vertices)

    # Spawn in 4 cars
    car1 = Vehicle(np.array([10*scale, 10*scale]), length=4*scale, width=2*scale, heading=np.pi/2, tau_steering=0.5, tau_throttle=0.5) 
    car2 = Vehicle(np.array([10*scale, 80*scale]), length=6*scale, width=3*scale, heading=np.pi, tau_steering=0.5, tau_throttle=0.5)
    car3 = Vehicle(np.array([90*scale, 70*scale]), length=4*scale, width=2*scale, heading=0, tau_steering=0.5, tau_throttle=0.5)
    car4 = Vehicle(np.array([100*scale, 30*scale]), length=4*scale, width=2*scale, heading=3*np.pi/2, tau_steering=0.5, tau_throttle=0.5)

    objects = [car1, car2, car3, car4, wall1, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1, car2, car3, car4]
    return gfx, objects, cars

def driving_with_single_random_driver():
    # Create a visualizer
    #gfx, objects, cars = map_circles(scale=1, height=1080, width=1920, pixels_per_unit=10)
    #gfx, objects, cars = map_maze(scale=1, height=1080, width=1920, pixels_per_unit=10)
    gfx, objects, cars = map_tube(scale=1, height=1080, width=1920, pixels_per_unit=10)
    car1 = cars[0]

    # Spawn a driver
    alpha_max = 0.9
    v_max = 8 # 8
    v_min = -4 # -4
    var_alpha = 0.3 # 0.3
    var_vel = 1 # 0.3
    agent = Agent(v_max, v_min, alpha_max, var_vel, var_alpha)
    # Make it a limo!
    limo = Limo(vehicle=car1, driver=agent)

    # Sets certain parameters
    steps = 100000
    circograms = True
    # TODO: move these params to the agent!
    alpha_ref = 0
    v_ref = 4 # 4
    collision = False
    for i in range(steps):
        if not collision:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    exit()

            
            ##############
            # Visualize! #
            ##############   
            gfx.clear_canvas()
            gfx.draw_all_objects(objects) 
            if circograms:
                # Generate circogram
                N = 36
                horizon = 100
                #
                static_circogram = car1.static_circogram_2(N, objects[1:], horizon)
                dynamic_circogram = car1.dynamic_cicogram_2(static_circogram, alpha_ref, v_ref, seconds=10)
                d1, d2, _, _, _ = static_circogram
                #collision = car1.collision_check(d1, d2)
                #gfx.draw_static_circogram_data(static_circogram, car1)
                gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=0, verbose=False)


            #gfx.draw_headings(cars, scale=True)
            gfx.draw_centers(cars)
            gfx.update_display()
            #time.sleep(1000)

            ##############
            # Kinematics #
            ##############
            #if (i*limo.vehicle.dt).is_integer: # Checks only for new commands on whole seconds
            #    #v_ref, alpha_ref = limo.driver.new_brownian_dc_action(dynamic_circogram, v_ref, alpha_ref, risk_threshold=0.7, stop_threshold=4, r_factor=0.05)
            #    #v_ref, alpha_ref = limo.driver.experimental_driving_action(dynamic_circogram, v_ref, alpha_ref, risk_threshold=0.5, stop_threshold=8, r_factor=0.05)
            #    _, alpha_ref = limo.driver.brownian_action(v_ref, alpha_ref, 0.05)
                
            # Run one cycle
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)
        else:
            print("Collision!")
            v_ref = 0
            alpha_ref = 0

def driving_with_multiple_random_drivers():
    # Create a visualizer
    gfx, objects, cars = map_circles_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    #gfx, objects, cars = map_maze(scale=1, height=1080, width=1920, pixels_per_unit=10)

    # Spawn a driver
    alpha_max = 1
    v_max = 8 # 8
    v_min = -4 # -4
    var_alpha = 0.3 # 0.3
    var_vel = 1 # 0.3
    agent1 = Agent(v_max, v_min, alpha_max, var_vel, var_alpha)
    agent2 = Agent(v_max, v_min, alpha_max, var_vel, var_alpha)
    agent3 = Agent(v_max, v_min, alpha_max, var_vel, var_alpha)
    agent4 = Agent(v_max, v_min, alpha_max, var_vel, var_alpha)
    # Make it a limo!
    limo1 = Limo(vehicle=cars[0], driver=agent1)
    limo2 = Limo(vehicle=cars[1], driver=agent2)
    limo3 = Limo(vehicle=cars[2], driver=agent3)
    limo4 = Limo(vehicle=cars[3], driver=agent4)
    limos = [limo1, limo2, limo3, limo4]

    # Sets certain parameters
    steps = 100000
    # TODO: move these params to the agent?
    alpha_refs = [0, 0, 0, 0]
    v_refs = [4, 4, 1, v_max] 
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Press x button
                exit()

        ##############
        # Visualize! #
        ##############   
        gfx.clear_canvas()
        gfx.draw_all_objects(objects) 
        
        # Generate circogram
        N = 36
        horizon = 100
        # Circograms!
        DCs = []
        SCs = []
        for n, car in enumerate(cars):
            static_circogram = car.static_circogram_2(N, objects[0:n]+objects[n+1:], horizon)
            dynamic_circogram = car.dynamic_cicogram_1(static_circogram, alpha_refs[n], v_refs[n], seconds=3)
            d1, d2, _, _, _ = static_circogram
            car.collision_check(d1, d2)
            #gfx.draw_static_circogram_data(static_circogram, car)
            gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=0.7, verbose=False)
            DCs.append(dynamic_circogram)
            SCs.append(static_circogram)


        gfx.draw_headings(cars, scale=True)
        gfx.draw_centers(cars)
        gfx.update_display()
        #time.sleep(1000)

        ##############
        # Kinematics #
        ##############
        for n, limo in enumerate(limos):
            if not limo.vehicle.collided:
                if (i*limo.vehicle.dt).is_integer: # Checks only for new commands on whole seconds
                    #v_refs[n], alpha_refs[n] = limo.driver.new_brownian_dc_action(DCs[n], v_refs[n], alpha_refs[n], risk_threshold=0.7, stop_threshold=4, r_factor=0.05)
                    v_refs[n], alpha_refs[n] = limo.driver.experimental_driving_action(DCs[n], v_refs[n], alpha_refs[n], risk_threshold=0.7, stop_threshold=4, r_factor=0.05)
                # Run one cycle
            else:
                v_refs[n], alpha_refs[n] = (0, 0)
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_refs[n], v_ref=v_refs[n])

def driving_with_multiple_random_drivers_maze():
    # Create a visualizer
    gfx, objects, cars = map_maze_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    
    # Spawn drivers
    alpha_max = 1
    v_max = 8 # 8
    v_min = -4 # -4
    var_alpha = 0.3 # 0.3
    var_vel = 1 # 0.3
    limos = []
    for car in cars:
        agent = Agent(v_max, v_min, alpha_max, var_vel, var_alpha)
        # Make it a limo!
        limo = Limo(vehicle=car, driver=agent)
        limos.append(limo)

    # Sets certain parameters
    steps = 100000
    # TODO: move these params to the agent?
    alpha_refs = [0, 0, 0, 0]
    v_refs = [4, 4, 4, 4] 
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Press x button
                exit()

        ##############
        # Visualize! #
        ##############   
        gfx.clear_canvas()
        gfx.draw_all_objects(objects) 
        
        # Generate circogram
        N = 36
        horizon = 100
        # Circograms!
        DCs = []
        SCs = []
        for n, car in enumerate(cars):
            static_circogram = car.static_circogram_2(N, objects[0:n]+objects[n+1:], horizon)
            dynamic_circogram = car.dynamic_cicogram_1(static_circogram, alpha_refs[n], v_refs[n], seconds=3)
            d1, d2, _, _, _ = static_circogram
            car.collision_check(d1, d2)
            #gfx.draw_static_circogram_data(static_circogram, car)
            gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=0.7, verbose=False)
            DCs.append(dynamic_circogram)
            SCs.append(static_circogram)


        gfx.draw_headings(cars, scale=True)
        gfx.draw_centers(cars)
        gfx.update_display()
        #time.sleep(1000)

        ##############
        # Kinematics #
        ##############
        for n, limo in enumerate(limos):
            if not limo.vehicle.collided:
                if (i*limo.vehicle.dt).is_integer: # Checks only for new commands on whole seconds
                    v_refs[n], alpha_refs[n] = limo.driver.experimental_driving_action(DCs[n], v_refs[n], alpha_refs[n], risk_threshold=0.7, stop_threshold=4, r_factor=0.05)
                    #v_refs[n], alpha_refs[n] = limo.driver.determined_driver(DCs[n], SCs[n], v_refs[n], alpha_refs[n], risk_threshold=0.7, stop_threshold = 3, verbose=False)
                # Run one cycle
            else:
                v_refs[n], alpha_refs[n] = (0, 0)
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_refs[n], v_ref=v_refs[n])
 
def map_tube(scale=1, height=1080, width=1920, pixels_per_unit=10):
    # Create a visualizer
    MAP_DIMENSIONS = (height*scale, width*scale)
    gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
    
    # Spawn in the walls:
    vertices = np.array([[5, 5], [5, 20], [100, 20], [100, 5]])
    wall1 = Object(np.array([0, 0]), vertices=vertices)

    # Spawn in 1 limo-cars
    car1 = Vehicle(np.array([12*scale, 12*scale]), length=4*scale, width=2*scale, heading=0, tau_steering=0.4, tau_throttle=0.4) #np.pi/2

    objects = [car1, wall1] #, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1]
    return gfx, objects, cars

def driving_with_single_determined_driver():
    # Create a visualizer
    #gfx, objects, cars = map_tube(scale=1, height=1080, width=1920, pixels_per_unit=10)
    gfx, objects, cars = map_lanes_single(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True)
    car1 = cars[0]

    # Spawn a driver
    alpha_max = 1.2
    v_max = 15 # 8
    v_min = -2 # -4
    agent = Agent(v_max, v_min, alpha_max)
    # Make it a limo!
    limo = Limo(vehicle=car1, driver=agent)

    # Sets certain parameters
    steps = 10000000
    # TODO: move these params to the agent!
    alpha_ref = 0
    v_ref = 4 # 4
    collision = False
    for i in range(steps):
        if not collision:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    exit()

            
            ##############
            # Visualize! #
            ##############   
            gfx.clear_canvas()
            gfx.draw_all_objects(objects) 
            
            # Generate circogram
            N = 36
            horizon = 100
            #
            static_circogram = car1.static_circogram_2(N, objects[1:], horizon)
            dynamic_circogram = car1.dynamic_cicogram_2(static_circogram, alpha_ref, v_ref, seconds=3)
            d1, d2, _, _, _ = static_circogram
            collision = car1.collision_check(d1, d2)
            if collision:
                print("Collision!")
                continue
            #gfx.draw_static_circogram_data(static_circogram, car1)
            gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=2.2, verbose=False)


            gfx.draw_headings(cars, scale=True)
            gfx.draw_centers(cars)
            gfx.update_display()
            #time.sleep(1000)

            ##############
            # Kinematics #
            ##############
            v_ref, alpha_ref = limo.driver.determined_driver(dynamic_circogram, static_circogram, v_ref, alpha_ref, risk_threshold=2.2, stop_threshold = 5, verbose=True)
                
            # Run one cycle
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)
        else:
            v_ref = 0
            alpha_ref = 0

def driving_with_single_determined_driver_2():
    # Create a visualizer
    #gfx, objects, cars = map_tube(scale=1, height=1080, width=1920, pixels_per_unit=10)
    gfx, objects, cars = map_lanes_single(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True)
    car1 = cars[0]

    # Spawn a driver
    alpha_max = 1.2
    v_max = 8 # 8
    v_min = -2 # -4
    agent = Agent(v_max, v_min, alpha_max)
    # Make it a limo!
    limo = Limo(vehicle=car1, driver=agent)

    # Sets certain parameters
    steps = 10000
    # TODO: move these params to the agent!
    alpha_ref = 0
    v_ref = 4 # 4
    collision = False

    # NOTE new ide!
    for i in range(steps):
        if not collision:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Press x button
                    exit()

            
            ##############
            # Visualize! #
            ##############   
            gfx.clear_canvas()
            gfx.draw_all_objects(objects) 
            
            # Generate circogram
            N = 36
            horizon = 100
            #
            static_circogram = car1.static_circogram_2(N, objects[1:], horizon)
            dynamic_circogram = car1.dynamic_cicogram_2(static_circogram, alpha_ref, v_ref, seconds=3)
            #
            d1, d2, _, _, _ = static_circogram
            collision = car1.collision_check(d1, d2)
            if collision:
                print("Collision!")
                continue
            #gfx.draw_static_circogram_data(static_circogram, car1)
            #gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=1, verbose=False)

            gfx.draw_headings(cars, scale=True)
            gfx.draw_centers(cars)
            
            #time.sleep(1000)

            ##############
            # Kinematics #
            ##############
            v_ref, alpha_ref, diff_risk = limo.driver.determined_driver_new_DC(dynamic_circogram, static_circogram, v_ref, alpha_ref, risk_threshold=2, stop_threshold = 10, verbose=True)
            gfx.draw_DC_3_data(diff_risk, static_circogram, risk_threshold=2, verbose=False)    
            # Run one cycle
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)
        else:
            v_ref = 0
            alpha_ref = 0

        gfx.update_display()

def map_tube_multi(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True, dt=0.1):
    # Spawn in the walls:
    vertices = np.array([[5, 5], [5, 80], [150, 80], [150, 5]])
    outer_rim = Object(np.array([0, 0]), vertices=vertices)

    # Some smaller obstacles
    vertices = PointsOnCircum(r=12, n=4, center=(5, 5))
    rim1 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(5, 80))
    rim2 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(150, 80))
    rim3 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(150, 5))
    rim4 = Object(np.array([0, 0]), vertices=vertices)

    # Volkswagen
    car1 = Vehicle(np.array([25, 15]),  length=4, width=2, heading=0,     tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car2 = Vehicle(np.array([125, 15]), length=4, width=2, heading=np.pi, tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car3 = Vehicle(np.array([25, 55]),  length=4, width=2, heading=0,     tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car4 = Vehicle(np.array([125, 55]), length=4, width=2, heading=np.pi, tau_steering=0.4, tau_throttle=0.4, dt=dt)
    # Extra
    car5 = Vehicle(np.array([65, 35]),  length=4, width=2, heading=np.pi,     tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car6 = Vehicle(np.array([85, 35]), length=4, width=2, heading=0,         tau_steering=0.4, tau_throttle=0.4, dt=dt)

    objects = [car1, car2, car3, car4, car5, car6, outer_rim, rim1, rim2, rim3, rim4] #, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1, car2, car3, car4, car5, car6]
    
    if viz:
        MAP_DIMENSIONS = (height*scale, width*scale)
        gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
        return gfx, objects, cars
    else:
        return objects, cars
    

def map_lanes_multi(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True):
    # Spawn in the outer wall:
    vertices = np.array([[5, 5], [5, 105], [180, 105], [180, 5]])
    outer_rim = Object(np.array([0, 0]), vertices=vertices)
    # Walls
    vertices = np.array([[20, 35], [20, 40], [160, 40], [160, 35]])
    wall1 = Object(np.array([0, 0]), vertices=vertices)

    vertices = np.array([[25, 65], [25, 75], [160, 75], [160, 65]])
    wall2 = Object(np.array([0, 0]), vertices=vertices)
    
    # Some smaller obstacles
    vertices = PointsOnCircum(r=12, n=4, center=(5, 5))
    rim1 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(5, 105))
    rim2 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(180, 105))
    rim3 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(180, 5))
    rim4 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=5, n=4, center=(180, 54))
    rim5 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=5, n=4, center=(5, 50))
    rim6 = Object(np.array([0, 0]), vertices=vertices)


    # Spawn in 1 limo-cars
    # Volkswagen
    car1 = Vehicle(np.array([35, 20]),  length=4, width=2, heading=0,     tau_steering=0.4, tau_throttle=0.4) 
    car2 = Vehicle(np.array([145, 20]), length=4, width=2, heading=np.pi, tau_steering=0.4, tau_throttle=0.4) 
    car3 = Vehicle(np.array([15, 50]),  length=4, width=2, heading=0,     tau_steering=0.4, tau_throttle=0.4) 
    car4 = Vehicle(np.array([100, 50]), length=4, width=2, heading=np.pi, tau_steering=0.4, tau_throttle=0.4) 

    obj = [car1, car2, car3, car4, outer_rim, wall1, wall2,  rim5, rim1, rim3, rim6]#, rim2, rim3, rim4, rim5] #, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1, car2, car3, car4]
    if viz:
        MAP_DIMENSIONS = (height*scale, width*scale)
        gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
        return gfx, obj, cars
    else:
        return obj, cars
    
def map_city_block_multi(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True):
    # Spawn in the outer wall:
    vertices = np.array([[5, 5], [5, 105], [145, 105], [145, 5]])
    outer_rim = Object(np.array([0, 0]), vertices=vertices)
    # City blocks
    vertices = np.array([[25, 25], [25, 45], [60, 45], [60, 25]])
    block1 = Object(np.array([0, 0]), vertices=vertices)

    vertices = np.array([[25, 65], [25, 85], [60, 85], [60, 65]])
    block2 = Object(np.array([0, 0]), vertices=vertices)

    vertices = np.array([[90, 25], [90, 45], [125, 45], [125, 25]])
    block3 = Object(np.array([0, 0]), vertices=vertices)

    vertices = np.array([[90, 65], [90, 85], [125, 85], [125, 65]])
    block4 = Object(np.array([0, 0]), vertices=vertices)
    
    # Some smaller obstacles
    vertices = PointsOnCircum(r=12, n=4, center=(5, 5))
    rim1 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(5, 105))
    rim2 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(145, 105))
    rim3 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(145, 5))
    rim4 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=5, n=4, center=(145, 54))



    # Spawn in 1 limo-cars
    # Volkswagen
    car1 = Vehicle(np.array([25, 15]),  length=4, width=2, heading=0,     tau_steering=0.4, tau_throttle=0.4, dt=0.1) 
    car2 = Vehicle(np.array([125, 15]), length=4, width=2, heading=np.pi, tau_steering=0.4, tau_throttle=0.4, dt=0.1) 
    car3 = Vehicle(np.array([25, 55]),  length=4, width=2, heading=0,     tau_steering=0.4, tau_throttle=0.4, dt=0.1) 
    car4 = Vehicle(np.array([125, 55]), length=4, width=2, heading=np.pi, tau_steering=0.4, tau_throttle=0.4, dt=0.1) 

    obj = [car1, car2, car3, car4, outer_rim, block1, block2, block3, block4, rim1, rim2, rim3, rim4]#, rim2, rim3, rim4, rim5] #, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1, car2, car3, car4]
    if viz:
        MAP_DIMENSIONS = (height*scale, width*scale)
        gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
        return gfx, obj, cars
    else:
        return obj, cars

def map_lanes_single(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True, dt=0.1, tau_steering=0.4, tau_throttle=0.4):
    # Spawn in the outer wall:
    vertices = np.array([[5, 5], [5, 105], [180, 105], [180, 5]])
    outer_rim = Object(np.array([0, 0]), vertices=vertices)
    # Walls
    vertices = np.array([[40, 35], [40, 40], [160, 40], [160, 35]])
    wall1 = Object(np.array([0, 0]), vertices=vertices)

    vertices = np.array([[25, 65], [25, 75], [160, 75], [160, 65]])
    wall2 = Object(np.array([0, 0]), vertices=vertices)
    
    # Some smaller obstacles
    vertices = PointsOnCircum(r=12, n=4, center=(5, 5))
    rim1 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(5, 105))
    rim2 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(180, 105))
    rim3 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=12, n=4, center=(180, 5))
    rim4 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=5, n=4, center=(180, 54))
    rim5 = Object(np.array([0, 0]), vertices=vertices)
    vertices = PointsOnCircum(r=5, n=4, center=(5, 50))
    rim6 = Object(np.array([0, 0]), vertices=vertices)


    # Spawn in 1 limo-cars
    car1 = Vehicle(np.array([35, 20]),  length=5, width=3, heading=0,     tau_steering=tau_steering, tau_throttle=tau_throttle, dt=dt) 

    obj = [car1, outer_rim, wall1, wall2,  rim5, rim1, rim3, rim6]#, rim2, rim3, rim4, rim5] #, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1]
    if viz:
        MAP_DIMENSIONS = (height*scale, width*scale)
        gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
        return gfx, obj, cars
    else:
        return obj, cars

def driving_with_many_determined_drivers_fishtank():
    # Create a visualizer
    gfx, objects, cars = map_tube_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    
    # Spawn drivers
    alpha_max = 1.2
    v_max = 7 # 6 
    v_min = -2 
    limos = []
    for car in cars:
        agent = Agent(v_max, v_min, alpha_max)
        # Make it a limo!
        limo = Limo(vehicle=car, driver=agent)
        limos.append(limo)

    # Sets certain parameters
    steps = 10000
    # TODO: move these params to the agent?
    alpha_refs = np.zeros(len(cars))
    v_refs = np.ones(len(cars))*2
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Press x button
                exit()

        ##############
        # Visualize! #
        ##############   
        gfx.clear_canvas()
        gfx.draw_all_objects(objects) 
        
        # Generate circogram
        N = 36
        horizon = 500
        # Circograms!
        DCs = []
        SCs = []
        for n, car in enumerate(cars):
            static_circogram = car.static_circogram_2(N, objects[0:n]+objects[n+1:], horizon)
            dynamic_circogram = car.dynamic_cicogram_2(static_circogram, alpha_refs[n], v_refs[n], seconds=3)
            d1, d2, _, _, _ = static_circogram
            car.collision_check(d1, d2)
            #gfx.draw_static_circogram_data(static_circogram, car)
            gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=1, verbose=False)
            DCs.append(dynamic_circogram)
            SCs.append(static_circogram)


        gfx.draw_headings(cars, scale=True)
        gfx.draw_centers(cars)
        gfx.update_display()

        ##############
        # Kinematics #
        ##############
        for n, limo in enumerate(limos):
            if not limo.vehicle.collided:
                v_refs[n], alpha_refs[n] = limo.driver.determined_driver(DCs[n], SCs[n], v_refs[n], alpha_refs[n], risk_threshold=1, stop_threshold = 3.5,  dist_wait=1, verbose=False)
                #v_refs[n], alpha_refs[n] = limo.driver.determined_driver(DCs[n], SCs[n], v_refs[n], alpha_refs[n], risk_threshold= 1.5, stop_threshold = 4,  dist_wait=10, verbose=False)
            else:
                v_refs[n], alpha_refs[n] = (0, 0)
            # Run one step
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_refs[n], v_ref=v_refs[n])

def driving_with_many_determined_drivers():
    # Create a visualizer

    #gfx, objects, cars = map_circle_multi(scale=1, height=1200, width=2000, pixels_per_unit=10)
    #gfx, objects, cars = map_tube_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    #gfx, objects, cars = map_lanes_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    gfx, objects, cars = map_city_block_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    
    # Spawn drivers
    alpha_max = 1.2 # Volkswagen
    v_max = 7 # 6 
    v_min = -2 
    limos = []
    for car in cars:
        agent = Agent(v_max, v_min, alpha_max)
        # Make it a limo!
        limo = Limo(vehicle=car, driver=agent)
        limos.append(limo)

    # Sets certain parameters
    steps = 1000
    # TODO: move these params to the agent?
    alpha_refs = np.zeros(len(cars))
    v_refs = np.ones(len(cars))*2
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Press x button
                exit()

        ##############
        # Visualize! #
        ##############   
        gfx.clear_canvas()
        gfx.draw_all_objects(objects) 
        
        # Generate circogram
        N = 36
        horizon = 500
        # Circograms!
        DCs = []
        SCs = []
        for n, car in enumerate(cars):
            static_circogram = car.static_circogram_2(N, objects[0:n]+objects[n+1:], horizon)
            dynamic_circogram = car.dynamic_cicogram_2(static_circogram, alpha_refs[n], v_refs[n], seconds=3)
            d1, d2, _, _, _ = static_circogram
            car.collision_check(d1, d2)
            #gfx.draw_static_circogram_data(static_circogram, car)
            #gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=1.5, verbose=False)
            DCs.append(dynamic_circogram)
            SCs.append(static_circogram)


        gfx.draw_headings(cars, scale=True)
        gfx.draw_centers(cars)
        gfx.update_display()

        ##############
        # Kinematics #
        ##############
        for n, limo in enumerate(limos):
            if not limo.vehicle.collided:
                v_refs[n], alpha_refs[n] = limo.driver.determined_driver(DCs[n], SCs[n], v_refs[n], alpha_refs[n], risk_threshold= 1.5, stop_threshold = 4,  dist_wait=10, verbose=False)
            else:
                v_refs[n], alpha_refs[n] = (0, 0)
            # Run one step
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_refs[n], v_ref=v_refs[n])

def map_circle_multi(scale=1, height=1080, width=1920, pixels_per_unit=10, viz=True, dt=0.1):
    # Spawn in the walls:
    #vertices = PointsOnCircum(r=100, n=50, center=(75, 55))

    vertices = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
    outer_rim = Object(np.array([0, 0]), vertices=vertices)

    # Volkswagen
    car1 = Vehicle(np.array([35, 20]),  length=8, width=4, heading=0,     tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car2 = Vehicle(np.array([65,20]), length=8, width=4, heading=np.pi, tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car3 = Vehicle(np.array([25, 55]),  length=8, width=4, heading=0,     tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car4 = Vehicle(np.array([85,55]), length=8, width=4, heading=np.pi, tau_steering=0.4, tau_throttle=0.4, dt=dt)
    # Extra
    car5 = Vehicle(np.array([65, 35]),  length=8, width=4, heading=np.pi,     tau_steering=0.4, tau_throttle=0.4, dt=dt) 
    car6 = Vehicle(np.array([85, 35]),  length=8, width=4, heading=0,         tau_steering=0.4, tau_throttle=0.4, dt=dt)

    objects = [car1, car2, car3, car4, car5, car6, outer_rim] #, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]#, wall10]
    cars = [car1, car2, car3, car4, car5, car6]
    
    if viz:
        MAP_DIMENSIONS = (height*scale, width*scale)
        gfx = Visualization(MAP_DIMENSIONS, pixels_per_unit=pixels_per_unit, map_img_path="graphics/test_map_2.png") # Also initializes the display
        return gfx, objects, cars
    else:
        return objects, cars

def driving_with_many_boats():
    # Create a visualizer
    divider = 10
    dt = 1/divider
    gfx, objects, cars = map_circle_multi(scale=1, height=1200, width=2200, pixels_per_unit=10, dt=dt)
    #gfx, objects, cars = map_tube_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    #gfx, objects, cars = map_lanes_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    #gfx, objects, cars = map_city_block_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    
    # Spawn drivers
    alpha_max = 0.8 # boat
    v_max = 7 # 6 
    v_min = -2 
    limos = []
    for car in cars:
        agent = Agent(v_max, v_min, alpha_max)
        # Make it a limo!
        limo = Limo(vehicle=car, driver=agent)
        limos.append(limo)

    # Sets certain parameters
    steps = 8000
    # TODO: move these params to the agent?
    alpha_refs = np.zeros(len(cars))
    v_refs = np.ones(len(cars))*2
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Press x button
                exit()

        ##############
        # Visualize! #
        ##############   
        gfx.clear_canvas()
        gfx.draw_all_objects(objects) 
        
        # Generate circogram
        N = 36
        horizon = 500
        # Circograms!
        DCs = []
        SCs = []
        if i %(divider//6) ==0: # Only every 0.1 seconds
            for n, car in enumerate(cars):
                static_circogram = car.static_circogram_2(N, objects[0:n]+objects[n+1:], horizon)
                dynamic_circogram = car.dynamic_cicogram_2(static_circogram, alpha_refs[n], v_refs[n], seconds=3)
                d1, d2, _, _, _ = static_circogram
                car.collision_check(d1, d2)
                #gfx.draw_static_circogram_data(static_circogram, car)
                #gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=1.5, verbose=False)
                DCs.append(dynamic_circogram)
                SCs.append(static_circogram)
        
            for n, limo in enumerate(limos):
                if not limo.vehicle.collided:
                    v_refs[n], alpha_refs[n] = limo.driver.determined_driver(DCs[n], SCs[n], v_refs[n], alpha_refs[n], risk_threshold=0.4, stop_threshold = 2,  dist_wait=10, verbose=False)
                else:
                    v_refs[n], alpha_refs[n] = (0, 0)

        ##############
        # Kinematics #
        ##############
        for n, limo in enumerate(limos):
            # Run one step
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_refs[n], v_ref=v_refs[n])

        gfx.draw_headings(cars, scale=True)
        gfx.draw_centers(cars)
        gfx.update_display()



if __name__ == "__main__":
    """
    Visualizing circograms:
    """

    #dynamic_circogram_test()
    #still_circogram_test()
    #driving_with_single_random_driver()
    #driving_with_multiple_random_drivers()
    #driving_with_multiple_random_drivers_maze()

    #driving_with_single_determined_driver()
    #driving_with_many_determined_drivers()

    driving_with_many_boats()


    





    