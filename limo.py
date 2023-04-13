from Vehicle import Vehicle
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

class Limo(Vehicle):

    def __init__(self, vehicle : Vehicle, driver : Agent ):
        self.vehicle = vehicle
        self.driver = driver




if __name__ == "__main__":
    # Driver
    alpha_max = 0.8
    v_max = 5
    var_alpha= 0.2 # 0.3
    var_vel= 0.5 # 0.5
    agent = Agent(v_max, alpha_max, var_alpha, var_vel)
    # Vehicle
    dt = 0.1 
    alpha_ref = 0
    v_ref = 0
    car = Vehicle(np.array([25, 25]), length=4, width=2, heading=np.pi/2, dt=0.1)
    # Limo
    limo = Limo(vehicle=car, driver=agent)
    # Bookkeeping
    steps = 1000
    states = np.zeros((4, 1, steps))
    alphas = np.zeros((steps,))
    v_refs = np.ones((steps,))*v_ref
    alpha_refs = np.ones((steps,))*alpha_ref
    psis = np.zeros((steps,))
    # Run for steps
    for i in range(steps):
        # Check for new random motion every second.
        if (i*limo.vehicle.dt).is_integer: # Checks only for new commands on whole seconds
            v_ref, alpha_ref = limo.driver.brownian_action(v_ref, alpha_ref, r_factor=0.02)
        
        # Book-keeping
        alphas[i] = limo.vehicle.alpha
        alpha_refs[i] = alpha_ref
        v_refs[i]= v_ref
        states[:, :, i] = limo.vehicle.X
        psis[i] = limo.vehicle.heading

        # Run one cycle
        limo.vehicle.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)
    
    # Plotting
    time_axis = np.linspace(0, steps*dt-dt, steps)
    Xs = states[0, :].T
    Ys = states[1, :].T
    Vs = states[2:, :]
    abs_speeds = np.linalg.norm(Vs.T, axis=2)

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
