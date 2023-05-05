### Some notes to guide future RL training
**Idea list**
- Enable setting vmax, vmin, alpha_max from environment constructor- and actually use it in the "reset" function!
- *Maybe* the speed and steering angle in the observation space should be given by -1->1, instead of actual values, or maybe not.
- TODO: v20 make sure goals do not spawn inside pillars. Maybe KISS by adding some pre-set goal locations to pick from.
- TODO: only feed SC that are in the "actual" moving direction - should speed up
- TODO: run a test run to see if the inputs the agent is getting makes sense (especially velocity and SC)
- TODO: Increase complexity in ANN! Now that env is getting progressively harder.
- TODO: make a function for loading and keep training an already implemented model. Make it faster to adapt to new scenarios.
    -> "RESIDUAL TRAINING": will probably be the key to success when RL + MPC!
- TODO: make sure to set a flag to remove / scale back on OAnoice for non training scenarios!
- TODO: Dynamic obstacles! Other car driving about!


#### Environment v00
- In this env, the input is only x, y and heading (WCF) - and then a rewarded distance functions is used to push the driver to the goal.
- Reversing is not implemented.
- The env. is still solved with 95% hitrate, with random spawn, within +-50 meters of the goal.

- Problems: 
    - Not knowing speed and turnrate (which are the inputs!) makes it very jerky in its movement.
    - Especially in combination with a 0.5 second decision interval, it looked like a poorely tuned P-controller.

##### Environment v01
- Added normed velocity (directional), and turn rate (current, not reference) to the state space.
- Also included reversing. Expecting smoother driving, and maybe some reversing to improve score.
- However, Some bugs have been spotted near 0 speed; where turn-rate => inf.! Be warned...


#### Environment v10
- Make an environment, wher all data is from driver's POV (CCF) (so only need to give him speed, turn rate and goal pose in driver CF of course)!
- This will also make it easier to add static circograms as well, as they are already given in CCF!
- All that is required is to feed the goal pose, in CCF! Then, heading and all other values are irrelevant.

##### Environment v20
- Static circogram addition-> Put the driver in a box/tunnel, with only a few exits so it needs to reverse/navigate out before reaching the goal.
- v20: small blocks scattered to add obstacles, random spawning goal point.
    - v20_2: Trained with [episodes=10000, sim_dt=0.1, decision_dt=0.1, v_max=25, v_min=-12, alpha_max=0.3, tau_steering=0.7, tau_throttle=0.7] => Really kills the environment!
    - Also N=36 rays.
- v21: Added walls blocking in all directions. Still gets stuck/ confused often
- v21_2: increase dt = 0.5 second; as 0.1 is not enough to discover ways out of local optimas! Adding higher dt is similar to adding momentum!
    - sim_dt=0.05, decision_dt=0.5, v_max=25, v_min=-8, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5
    - For some reason: always prefer to reverse!....
    - Also added N=36 static rays for better vision.


##### V40
- FW environment! v41_IRL_Fw: Some bug made it look bad (looking at score): but only collided 856 times after 50.000 episodes!
    - Used settings: "collision_rejection=True", to reject colliding trajectories.
    - Why would it still collide? It would still collide if proposedtrajectory stopped right in front of wall, so when new halu came along, nothing to do - due to sliding.
    - Compared to v22: (waiting)