from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium import utils


DEFAULT_X = 0.0
DEFAULT_Y = 0.0

class InvertedPendulumTorqueEnv(gym.Env):
    """
    ##Adapted from the Built-in Gym Environment

    # balance pendulum at the unstable maximum. The intial position is the top with a small perturbation
    # penalize if the pendulum falls down or if the angle is too far from the upright position.
    

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    theta = 0 at the top, theta = pi at the bottom
    so the further away from 0 theta gets, the higher the cost.
    also, we'll penalize how much torque is applied.
    *r = (-theta**2  + 0.001*torque**2))*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).


    ## Starting State

    The starting state is a random angle in theta=0 +/- eta, where eta is randomly chosen  
    and a random angular velocity in *[-zeta,zeta]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: .
    - `eta`: perturbation in angle.
    - `zeta`: perturbation in angular velocity.

    """

    def __init__(self, g=10.0, eta=0.05,zeta=0.05, max_steps = 200):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.eta = eta # peturbation in angle
        self.zeta = zeta #perturbation in angular velocity
        self.screen = None
        self.clock = None
        self.isopen = True

        self.max_steps = max_steps
        

        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # observation space is a 3D vector with x, y and angular velocity
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        #print("th, thdot", th, thdot)

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        costs = angle_normalize(th) ** 2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        self.steps_left -= 1
        done = self.steps_left <= 0
        

        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, done, False, {}
    
    # fancy reset function

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            theta = options.get("theta_init") if "theta_init" in options else 0.0
            # apply perturbation to theta
            theta += self.np_random.uniform(-self.eta, self.eta)


            thetadot = options.get("thetadot_init") if "thetadot_init" in options else 0.0  
            # apply perturbation to thetadot
            thetadot += self.np_random.uniform(-self.zeta, self.zeta)

            print('theta, thetadot', theta, thetadot)
       


            x = np.cos(theta)
            y = np.sin(theta)   

            print ("x,y", x,y)
 

            #x = options.get("x_init") if "x_init" in options else DEFAULT_X
            #y = options.get("y_init") if "y_init" in options else DEFAULT_Y

        
            #x = utils.verify_number_and_cast(x)
            #y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        self.steps_left = self.max_steps

        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    
    def close(self):
        pass

class InvertedPendulumEnv(gym.Env):
    """
    ##Adapted from the Built-in Gym Environment

    # balance pendulum at the unstable maximum. The intial position is the top with a small perturbation
    # penalize if the pendulum falls down or if the angle is too far from the upright position.
    

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `u`: Horizontal force applied at the bottom of the pendulum

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | u      | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    theta = 0 at the top, theta = pi at the bottom
    so the further away from 0 theta gets, the higher the cost.
    also, we'll penalize how much torque is applied.
    *r = -theta**2 

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).


    ## Starting State

    The starting state is a random angle in theta=0 +/- eta, where eta is randomly chosen  
    and a random angular velocity in *[-zeta,zeta]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `c`: : damping coefficient
    - `eta`: perturbation in angle.
    - `zeta`: perturbation in angular velocity.

    """

    def __init__(self, c=0.1, eta=0.05,zeta=0.02, dt=0.05, max_steps = 200):
        self.max_speed = 8
        self.max_force = 8.0
        self.dt = dt
        self.c = 1

        self.eta = eta # peturbation in angle
        self.zeta = zeta #perturbation in angular velocity
        self.screen = None
        self.clock = None
        self.isopen = True

        self.max_steps = max_steps
        

        self.action_space = spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32
        )

        # observation space is a 3D vector with x, y and angular velocity
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.c
        dt = self.dt

        # penalizes deviation from upright (theta=0) and optionally penalizes angular velocity and control effort.
        # Example: 
        # use squares to ensure positive
        if u == None: # apply analytically obtained feedback law
            costs= angle_normalize(th)**2 + 0.1 * thdot ** 2 
        else:
            costs = angle_normalize(th)**2 + 0.1 * thdot ** 2 + 0.02 * (u**2)
        # For just uprightness:
        #costs = angle_normalize(th) ** 2


        # Euler approximation...
        if u == None: # apply analytically obtained feedback law
            a = 0.7
            u = -2*a*np.sin(th)-thdot*np.cos(th)
        else:
            u = np.clip(u, -self.max_force, self.max_force)[0]

        self.last_u = u  # for info
        newthdot = thdot + (np.sin(th) - self.c*thdot + u*np.cos(th)) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        self.steps_left -= 1
        done = self.steps_left <= 0

        info = self._get_info()
       

        return self._get_obs(), -costs, done, False, info
    
    # reset function
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
   
 
        # apply perturbation to theta
        self.th = 0.0 + self.np_random.uniform(-self.eta, self.eta)
        self.thdot = 0.0 + self.np_random.uniform(-self.zeta, self.zeta)
        self.state = self.th, self.thdot
        self.last_u = None
        self.steps_left = self.max_steps


        x = np.cos(self.th)
        y = np.sin(self.th)   
        #print ("x,y", x,y)
       

        info = self._get_info()

        return self._get_obs(), info
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "action": self.last_u 
            
            }

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    
    def close(self):
        pass
    
def angle_normalize(x): # wrap angle between -pi and pi
    return ((x + np.pi) % (2 * np.pi)) - np.pi
