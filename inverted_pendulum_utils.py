import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

class PendulumAnimator:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.x = trajectory[:, 0]
        self.y = trajectory[:, 1]
        self.t = trajectory[:, 2]
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.trail, = self.ax.plot([], [], 'r-', alpha=0.5)
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

    def init(self):
        self.line.set_data([], [])
        self.trail.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.trail, self.time_text

    def animate(self, i):
        self.line.set_data([0, self.x[i]], [0, self.y[i]])
        self.trail.set_data(self.x[max(0,i-10):i+1], self.y[max(0,i-10):i+1])
        self.time_text.set_text(f'Time = {int(self.t[i])}')
        return self.line, self.trail, self.time_text

    def save_gif(self, filename='pendulum.gif', fps=30):
        ani = animation.FuncAnimation(
            self.fig, self.animate, frames=len(self.x),
            init_func=self.init, blit=True
        )
        ani.save(filename, writer='pillow', fps=fps)
        print(f"GIF saved as {filename}")

    def show(self):
        ani = animation.FuncAnimation(
            self.fig, self.animate, frames=len(self.x),
            init_func=self.init, blit=True
        )
        return ani

def get_theta_from_state(state):
            """
            Given a state where state[0]=cos(theta), state[1]=sin(theta),
            return theta in radians in the range [-pi, pi].
            """
            return np.arctan2(state[1], state[0])

def get_pendulum_xy(state):
    # For classic gym pendulum: state[0]=cos(theta), state[1]=sin(theta)
    x = state[1]
    y = state[0]
    return x, y

def collect_trajectory(env, model, max_steps=500):
    trajectory = np.zeros((max_steps, 3))
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        x, y = get_pendulum_xy(obs)
        trajectory[step] = [x, y, step]
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        step += 1
    return trajectory[:step]

def collect_info(env, model, max_steps=500):
    u_record = np.zeros(max_steps)
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        x, y = get_pendulum_xy(obs)
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info:
            u_record[step] = info["action"]
        else:
            print('no info provided')
        done = terminated or truncated
        step += 1
    return u_record[:step]

def plot_actions(u_record, save=False, filename ='u_record.png'):
        plt.figure(figsize=(8, 4))
        plt.plot(u_record, marker='o', linestyle='-')
        plt.xlabel('Timestep')
        plt.ylabel('Action')
        plt.title('Action at Each Timestep')
        plt.grid(True)
        plt.tight_layout()

        if save==True:
            plt.savefig(filename)
        else:
            plt.show()


def train_rl_agent(env, total_timesteps=50000, **ppo_kwargs):
    import stable_baselines3 as sb3
    model = sb3.PPO('MlpPolicy', env, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model

def test_model(env, model):
    # Reset environment
    obs, info = env.reset()
    ep_len = 0
    ep_rew = 0

    # Run episode until complete
    while True:
        # Provide observation to policy to predict the next action
        action, _ = model.predict(obs)
        # Perform action, update total reward
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += reward
        # Increase step counter
        ep_len += 1
        # Check to see if episode has ended
        if terminated or truncated:
            return ep_len, ep_rew/ep_len
        
def plot_phase_streamlines(pendulum, var_ranges=[(-np.pi, np.pi), (-5, 5)], density=20):
    """
    Plot streamlines of the phase space for the single pendulum (theta, omega).
    var_ranges: [(theta_min, theta_max), (omega_min, omega_max)]
    """
    theta_range = np.linspace(var_ranges[0][0], var_ranges[0][1], density)
    omega_range = np.linspace(var_ranges[1][0], var_ranges[1][1], density)
    Theta, Omega = np.meshgrid(theta_range, omega_range)
    dTheta = np.zeros_like(Theta)
    dOmega = np.zeros_like(Omega)

    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            state = [Theta[i, j], Omega[i, j]]
            derivs = pendulum.derivatives(state, 0)
            dTheta[i, j] = derivs[0]
            dOmega[i, j] = derivs[1]

    plt.figure(figsize=(8, 6))
    plt.streamplot(Theta, Omega, dTheta, dOmega, color='b', density=1.2)
    plt.xlabel('Theta')
    plt.ylabel('Omega')
    plt.title('Phase Space: Omega vs Theta')
    plt.grid()
    plt.tight_layout()
    plt.show()