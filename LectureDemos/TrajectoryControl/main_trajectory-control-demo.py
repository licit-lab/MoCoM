"""
main-trajectory-control-demo.py

Comprehensive pedagogical collection of control examples (PID, Q-learning, MPC)
for vehicle and single-intersection traffic-light control with two approaches
(North-South and East-West).

Vehicle trajectory:
    - PID controller (position-only)
    - Tabular Q-learning (trajectory following)
    - MPC (linear, cvxpy)

Defaults and notes:
  - Discretizations and hyperparameters chosen for pedagogical clarity.
  - Requires: numpy, matplotlib, cvxpy
    pip install numpy matplotlib cvxpy

Run:
  python main-trajectory-control-demo.py

Author: P-A LAHAROTTE (2025),
with support by ChatGPT (English comments)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from utils import pid_tuning
from utils.plotting_helper import plot_time_series
import pathlib


# MPC requires cvxpy; wrap import so file can still be inspected without it
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False

# -----------------------------
# Setting Up your environment
# -----------------------------
LocalDirectory = str(pathlib.Path(__file__).parent.resolve())
os.chdir(LocalDirectory)

FigureDirectory = "./_img/TrajectoryControl/"
if not os.path.exists(FigureDirectory):
    os.makedirs(FigureDirectory)



# -----------------------------
# VEHICLE: PID, MPC, Q-learning
# -----------------------------

#######################################
# 0/ Reference signal
#######################################
# * Exhaustive smooth signal - manually tuned signal, but not representative from what we provide in the real-world
def build_reference_signal_advanced(dt=0.1, Tsim=800):
    # Compute a realistic reference position for a vehicle
    t = np.arange(Tsim)
    # x_ref = 0.5 * (1 - np.cos(0.05 * t)) * 20
    smoothing_factor = 1.2
    x_ref = (1 + np.tanh( 0.1 * dt * (-int(smoothing_factor*Tsim/4) + t ))) * 50 + (1 + np.tanh( 0.125 * dt * (-int(5*smoothing_factor*Tsim/8) + t )))*100
    return x_ref

x_ref_test = build_reference_signal_advanced()
plot_time_series(t=np.arange(800), series = [x_ref_test], labels='x_ref', title='Reference Signal', ylabel='Position (m)')


# * Usual raw signal provided as input
def build_reference_signal_simple(dt=0.1, Tsim=800):
    # Compute a simple piecewise linear reference position for a vehicle
    t = np.arange(Tsim)
    t_warning = int(np.trunc(Tsim/2))
    # Build up the Piecewise linear function
    sliced_time_part1 = np.zeros(Tsim)
    sliced_time_part1[:t_warning-100] = t[:t_warning-100]
    #print(sliced_time_part1)
    sliced_time_part2 = np.zeros(Tsim)
    sliced_time_part2[t_warning-100:t_warning+100] = t[t_warning-100:t_warning+100]
    #print(sliced_time_part2)
    sliced_time_part3 = np.zeros(Tsim)
    sliced_time_part3[t_warning+100:Tsim-100] = t[t_warning+100:Tsim-100]
    #print(sliced_time_part3)
    sliced_time_part4 = np.zeros(Tsim)
    sliced_time_part4[Tsim-100:] = t[Tsim-100:]
    #print(sliced_time_part4)
    x_ref = (
        0.33*(sliced_time_part1) + 
        (0 * (sliced_time_part2)  + (sliced_time_part2>0) * ( - 0 * (t_warning-100) + 100) ) +
        (1 * (sliced_time_part3) + (sliced_time_part3>0) * ( - 1 * (t_warning+100) + 100) ) +
        (0 * (sliced_time_part4 - (Tsim-100)) + 300*(sliced_time_part4>0))
    )
    return x_ref

x_ref_test2 = build_reference_signal_simple()
plot_time_series(np.arange(800), [x_ref_test2], 'x_ref', title='Reference Signal', ylabel='Position (m)')


#######################################
# 1) PID controller (position-only)
#######################################

# * Set up the directory
FigureDirectoryPID = "./_img/TrajectoryControl/PID/"
if not os.path.exists(FigureDirectoryPID):
    os.makedirs(FigureDirectoryPID)

# * Model the problem in the reactive PID Framework
def pid_vehicle(dt=0.1, Tsim=800, BoolPrint=False, BoolSave=False, ref_function = build_reference_signal_simple, Kp=2.0, Ki=0.1, Kd=0.5):
    """Simple PID on position error controlling acceleration."""
    t = np.arange(Tsim)
    # x_ref = 0.5 * (1 - np.cos(0.05 * t)) * 20
    x_ref = ref_function(dt, Tsim)

    # PID gains
    #Kp, Ki, Kd = 2.0, 0.1, 0.5 # 2.0, 0.1, 0.5

    a_min, a_max = -3.0, 2.0
    v_min, v_max = 0.0, 30.0

    x = 0.0; v = 0.0

    integral = 0.0
    e_prev = x_ref[0] - x

    x_log = np.zeros(Tsim+1)
    v_log = np.zeros(Tsim+1)
    a_log = np.zeros(Tsim)
    e_log = np.zeros(Tsim)

    x_log[0]=x; v_log[0]=v

    # TEMPORAL LOOP ALONG THE SIMULATION
    for k in range(Tsim):
        # Compute the error terms
        e = x_ref[k] - x 
        integral += e * dt
        derivative = (e - e_prev) / dt
        e_prev = e

        e_log[k] = e # for log  purpose
        # Compute the actuator signals
        a = Kp*e + Ki*integral + Kd*derivative
        a = np.clip(a, a_min, a_max)
        # Update the decision and the environment (dynamic system)
        v = v + a * dt
        v = np.clip(v, v_min, v_max)
        x = x + v * dt
        x_log[k+1]=x; v_log[k+1]=v; a_log[k]=a

    # print(Tsim, np.size(x_log), np.size(x_ref))
    if BoolSave:
        plot_time_series(np.arange(Tsim), [x_log[1:], x_ref], ['x','x_ref'], "PID Vehicle: Position-kp-"+str(Kp)+"-ki-"+str(Ki)+"-kd-"+str(Kd), 'Position (m)', filename=FigureDirectoryPID+"PID-TrajectoryTracking-position-kp-"+str(Kp)+"-ki-"+str(Ki)+"-kd-"+str(Kd)+".png")
        plot_time_series(np.arange(Tsim), [v_log[1:]], ['v'], 'PID Vehicle: Velocity', 'Velocity (m/s)', filename=FigureDirectoryPID+"PID-TrajectoryTracking-speed-kp-"+str(Kp)+"-ki-"+str(Ki)+"-kd-"+str(Kd)+".png")
        plot_time_series(np.arange(Tsim), [a_log], ['a'], 'PID Vehicle: Acceleration', 'Acceleration (m/s^2)', filename=FigureDirectoryPID+"PID-TrajectoryTracking-acceleration-kp-"+str(Kp)+"-ki-"+str(Ki)+"-kd-"+str(Kd)+".png")
    elif BoolPrint: 
        plot_time_series(np.arange(Tsim), [x_log[1:], x_ref], ['x','x_ref'], "PID Vehicle: Position-kp-"+str(Kp)+"-ki-"+str(Ki)+"-kd-"+str(Kd), 'Position (m)')
        plot_time_series(np.arange(Tsim), [v_log[1:]], ['v'], 'PID Vehicle: Velocity', 'Velocity (m/s)')
        plot_time_series(np.arange(Tsim), [a_log], ['a'], 'PID Vehicle: Acceleration', 'Acceleration (m/s^2)')

    return x_log, v_log, a_log, e_log, x_ref
    

## * Manual Tests
pid_vehicle(Kp=1, Ki=1, Kd=1, BoolSave=True)
pid_vehicle(Kp=2, Ki=1, Kd=1, BoolSave=True)
pid_vehicle(Kp=3, Ki=1, Kd=1, BoolSave=True)
pid_vehicle(Kp=10, Ki=1, Kd=1, BoolSave=True)
pid_vehicle(Kp=1, Ki=0.1, Kd=1, BoolSave=True)
pid_vehicle(Kp=1, Ki=0.1, Kd=0.1, BoolSave=True)

## Automated Tuning / Calibration of the P,I,D parameters
# refer to - https://simonebertonilab.com/pid-tuning-the-power-of-python/
pid_tuning.grid_search_tuning(
    Kp_range=np.arange(0, 10, 1),
    Ki_range=np.arange(0, 1, 0.1), 
    Kd_range=np.arange(0, 5, 0.1),
    ref_function = build_reference_signal_simple, 
    simulate_pid=pid_vehicle)

pid_vehicle(Kp=1, Ki=0, Kd=0.8, BoolSave=True)


pid_tuning.gradient_based_tuning(
    ref_function = build_reference_signal_simple, 
    simulate_pid=pid_vehicle, 
    initial_guess=[2, 0.1, 1]
    )

pid_vehicle(Kp=0.1566, Ki=0.06, Kd=0.67, BoolSave=True)


bounds_test = [(0,10), (0, 1), (0, 5)]
pid_tuning.evolutionary_tuning(
    ref_function = build_reference_signal_simple, 
    simulate_pid=pid_vehicle, 
    bounds=bounds_test
)

pid_vehicle(Kp=4.97, Ki=0.51, Kd=4.89, BoolSave=True)







#######################################
# 2) MPC for vehicle (linear MPC with cvxpy)
#######################################

# * Set up the directory
FigureDirectoryMPC = "./_img/TrajectoryControl/MPC/"
if not os.path.exists(FigureDirectoryMPC):
    os.makedirs(FigureDirectoryMPC)

# * Model the problem in the Model Predictive Control Framework
def mpc_vehicle(dt=0.1, Tsim=800, N=15, Q = np.diag([10.0,1.0]), R = 0.1, BoolPrint=False, ref_function = build_reference_signal_simple):
    if not CVXPY_AVAILABLE:
        print('cvxpy not available: skip mpc_vehicle')
        return
    A = np.array([[1.0, dt],[0.0,1.0]]) # Matrix for speed and position equation
    B = np.array([[0.0],[dt]])
    #Q = np.diag([10.0,1.0]) # Parameter to set up / calibrate - weight given to the error/gap in the objctive function
    #R = 0.1 # Parameter to set up / calibrate - weight given to the actuator range in the objective function
    a_min, a_max = -3.0, 2.0
    v_min, v_max = 0.0, 30.0
    #t = np.arange(Tsim)
    #x_ref_traj = 0.5 * (1 - np.cos(0.05 * t)) * 20
    x_ref_traj = ref_function(dt, Tsim=Tsim)
    v_ref_traj = np.gradient(x_ref_traj, dt)

    x = np.array([0.0,0.0])
    x_log = np.zeros((Tsim+1, 2))
    u_log = np.zeros(Tsim)
    x_log[0,:]=x

    for k in range(Tsim):
        #
        xr = np.zeros((N,2))
        for i in range(N):
            idx = min(k+i, Tsim-1)
            xr[i, 0] = x_ref_traj[idx] # Reference position
            xr[i, 1] = v_ref_traj[idx] # Reference Speed
        # Set up the optimisation process in CVXPY environment (solver)
        U = cp.Variable(N) # actuator value we are seeking for (over an horizon N, so N values)
        X = cp.Variable((N+1,2))
        cost = 0
        constraints = [X[0,:]==x]
        for i in range(N): # loop on the prediction horizon N
            constraints += [X[i+1,:] == A @ X[i,:] + B.flatten()*U[i]] # ( @ is equivalent to np.dot() )
            constraints += [U[i] >= a_min, U[i] <= a_max]
            constraints += [X[i+1,1] >= v_min, X[i+1,1] <= v_max]
            dx = X[i+1,:] - xr[i,:]
            cost += cp.quad_form(dx, Q) + R*cp.square(U[i]) # design of the objective function: minimum action on the accéleration (ie minimize the actuator U) and minimum error with the reference
        dxN = X[N,:] - xr[-1,:]
        cost += cp.quad_form(dxN, Q) # Fina update of the Objective Function
        prob = cp.Problem(cp.Minimize(cost), constraints) # Send the problem to the solver
        # Compute the optimal solution over an horizon of N
        prob.solve(solver=cp.OSQP, warm_start=True)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print('MPC vehicle solver status', prob.status)
        # Apply the actuation value to the current speed value
        u_apply = float(U.value[0]) # Only use the first value (among the N) as the actuator
        x = A @ x + B.flatten()*u_apply # Update position and speed
        x_log[k+1,:]=x # store the log of the position and speed
        u_log[k]=u_apply # store the log of the actuator (acceleration)

    if BoolPrint:
        plot_time_series(np.arange(Tsim), [x_log[1:,0], x_ref_traj], ['x','x_ref'], 'MPC Vehicle: Position', 'Position (m)', filename = FigureDirectoryMPC+"MPC-trajectory-position.png")
        plot_time_series(np.arange(Tsim), [x_log[1:,1], v_ref_traj], ['v','v_ref'], 'MPC Vehicle: Velocity', 'Velocity (m/s)', filename = FigureDirectoryMPC+"MPC-trajectory-speed.png")
        plot_time_series(np.arange(Tsim), [u_log], ['a'], 'MPC Vehicle: Acceleration', 'Acceleration (m/s^2)', filename = FigureDirectoryMPC+"MPC-trajectory-acceleration.png")
    else:
        plot_time_series(np.arange(Tsim), [x_log[1:,0], x_ref_traj], ['x','x_ref'], 'MPC Vehicle: Position', 'Position (m)')
        plot_time_series(np.arange(Tsim), [x_log[1:,1], v_ref_traj], ['v','v_ref'], 'MPC Vehicle: Velocity', 'Velocity (m/s)')
        plot_time_series(np.arange(Tsim), [u_log], ['a'], 'MPC Vehicle: Acceleration', 'Acceleration (m/s^2)')


# * Test
mpc_vehicle(BoolPrint=True)






#######################################
# 3) Tabular Q-learning for vehicle
#######################################

# * Set up the directory
FigureDirectoryRL = "./_img/TrajectoryControl/RL/"
if not os.path.exists(FigureDirectoryRL):
    os.makedirs(FigureDirectoryRL)

# * Model the problem in the Q-Leaning Framework
def qlearning_vehicle(n_episodes=5000, dt=0.1, T_episode = 800, BoolPrint=False, reward_weight = 0.99, ref_function = build_reference_signal_simple):
    """
    Tabular Q-learning to follow a reference trajectory.
    Discretized state: (position, velocity) bins.
    """

    pos_min, pos_max, pos_bin = 0.0, 320.0, 20.0
    vel_min, vel_max, vel_bin = 0.0, 11.0, 1.0 # 0.0, 20.0, 2.5
    n_pos = int((pos_max-pos_min)/pos_bin)+1
    n_vel = int((vel_max-vel_min)/vel_bin)+1

    actions = np.array([-2.0, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0])
    n_actions = len(actions)

    Q = np.zeros((n_pos, n_vel, n_actions))

    # t = np.arange(T_episode) 
    # x_ref_traj = 0.5 * (1 - np.cos(0.05 * t)) * 20
    x_ref_traj = ref_function(dt, Tsim=T_episode)
    v_ref_traj = np.gradient(x_ref_traj, dt)

    # Parameters to set up
    #alpha, gamma = 0.1, 0.8 # 0.1, 0.95
    alpha = 0.2 # weight between the current Q and the updated Q
    gamma = 0.6 # weight of the future expected cumulated rewards (Q) with respect to the instantaneous reward
    eps = 1.0; eps_end = 0.05; eps_decay = 0.999 # 0.995 #0.999 #0.995
    # epsilon -> trend to explore/exploit (choose a random action instead of the best according to Q-table)

    def discretize(x, v):
        xi = int(min(max(round((x-pos_min)/pos_bin),0), n_pos-1))
        vi = int(min(max(round((v-vel_min)/vel_bin),0), n_vel-1))
        return xi, vi

    # [TRAINING STAGE]
    reward_history = []
    episodes_to_print = (n_episodes*np.arange(5)/5).astype(int)
    for ep in range(n_episodes):
        x, v = 0.0, 0.0
        total_r = 0.0
        for k in range(T_episode):
            xi, vi = discretize(x, v)
            if np.random.rand() < eps:
                a_idx = np.random.randint(n_actions)
            else:
                a_idx = int(np.argmax(Q[xi,vi,:]))
            a = actions[a_idx]
            v = float(np.clip(v + a*dt, vel_min, vel_max))
            x = float(np.clip(x + v*dt, pos_min, pos_max))
            x_ref = x_ref_traj[k]; v_ref = v_ref_traj[k]

            r = -( (1-reward_weight)*(x - x_ref)**2 + reward_weight*(v - v_ref)**2 )
            
            total_r += r
            xi2, vi2 = discretize(x,v)
            best_next = np.max(Q[xi2,vi2,:])

            Q[xi,vi,a_idx] += alpha*(r + gamma*best_next - Q[xi,vi,a_idx])
        
        eps = max(eps_end, eps*eps_decay)
        reward_history.append(total_r)

        # Print regularly the evolution of the Q-table
        if ep in episodes_to_print:
            # Flatten the Q-table (xi, vi) in a unique index: col_idx = xi*n_v + vi
            n_states = n_pos * n_vel
            Q_flat = np.zeros((n_actions, n_states))
            for xi in range(n_pos):
                for vi in range(n_vel):
                    state_idx = xi * n_vel + vi
                    Q_flat[:, state_idx] = Q[xi, vi, :]
            # Plot the Q-table
            fig, ax = plt.subplots(figsize=(max(10, n_states*0.4), n_actions*1.2))
            im = ax.imshow(Q_flat, cmap='coolwarm', aspect='auto', origin='lower')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Q-value')
            ax.set_xlabel('Flattened state index (xi * n_v + vi)')
            ax.set_ylabel('Action index')
            ax.set_title('Full Q-table visualization (all actions, flattened state)')
            plt.tight_layout()
            print("#### Episode - "+str(ep))
            if BoolPrint:
                plt.savefig(FigureDirectoryRL+"Q-learning-trajectory_Qtable_episode"+str(ep)+".png", transparent=True, format="png")
            plt.show()

    # [TRAINING PERFORMANCE] plot learning curve
    window = 50
    smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid') # moving window
    plt.figure(figsize=(8,4))
    plt.plot(smoothed)
    plt.xlabel('Episode (smoothed)'); plt.ylabel('Total reward'); plt.title('Q-learning Vehicle Learning Curve'); plt.grid(True)
    if BoolPrint:
        plt.savefig(FigureDirectoryRL+"Q-learning-trajectory_training_reward.png", transparent=True, format="png")
    plt.show()

    # [TEST STAGE] evaluate greedy policy
    Tsim = T_episode
    x, v = 0.0, 0.0
    x_log = np.zeros(Tsim+1); v_log = np.zeros(Tsim+1); a_log = np.zeros(Tsim)
    x_log[0]=x; v_log[0]=v
    for k in range(Tsim):
        xi, vi = discretize(x,v)
        a_idx = int(np.argmax(Q[xi,vi,:]))
        a = actions[a_idx]
        v = float(np.clip(v + a*dt, vel_min, vel_max))
        x = float(np.clip(x + v*dt, pos_min, pos_max))
        x_log[k+1]=x; v_log[k+1]=v; a_log[k]=a

    x_ref = build_reference_signal_simple(dt, Tsim=T_episode) + np.random.normal()
    v_ref = np.gradient(x_ref, dt)
    if BoolPrint:
        plot_time_series(np.arange(Tsim), [x_log[1:], x_ref], ['x','x_ref'], 'Q-learning Vehicle: Position', 'Position (m)', filename = FigureDirectoryRL+"RL-trajectory-position.png")
        # plot_time_series(np.arange(Tsim), [v_log[1:], np.concatenate([v_ref, v_ref[-1:]])], ['v','v_ref'], 'Q-learning Vehicle: Velocity', 'Velocity (m/s)', filename = FigureDirectory+"RL-trajectory-speed.png")
        plot_time_series(np.arange(Tsim), [v_log[1:], v_ref], ['v','v_ref'], 'Q-learning Vehicle: Velocity', 'Velocity (m/s)', filename = FigureDirectoryRL+"RL-trajectory-speed.png")
        plot_time_series(np.arange(Tsim), [a_log], ['a'], 'Q-learning Vehicle: Acceleration', 'Acceleration (m/s^2)', filename = FigureDirectoryRL+"RL-trajectory-acceleration.png")
    else:
        plot_time_series(np.arange(Tsim), [x_log[1:], x_ref], ['x','x_ref'], 'Q-learning Vehicle: Position', 'Position (m)')
        plot_time_series(np.arange(Tsim), [v_log[1:], v_ref], ['v','v_ref'], 'Q-learning Vehicle: Velocity', 'Velocity (m/s)')
        plot_time_series(np.arange(Tsim), [a_log], ['a'], 'Q-learning Vehicle: Acceleration', 'Acceleration (m/s^2)')


# * Test 
qlearning_vehicle(BoolPrint=True)



"""
Avenues for Enhancement 
-> change reward 
-> extend state description 
(eg by incorporating the last reward, like Integrated measurement)
"""    




# -----------------------------
# Main: run selected examples
# -----------------------------
if __name__ == '__main__':
    print('1) Vehicle: PID')
    pid_vehicle()

    print('2) Vehicle: MPC')
    mpc_vehicle()

    print('3) Vehicle: Q-learning')
    qlearning_vehicle()

    print('All demos finished.')
