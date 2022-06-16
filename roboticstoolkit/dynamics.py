import sympy as sp
from kinematics import *

# Define a useful constant here
z_vec = sp.Matrix([0, 0, 1])

# Propagation equations
# By default, properties relate to frame i. If labeled 'next', then it relates to frames i+1.
# All vector properties of a frame are measured in their respective frame, with ground as the reference.
# Rotations map from the next frame to the current frame, and are represented in the current frame.
# Translations give the position of the next frame, represented in the current frame.
# The moment of inertia is calculated at the centre of mass, and represented in frame i.

# Outward ropagation equations
def omega_next_frame(rotation, omega, theta_vel_next):
    return rotation.T * omega + theta_vel_next * z_vec

def alpha_next_frame(rotation, alpha, omega, theta_vel_next, theta_accel_next):
    return rotation.T * alpha + (rotation.T * omega).cross(theta_vel_next * z_vec) + theta_accel_next * z_vec

def accel_curr_frame(accel, alpha, omega, translation):
    return accel + alpha.cross(translation) + omega.cross(omega.cross(translation))

def accel_next_frame(rotation, accel, alpha, omega, translation, d_vel_next, d_accel_next):
    return rotation.T * accel_curr_frame(accel, alpha, omega, translation) + 2*(rotation.T * omega).cross(d_vel_next * z_vec) + d_accel_next * z_vec

def force_com(mass, accel_com):
    return mass * accel_com

def moment_com(inertia, alpha, omega):
    return inertia * alpha + omega.cross(inertia * omega)

# Inward propagation equations
def force_curr_frame(R_next, force_next, force_com):
    return R_next * force_next + force_com

def moment_curr_frame(R_next, moment_next, moment_com, p_next, force_next, p_com, force_com):
    return R_next * moment_next + moment_com + p_next.cross(R_next * force_next) + p_com.cross(force_com)

def joint_force(force):
    return force.dot(z_vec)


def equations_of_motion(transforms, p_coms, masses, inertias, joint_types, gravity, f_end_effector, n_end_effector):
    """
    Compute the equations of motion of a serial manipulator.
    
    Compute the inverse dynamics using outward and inward propagation
    Args:
        transforms - List of incremental transformation matrices. Include all links and the end effector frame.
        p_com - List of 3-vectors. 0 for the ground frame, then positions of the centre of mass of each link. Measured relative
            to the asociated link's frame origin, represented in the link's associated frame.
        masses - List of scalars. 0 for the ground frame then masses of the links.
        inertias - List of 3x3 tensors. 0 for the ground frame, then inertia of each link, calculated at the centre of mass of
            each link, and represented in the associated link's frame.
        joint_types - List with 0 for the ground frame, then 'R' or 'P' for each joint depending on the joint type.
        gravity - 3-vector. Acceleration due to gravity.
        f_end_effector - 3-vector. Force applied by end effector to the environment, in frame of end effector.
        n_end_effector - 3-vector. Moment applied by end effector to the environment, in frame of end effector.
    Return:
        List of symbolic equations
    """

    # Unpack transformation matrices
    rotations, translations = zip(*((rotation(T), translation(T)) for T in transforms))

    # Get number of frames. Includes base frame, all links, and end effector frame
    num_frames = len(transforms) + 1
    
    # Define joint-space symbolic variables for all frames
    theta_vel = sp.zeros(num_frames, 1)
    theta_accel = sp.zeros(num_frames, 1)
    d_vel = sp.zeros(num_frames, 1)
    d_accel = sp.zeros(num_frames, 1)
    for i in range(len(joint_types)):
        if joint_types[i] == 'R':
            theta_vel[i] = sp.symbol(f'theta_vel{i}')
            theta_accel[i] = sp.symbol(f'theta_accel{i}')
        elif joint_types[i] == 'P':
            d_vel[i] = sp.symbol(f'd_vel{i}')
            d_accel[i] = sp.symbol(f'd_accel{i}')
    
    # Define task-space vectors for all frames
    omega = [sp.zeros(3,1) for _ in range(num_frames)]
    alpha = [sp.zeros(3,1) for _ in range(num_frames)]
    accel = [sp.zeros(3,1) for _ in range(num_frames)]
    accel_com = [sp.zeros(3,1) for _ in range(num_frames)]
    force_link = [sp.zeros(3,1) for _ in range(num_frames)]
    force_com = [sp.zeros(3,1) for _ in range(num_frames)]
    moment_link = [sp.zeros(3,1) for _ in range(num_frames)]
    moment_com = [sp.zeros(3,1) for _ in range(num_frames)]

    # Define the output list of equations
    equations = sp.zeros(num_frames, 1)

    # Set boundary conditions
    accel[0] = sp.Matrix(-gravity)
    force_link[-1] = sp.Matrix(f_end_effector)
    moment_link[-1] = sp.Matrix(n_end_effector)

    # Outward propagation
    for i in range(1, num_frames - 1):
        omega[i] = sp.simplify(omega_next_frame(rotations[i-1], omega[i-1], theta_vel[i]))
        alpha[i] = sp.simplify(alpha_next_frame(rotations[i-1], alpha[i-1], omega[i-1], theta_vel[i], theta_accel[i]))
        accel[i] = sp.simplify(accel_next_frame(rotations[i-1], accel[i-1], alpha[i-1], omega[i-1], translations[i-1], d_vel[i], d_accel[i]))
        accel_com[i] = sp.simplify(accel_curr_frame(accel[i], alpha[i], omega[i], p_coms[i]))
        force_com[i] = sp.simplify(force_com(masses[i], accel_com[i]))
        moment_com[i] = sp.simplify(moment_com(inertias[i], alpha[i], omega[i]))

    # Inward propagation
    for i in range(num_frames - 2, 0, -1):
        force_link[i] = sp.simplify(force_curr_frame(rotations[i], force_link[i+1], force_com[i]))
        moment_link[i] = sp.simplify(moment_curr_frame(rotations[i], moment_link[i+1], moment_com[i], translations[i], force_link[i+1], p_coms[i], force_com[i]))

        # Get the generalised joint-space force, construct the output equation
        if joint_types[i] == 'R':
            force = moment_link[i]
            force_symbol = sp.symbol(f'torque{i}')
        elif joint_types[i] == 'P':
            force = force_link[i]
            force_symbol = sp.symbol(f'force{i}')
        equations[i] = sp.Eq(force_symbol, sp.collect(sp.expand(joint_force(force)), [*theta_accel, *d_accel]))
    
    return equations
