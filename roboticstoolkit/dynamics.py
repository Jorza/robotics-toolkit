import sympy as sp
from roboticstoolkit.transforms import rotation, translation
from roboticstoolkit.propagations import *


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
        Dictionary of symbolic equations
    """

    # Unpack transformation matrices
    rotations, translations = zip(*((rotation(T), translation(T)) for T in transforms))

    # Get number of frames. Includes base frame, all links, and end effector frame
    num_frames = len(transforms) + 1
    
    # Define joint-space symbolic variables for all frames
    # Use lists (as opposed to sympy matrices) since these are always accessed individually
    theta_vel = [0] * (num_frames - 1)
    theta_accel = [0] * (num_frames - 1)
    d_vel = [0] * (num_frames - 1)
    d_accel = [0] * (num_frames - 1)
    for i in range(len(joint_types)):
        if joint_types[i] == 'R':
            theta_vel[i] = sp.symbols(f'theta_vel{i}')
            theta_accel[i] = sp.symbols(f'theta_accel{i}')
        elif joint_types[i] == 'P':
            d_vel[i] = sp.symbols(f'd_vel{i}')
            d_accel[i] = sp.symbols(f'd_accel{i}')
    
    # Define task-space vectors for all frames
    omega = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    alpha = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    accel = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    accel_com = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    force_com = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    moment_com = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    force_link = [sp.zeros(3,1) for _ in range(num_frames)]
    moment_link = [sp.zeros(3,1) for _ in range(num_frames)]

    # Define the output list of joint generalised forces
    joint_force = [0] * (num_frames - 1)

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
        force_com[i] = sp.simplify(force_com_curr_frame(masses[i], accel_com[i]))
        moment_com[i] = sp.simplify(moment_com_curr_frame(inertias[i], alpha[i], omega[i]))

    # Inward propagation
    for i in range(num_frames - 2, 0, -1):
        force_link[i] = sp.simplify(force_curr_frame(rotations[i], force_link[i+1], force_com[i]))
        moment_link[i] = sp.simplify(moment_curr_frame(rotations[i], moment_link[i+1], moment_com[i], translations[i], force_link[i+1], p_coms[i], force_com[i]))

        # Get the generalised joint-space force, construct the output equation
        force = moment_link[i] if joint_types[i] == 'R' else force_link[i]
        joint_force[i] = sp.collect(sp.expand(force.dot(z_vec)), [*theta_accel, *d_accel])

    # Construct dictionary of equations
    return {
        'joint_force': joint_force,
        'omega': omega,
        'alpha': alpha,
        'accel': accel,
        'accel_com': accel_com,
        'force_com': force_com,
        'moment_com': moment_com,
        'force_link': force_link,
        'moment_link': moment_link
    }


def print_equations_dict(equations_dict, keys=None):
    """
    Print equations from a dictionary

    Args:
        equation_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Is printed as 'symbolic_name == expression'.
            Each expression can be a single expression or a list of expressions.
            If it is a list, the entries will be printed as separate equations.
        keys - A list of keys (symbolic names) to print. If omitted, all equations are printed
    """

    # Print all equations by default
    if keys is None:
        keys = equations_dict.keys()
    
    # Print each of the given equations
    for name in keys:
        equations = equations_dict[name]
        if isinstance(equations, list):
            for i, eq in enumerate(equations):
                print(f'{name}{i} == {eq}')
        else:
            print(f'{name} == {equations}')
