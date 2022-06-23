import sympy as sp
from roboticstoolkit.transforms import rotation, translation, three_vector, four_vector
from roboticstoolkit.propagations import *
from roboticstoolkit.kinematics import base_transforms
from roboticstoolkit.core import diff_total
from functools import reduce


def dynamics_newton_euler(transforms, pos_coms, masses, inertias, joint_types, gravity, f_end_effector, n_end_effector):
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
    theta_vel = [sp.S(0)] * (num_frames - 1)
    theta_accel = [sp.S(0)] * (num_frames - 1)
    d_vel = [sp.S(0)] * (num_frames - 1)
    d_accel = [sp.S(0)] * (num_frames - 1)
    for i in range(len(joint_types)):
        if joint_types[i] == 'R':
            theta_vel[i] = sp.symbols(f'\dot{{\\theta_{i}}}')
            theta_accel[i] = sp.symbols(f'\ddot{{\\theta_{i}}}')
        elif joint_types[i] == 'P':
            d_vel[i] = sp.symbols(f'\dot{{d_{i}}}')
            d_accel[i] = sp.symbols(f'\ddot{{d_{i}}}')
    
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
    joint_force = [sp.S(0)] * (num_frames - 1)

    # Set boundary conditions
    accel[0] = sp.Matrix(-gravity)
    force_link[-1] = sp.Matrix(f_end_effector)
    moment_link[-1] = sp.Matrix(n_end_effector)

    # Outward propagation
    for i in range(1, num_frames - 1):
        omega[i] = sp.simplify(omega_next_frame(rotations[i-1], omega[i-1], theta_vel[i]))
        alpha[i] = sp.simplify(alpha_next_frame(rotations[i-1], alpha[i-1], omega[i-1], theta_vel[i], theta_accel[i]))
        accel[i] = sp.simplify(accel_next_frame(rotations[i-1], accel[i-1], alpha[i-1], omega[i-1], translations[i-1], d_vel[i], d_accel[i]))
        accel_com[i] = sp.simplify(accel_curr_frame(accel[i], alpha[i], omega[i], pos_coms[i]))
        force_com[i] = sp.simplify(force_com_curr_frame(masses[i], accel_com[i]))
        moment_com[i] = sp.simplify(moment_com_curr_frame(inertias[i], alpha[i], omega[i]))

    # Inward propagation
    for i in range(num_frames - 2, 0, -1):
        force_link[i] = sp.simplify(force_curr_frame(rotations[i], force_link[i+1], force_com[i]))
        moment_link[i] = sp.simplify(moment_curr_frame(rotations[i], moment_link[i+1], moment_com[i], translations[i], force_link[i+1], pos_coms[i], force_com[i]))

        # Get the generalised joint-space force, construct the output equation
        force = moment_link[i] if joint_types[i] == 'R' else force_link[i]
        joint_force[i] = sp.collect(sp.expand(force.dot(z_vec)), [*theta_accel, *d_accel])

    # Construct dictionary of equations
    return {
        'tau': joint_force,
        'omega': omega,
        'alpha': alpha,
        'a': accel,
        'a_c': accel_com,
        'f_c': force_com,
        'n_c': moment_com,
        'f': force_link,
        'n': moment_link
    }


def dynamics_lagrange(transforms, pos_coms, masses, inertias, joint_types, gravity, variables):
    """
    Compute the equations of motion of a serial manipulator.
    
    Compute the inverse dynamics using Lagrangian dynamics
    Args:
        transforms - List of incremental transformation matrices. Include all links and the end effector frame.
        p_com - List of 3-vectors. 0 for the ground frame, then positions of the centre of mass of each link. Measured relative
            to the asociated link's frame origin, represented in the link's associated frame.
        masses - List of scalars. 0 for the ground frame then masses of the links.
        inertias - List of 3x3 tensors. 0 for the ground frame, then inertia of each link, calculated at the centre of mass of
            each link, and represented in the associated link's frame.
        joint_types - List with 0 for the ground frame, then 'R' or 'P' for each joint depending on the joint type.
        gravity - 3-vector. Acceleration due to gravity.
        variables - List of symbols, representing the time-dependent generalised variables
    Return:
        Dictionary of symbolic equations
    """

    # Unpack transformation matrices
    rotations = [rotation(T) for T in transforms]

    # Construct mapping for symbolic derivatives
    variables_vel = [0] * len(variables)
    variables_accel = [0] *len(variables)
    for i in range(1, len(joint_types)):
        if joint_types[i] == 'R':
            variables_vel[i - 1] = sp.symbols(f'\dot{{\\theta_{i}}}')
            variables_accel[i - 1] = sp.symbols(f'\ddot{{\\theta_{i}}}')
        else:
            variables_vel[i - 1] = sp.symbols(f'\dot{{d_{i}}}')
            variables_accel[i - 1] = sp.symbols(f'\ddot{{d_{i}}}')

    diff_map = {variables[i]:variables_vel[i] for i in range(len(variables))}
    diff_map.update({variables_vel[i]:variables_accel[i] for i in range(len(variables))})

    # Get number of frames. Includes base frame, all links, and end effector frame
    num_frames = len(transforms) + 1

    # Find positions and velocities of CoM of each link in the ground frame
    pos_com_ground = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    vel_com_ground = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    t = sp.symbols('t')
    base_transform_matrices = base_transforms(transforms)
    for i in range(1, num_frames - 1):
        pos_com_ground[i] = three_vector(base_transform_matrices[i-1] * four_vector(pos_coms[i]))
        vel_com_ground[i] = diff_total(pos_com_ground[i], t, diff_map)
    
    # Find angular velocities of each link using propagation law
    omega = [sp.zeros(3,1) for _ in range(num_frames - 1)]
    theta_vel = [sp.symbols(f'\dot{{\\theta_{i}}}') if joint_types[i] == 'R' else 0 for i in range(len(joint_types))]
    for i in range(1, num_frames - 1):
        omega[i] = sp.simplify(omega_next_frame(rotations[i-1], omega[i-1], theta_vel[i]))

    # Find kinetic and potential energies
    kinetic_energies = [sp.S(0)] * (num_frames - 1)
    potential_energies = [sp.S(0)] * (num_frames - 1)
    for i in range(1, num_frames - 1):
        kinetic_energies[i] = sp.S(1)/2 * masses[i] * vel_com_ground[i].dot(vel_com_ground[i]) + sp.S(1)/2 * omega[i].dot(inertias[i] * omega[i])
        potential_energies[i] = -masses[i] * gravity.dot(pos_com_ground[i])

    # Find total kinetic and potential energies, find lagrangian
    kinetic_energy_total = sp.simplify(sum(kinetic_energies))
    potential_energy_total = sp.simplify(sum(potential_energies))
    lagrangian = kinetic_energy_total - potential_energy_total

    # Apply the Euler-Lagrange equation to find the equations of motion
    joint_force = [sp.S(0)] * (num_frames - 1)
    for i in range(1, num_frames - 1):
        force = diff_total(lagrangian.diff(variables_vel[i-1]), t, diff_map) - lagrangian.diff(variables[i-1])
        joint_force[i] = sp.collect(sp.expand(force), [*variables_accel])

    # Construct dictionary of equations
    return {
        'tau': joint_force,
        'p_c': pos_com_ground,
        'v_c': vel_com_ground,
        'omega': omega,
        'K': kinetic_energies,
        'V': potential_energies,
        'K_total': kinetic_energy_total,
        'V_total': potential_energy_total,
        'L': lagrangian,
    }
