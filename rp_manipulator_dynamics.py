import sympy as sp
import roboticstoolkit as rtk


def main():
    # Solve the inverse dynamics for an RP manipulator

    # Define symbolic variables
    theta1, d2 = sp.symbols('theta1, d2')
    L1, L2, m1, m2, g = sp.symbols('L1, L2, m1, m2, g')

    # Define the manipulator
    dh_table = [
        [0,       0,     0, theta1],
        [0, sp.pi/2,    d2,      0],
        [0,       0,     0,      0]
    ]
    transforms = rtk.link_transforms(dh_table)
    p_coms = [
        sp.zeros(3,1),
        sp.Matrix([0, -L1/2, 0]),
        sp.Matrix([0, 0, 0])
    ]
    masses = [0, m1, m2]
    inertias = [
        sp.zeros(3),
        sp.Matrix([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]) * m1 * L1 * L1 / 12,
        sp.Matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]) * m2 * L2 * L2 / 12,
    ]
    joint_types = [0, 'R', 'P']

    # Define the boundary conditions
    gravity = sp.Matrix([0, -g, 0])
    f_end_effector = sp.zeros(3, 1)
    n_end_effector = sp.zeros(3, 1)

    # Solve the inverse dynamics
    equations = rtk.equations_of_motion(transforms, p_coms, masses, inertias, joint_types, gravity, f_end_effector, n_end_effector)

    # Print the equations
    for i, eq in enumerate(equations):
        force_string = 'torque' if joint_types[i] == 'R' else 'force'
        print(f'{force_string}{i} == {eq}')

if __name__ == '__main__':
    main()
