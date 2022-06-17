import sympy as sp
import roboticstoolkit as rtk


def main():
    # Solve the inverse dynamics for an RR manipulator

    # Define symbolic variables
    theta1, theta2 = sp.symbols('theta1, theta2')
    L1, L2, m1, m2, g = sp.symbols('L1, L2, m1, m2, g')

    # Define the manipulator
    transforms = [
        rtk.rot_z(theta1),
        rtk.trans(L1, 0, 0) * rtk.rot_z(theta2),
        rtk.trans(L2, 0, 0)
    ]
    p_coms = [
        sp.zeros(3,1),
        sp.Matrix([L1/2, 0, 0]),
        sp.Matrix([L2/2, 0, 0])
    ]
    masses = [0, m1, m2]
    inertias = [
        sp.zeros(3),
        sp.Matrix([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * m1 * L1 * L1 / 12,
        sp.Matrix([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * m2 * L2 * L2 / 12,
    ]
    joint_types = [0, 'R', 'R']

    # Define the boundary conditions
    gravity = sp.Matrix([0, -g, 0])
    f_end_effector = sp.zeros(3, 1)
    n_end_effector = sp.zeros(3, 1)

    # Solve the inverse dynamics, print the equations
    equations = rtk.equations_of_motion(transforms, p_coms, masses, inertias, joint_types, gravity, f_end_effector, n_end_effector)

    print('Joint forces only')
    rtk.print_equations_dict(equations, ['joint_force'])

    print('\nAll equations')
    rtk.print_equations_dict(equations)

if __name__ == '__main__':
    main()
