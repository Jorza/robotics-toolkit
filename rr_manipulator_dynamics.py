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
    pos_coms = [
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

    # Define gravity
    gravity = sp.Matrix([0, -g, 0])
    
    # Define the boundary conditions (for Newton-Euler)
    f_end_effector = sp.zeros(3, 1)
    n_end_effector = sp.zeros(3, 1)

    # Define symbols for the variables and derivative (for Lagrange)
    variables = theta1, theta2

    # Solve the inverse dynamics, print the equations
    equations_newton_euler = rtk.dynamics_newton_euler(transforms, pos_coms, masses, inertias, joint_types, gravity, f_end_effector, n_end_effector)
    equations_lagrange = rtk.dynamics_lagrange(transforms, pos_coms, masses, inertias, joint_types, gravity, variables)
    
    print('Joint forces only')
    print('Newton-Euler')
    rtk.print_equations_dict(equations_newton_euler, ['tau'])
    print('\nLagrange')
    rtk.print_equations_dict(equations_lagrange, ['tau'])

    print('\nAll equations')
    print('Newton-Euler')
    rtk.print_equations_dict(equations_newton_euler)
    print('\nLagrange')
    rtk.print_equations_dict(equations_lagrange)

    # Are the two formulations equal?
    joint_force_newton_euler = equations_newton_euler['tau']
    joint_force_lagrange = equations_lagrange['tau']
    print("\nAre the two formulations equal?")
    print(all((sp.simplify(joint_force_newton_euler[i] - joint_force_lagrange[i]) == 0 for i in range(len(joint_force_lagrange)))))


if __name__ == '__main__':
    main()
