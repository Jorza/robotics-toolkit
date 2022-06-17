import sympy as sp
import roboticstoolkit as rtk


def main():
    # Find the jacobian for an RR manipulator

    # Define symbolic variables
    theta1, theta2 = sp.symbols('theta1, theta2')
    L1, L2 = sp.symbols('L1, L2')

    # Define the manipulator
    transforms = [
        rtk.rot_z(theta1),
        rtk.trans(L1, 0, 0) * rtk.rot_z(theta2),
        rtk.trans(L2, 0, 0)
    ]
    joint_types = ['R', 'R']

    # Find the jacobian
    print("Full 3D Jacobian")
    print(rtk.jacobian(rtk.base_transforms(transforms), joint_types, position_only=False))

    print("\n3D Jacobian, position only")
    print(rtk.jacobian(rtk.base_transforms(transforms), joint_types, position_only=True))

    print("\n2D Jacobian")
    print(rtk.jacobian_planar(rtk.z_vec, rtk.base_transforms(transforms), joint_types, position_only=False))

    print("\n2D Jacobian, position only")
    print(rtk.jacobian_planar(rtk.z_vec, rtk.base_transforms(transforms), joint_types, position_only=True))

if __name__ == '__main__':
    main()
