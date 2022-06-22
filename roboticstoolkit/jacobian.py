from multiprocessing.sharedctypes import Value
import sympy as sp
from roboticstoolkit.core import *
from roboticstoolkit.transforms import axis_z, translation


def jacobian(base_transforms, joint_types, position_only=False):
    """
    Compute the jacobian of a manipulator using the Plucker coordinates.

    Satisfies the equation [vel.T, omega.T].T = J * [q1, q2, q3, ...., qn].T
    
    Args:
        base_transforms - List of transforms describing frames 1, 2, 3, ..., n where n is the end-effcetor frame.
            Can use the output from base_transforms() directly
        joint_types - List with 'R' or 'P' for each joint depending on the joint type.
        position_only - If True, only include first 3 rows in Jacobian as opposed to usual 6.
    Return:
        Jacobian matrix as a sympy Matrix
    """
    num_joints = len(base_transforms) - 1
    
    # Prepare output data structure
    jacobian = sp.zeros(3 if position_only else 6, num_joints)

    # Construct jacobian using Plucker coordinates
    end_effector_position = translation(base_transforms[-1])
    for i in range(num_joints):
        joint_axis = axis_z(base_transforms[i])
        if joint_types[i] == 'R':
            moment_arm = end_effector_position - translation(base_transforms[i])
            jacobian[0:3, i] = sp.simplify(joint_axis.cross(moment_arm))
            if not position_only:
                jacobian[3:6, i] = joint_axis
        elif joint_types[i] == 'P':
            jacobian[0:3, i] = joint_axis
            # If not position only, rotational portion of Jacobian column already has all 0s
    
    return jacobian


def jacobian_planar(plane_axis, base_transforms, joint_types, position_only=False):
    """
    Computes jacobian as in jacobian() but then discards rows corresponding to out-of-plane movements

    For example, if plane_axis == z_vec, then the output satisfies the equation
    [vel_x, vel_y, omega_z].T = J * [q1, q2, q3].T

    Args:
        plane_axis - Either x_vec, y_vec or z_vec. Normal to the plane
    """
    assert plane_axis == x_vec or plane_axis == y_vec or plane_axis == z_vec
    
    jacobian_matrix = jacobian(base_transforms, joint_types, position_only=False)
    if plane_axis == x_vec:
        keep_rows = [1, 2, 3]
    elif plane_axis == y_vec:
        keep_rows = [0, 2, 4]
    elif plane_axis == z_vec:
        keep_rows = [0, 1, 5]
    else:
        raise ValueError('Plane axis must be a coordinate axis')
    
    if position_only:
        keep_rows = keep_rows[0:2]

    return jacobian_matrix[keep_rows, :]
