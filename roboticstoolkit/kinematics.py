import sympy as sp
from roboticstoolkit.transforms import rot_z, rot_y, rot_x, rotation, cross_matrix
from roboticstoolkit.transforms import dh_transform


# Rotations

def euler_ZYX(alpha, beta, gamma):
    return rotation(rot_z(alpha) * rot_y(beta) * rot_x(gamma))


def euler_ZYX_inverse(rotation):
    """
    Args:
        rotation - 3x3 rotation matrix
    Return:
        3-tuple (alpha, beta, gamma). Rotation of the frame about the z-axis, then y, then x
    """
    # Are we in gimbal lock?
    if rotation[0, 0] == 0 and rotation[1, 0] == 0 and rotation[2, 1] == 0 and rotation [2, 2] == 0 and abs(rotation[2, 0]) == 1:
        # Yes. The negative of this is also a solution
        return (
            0,
            sp.pi/2,
            sp.atan2(rotation[0, 1], rotation[1, 1])
        )
    else:
        # No
        return (
            sp.atan2(rotation[1, 0], rotation[0, 0]),
            -sp.asin(rotation[2, 0]),
            sp.atan2(rotation[2, 1], rotation[2, 2])
        )


def angle_axis(angle, axis):
    axis = axis.normalized()
    return sp.simplify(axis*axis.T * (1-sp.cos(angle)) + sp.eye(3)*sp.cos(angle) + cross_matrix(axis)*sp.sin(angle))


def angle_axis_inverse(rotation):
    """
    Args:
        rotation - 3x3 rotation matrix
    Return:
        2-tuple (angle, axis), with 3-vector axis
    """
    diag = sp.Matrix([rotation[i, i] for i in range(3)])
    angle = sp.simplify(sp.acos((sum(diag) - 1)/2))
    if angle == 0:
        axis = sp.Matrix([1, 0, 0])
    elif angle == sp.pi:
        axis = sp.Matrix([sp.sqrt((diag[i] + 1)/2) for i in range(3)])
    else:
        axis = sp.Matrix([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1]
        ]) / (2 * sp.sin(angle))

    return angle, sp.simplify(axis)


# Forward kinematics

def link_transforms(dh_table):
    """
    Compute incremental frame transform from a DH table

    Args:
        dh_table - List of DH parameters. Each entry corrsponds to one joint, and has
            DH parameters listed in the order a, alpha, d, theta
    Return:
        List of incremental frame transformations, each stored as a sympy array
    """
    return [dh_transform(*dh_table[i]) for i in range(len(dh_table))]


def base_transforms(link_transforms):
    """
    Comput transformations to base frame from each link

    Args:
        link_transforms - list of incremental frame transforms, each stored as a sympy array
    Return:
        list of base frame transforms, each stored as a sympy array
    """

    num_frames = len(link_transforms)
    base_transforms = [None] * num_frames

    current_base_transform = sp.eye(4)
    for i in range(num_frames):
        current_base_transform = current_base_transform * link_transforms[i]
        base_transforms[i] = current_base_transform
    return base_transforms


def end_transform(dh_table):
    return base_transforms(link_transforms(dh_table))[-1]
