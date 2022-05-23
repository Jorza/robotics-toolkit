import sympy as sp


def rot_z(angle):
    return sp.Matrix([
        [sp.cos(angle), -sp.sin(angle), 0, 0],
        [sp.sin(angle),  sp.cos(angle), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
    

def rot_y(angle):
    return sp.Matrix([
        [sp.cos(angle),  0, sp.sin(angle), 0],
        [0,              1, 0,             0],
        [-sp.sin(angle), 0, sp.cos(angle), 0],
        [0,              0, 0,             1]
    ])


def rot_x(angle):
    return sp.Matrix([
        [1, 0,              0,             0],
        [0, sp.cos(angle), -sp.sin(angle), 0],
        [0, sp.sin(angle),  sp.cos(angle), 0],
        [0, 0,              0,             1]
    ])


def euler_ZYX(alpha, beta, gamma):
    return rot_z(alpha) * rot_y(beta) * rot_x(gamma)


def trans(x, y, z):
    return sp.Matrix([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def screw_x(offset, angle):
    return trans(offset, 0, 0) * rot_x(angle)


def screw_y(offset, angle):
    return trans(0, offset, 0) * rot_y(angle)


def screw_z(offset, angle):
    return trans(0, 0, offset) * rot_z(angle)


def dh_transform(length, twist, offset, angle):
    return screw_x(length, twist) * screw_z(offset, angle)


def rotation(transform):
    return transform[:3,:3]


def translation(transform):
    return transform[:3, 3]


def cross_matrix(vector):
    x, y, z = vector
    return sp.Matrix([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])


def inverse_transform(transform):
    rot = rotation(transform)
    inverse = sp.eye(4)
    inverse[:3, :3] = rot.T
    inverse[:3, 3] = -rot.T * translation(transform)
    return inverse


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
