import sympy as sp
from roboticstoolkit.transforms import dh_transform

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
