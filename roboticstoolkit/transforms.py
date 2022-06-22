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


def axis_x(transform):
    return transform[:3, 0]


def axis_y(transform):
    return transform[:3, 1]


def axis_z(transform):
    return transform[:3, 2]


def translation(transform):
    return transform[:3, 3]


def inverse_transform(transform):
    rot = rotation(transform)
    inverse = sp.eye(4)
    inverse[:3, :3] = rot.T
    inverse[:3, 3] = -rot.T * translation(transform)
    return inverse


def cross_matrix(vector):
    x, y, z = vector
    return sp.Matrix([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])


def three_vector(vector):
    return sp.Matrix(vector[:3])


def four_vector(vector):
    return vector.row_insert(3, sp.Matrix([1]))
