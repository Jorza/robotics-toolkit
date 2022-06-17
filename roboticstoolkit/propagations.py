from roboticstoolkit.constants import z_vec


# Propagation equations
# By default, properties relate to frame i. If labeled 'next', then it relates to frames i+1.
# All vector properties of a frame are measured in their respective frame, with ground as the reference.
# Rotations map from the next frame to the current frame, and are represented in the current frame.
# Translations give the position of the next frame, represented in the current frame.
# The moment of inertia is calculated at the centre of mass, and represented in frame i.

# Outward kinematics propagations
def omega_next_frame(rotation, omega, theta_vel_next):
    return rotation.T * omega + theta_vel_next * z_vec

def vel_curr_frame(vel, omega, translation):
    return vel + omega.cross(translation)

def vel_next_frame(rotation, vel, omega, translation, d_vel_next):
    return rotation.T * vel_curr_frame(vel, omega, translation) + d_vel_next * z_vec

def alpha_next_frame(rotation, alpha, omega, theta_vel_next, theta_accel_next):
    return rotation.T * alpha + (rotation.T * omega).cross(theta_vel_next * z_vec) + theta_accel_next * z_vec

def accel_curr_frame(accel, alpha, omega, translation):
    return accel + alpha.cross(translation) + omega.cross(omega.cross(translation))

def accel_next_frame(rotation, accel, alpha, omega, translation, d_vel_next, d_accel_next):
    return rotation.T * accel_curr_frame(accel, alpha, omega, translation) + 2*(rotation.T * omega).cross(d_vel_next * z_vec) + d_accel_next * z_vec

# Outwrd dynamics equations
def force_com_curr_frame(mass, accel_com):
    return mass * accel_com

def moment_com_curr_frame(inertia, alpha, omega):
    return inertia * alpha + omega.cross(inertia * omega)

# Inward propagation equations
def force_curr_frame(rotation, force_next, force_com):
    return rotation * force_next + force_com

def moment_curr_frame(rotation, moment_next, moment_com, p_next, force_next, p_com, force_com):
    return rotation * moment_next + moment_com + p_next.cross(rotation * force_next) + p_com.cross(force_com)
