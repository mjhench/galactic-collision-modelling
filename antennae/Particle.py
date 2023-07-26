import numpy as np


class Particle:
    """
    A particle in space.

    Methods:
        update_vals(new_pos, dt)    : update position & velocity based on new position and change in time
        move_part(dx, dy, dz)       : move particle in 3D Cartesian space from (x, y, z) -> (x + dx, y + dy, z + dz)
    """

    def __init__(self, pos, vel=np.array([0, 0, 0]), mass=0):
        if len(pos) != 3 or len(vel) != 3:
            raise Exception("Invalid inputs: position and velocity arguments must be 3D (numpy) arrays!")

        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.mass = mass

    def __str__(self):
        return (
            f"Pos:({self.pos[0]}, {self.pos[1]}, {self.pos[2]})\n"
            f"Vel:({self.vel[0]}, {self.vel[1]}, {self.vel[2]})"
        )

    def update_vals(self, new_pos, dt):
        """
        Update position and velocity parameters of particle given new position vector and change in time dt.
        :param new_pos  : (list, np.ndarray) 3D vector of new particle position (x, y, z)
        :param dt       : (float) change in time
        """
        if len(new_pos) != 3:
            raise Exception("Invalid inputs: position argument must be a 3D (numpy) array!")

        self.vel = (np.array(new_pos) - self.pos) / dt
        self.pos = np.array(new_pos)

    def move_part(self, dx, dy, dz):
        """
        Translate the particle in 3D Cartesian space.
        :param dx   : (float) amount to translate particle by along x-axis
        :param dy   : (float) amount to translate particle by along y-axis
        :param dz   : (float) amount to translate particle by along z-axis
        """
        self.pos = self.pos + np.array([dx, dy, dz])
