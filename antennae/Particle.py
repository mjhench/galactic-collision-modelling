import numpy as np


class Particle:
    """
    A particle in space.

    Methods:
        get_pos()                   : get position vector
        set_pos(new_pos)            : set position vector
        get_vel()                   : get velocity vector
        set_vel(new_vel)            : set velocity vector
        get_mass()                  : get particle mass
        update_vals(new_pos, dt)    : update position & velocity based on new position and change in time
        move_part(dx, dy, dz)       : move particle in 3D Cartesian space from (x, y, z) -> (x + dx, y + dy, z + dz)
    """

    def __init__(self, pos, vel=np.array([0, 0, 0]), mass=0):
        if len(pos) != 3 or len(vel) != 3:
            raise Exception("Invalid inputs: position and velocity arguments must be 3D (numpy) arrays!")

        self._pos = np.array(pos)
        self._vel = np.array(vel)
        self._mass = mass

    def __str__(self):
        return (
            f"Pos:({self._pos[0]}, {self._pos[1]}, {self._pos[2]})\n"
            f"Vel:({self._vel[0]}, {self._vel[1]}, {self._vel[2]})"
        )

    def get_pos(self):
        """
        Get position vector.
        :return : (np.ndarray) 3D position vector (x, y, z)
        """
        return self._pos

    def set_pos(self, new_pos):
        """
        Set position vector.
        :param new_pos  : (list, np.ndarray) 3D position vector (x, y, z)
        """
        if len(new_pos) != 3:
            raise Exception("Invalid inputs: position argument must be a 3D (numpy) array!")

        self._pos = np.array(new_pos)

    def get_vel(self):
        """
        Get velocity vector.
        :return : (np.ndarray) 3D velocity vector (vx, vy, vz)
        """
        return self._vel

    def set_vel(self, new_vel):
        """
        Set velocity vector.
        :param new_vel  : (list, np.ndarray) 3D velocity vector (vx, vy, vz)
        """
        if len(new_vel) != 3:
            raise Exception("Invalid inputs: velocity argument must be a 3D (numpy) array!")

        self._vel = np.array(new_vel)

    def get_mass(self):
        """
        Get mass of the particle
        :return : (float) mass of the particle
        """
        return self._mass

    def update_vals(self, new_pos, dt):
        """
        Update position and velocity parameters of particle given new position vector and change in time dt.
        :param new_pos  : (list, np.ndarray) 3D vector of new particle position (x, y, z)
        :param dt       : (float) change in time
        """
        if len(new_pos) != 3:
            raise Exception("Invalid inputs: position argument must be a 3D (numpy) array!")

        self._vel = (np.array(new_pos) - self._pos) / dt
        self._pos = np.array(new_pos)

    def move_part(self, dx, dy, dz):
        """
        Translate the particle in 3D Cartesian space.
        :param dx   : (float) amount to translate particle by along x-axis
        :param dy   : (float) amount to translate particle by along y-axis
        :param dz   : (float) amount to translate particle by along z-axis
        """
        self._pos = self._pos + np.array([dx, dy, dz])
