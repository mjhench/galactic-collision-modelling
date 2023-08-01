import numpy as np
from antennae.Particle import Particle
from antennae import GalaxyTools as gt


class Galaxy:
    """
    A galaxy of particles in twelve concentric rings and a black hole in the center.

        Methods:
            get_all_part()          : get list of all particles in the galaxy starting with the black hole at the center
            get_rings()             : get list of all rings in the galaxy
            get_num_part()          : get number of particles in the galaxy
            add_init_vel(vx, vy, vz): add initial velocities to the galaxy along the x, y, and z axes
            move_galaxy(dx, dy, dz) : move galaxy from (x, y, z) --> (x + dx, y + dy, z + dz)
            plot_galaxy(fig, ax)    : plot galaxy
    """

    def __init__(self, r_min, theta, axis, m, name="Unnamed Galaxy"):
        # hardcoded value to comply with project requirements
        self._num_rings = 12

        # preallocate rings for speed
        self._rings = [None] * self._num_rings

        # create large body at center
        self.black_hole = Particle([0, 0, 0], mass=m)

        # create and fill rings
        for i in range(self._num_rings):
            self._rings[i] = Galaxy.__generate_ring(
                0.2 * r_min + 0.05 * r_min * i, 12 + 3 * i, self.black_hole, theta, axis
            )

        # misc
        self._name = name

    def __str__(self):
        return (
            f"{self._name}:\n"
            f"    Number of rings: {self._num_rings}\n"
            f"    Number of particles: {self.get_num_part()}\n"
            f"    Position of black hole: {self.black_hole.pos}"
        )

    @staticmethod
    def __generate_ring(r, num_part, black_hole, theta, axis):
        """
        Generate a circular ring of equally-spaced particles
        :param r            : (float) radius of ring
        :param num_part     : (int) number of particles in the ring
        :param black_hole   : (Particle) particle object with very large mass in the center of the galaxy
        :param theta        : (float) optional polar angle (default = 0)
        :return             : (np.ndarray) num_part x 3 array of (x, y, z) coordinates for all particles
        """

        # define G
        G = 1  # gravitational constant in units of 25kpc, 1e8years, solar mass

        # find azimuthal angles where particles should be located
        azim = np.linspace(0, 2 * np.pi, num_part, endpoint=False)

        # calculate initial positions & velocities and rotate to desired initial galactic plane
        pos = gt.cylindrical_to_xyz(r, azim, 0)
        pos_rot = gt.rotate(pos, theta=theta, axis=axis)

        vel = gt.vel_init(G, black_hole.mass, azim, r, 5)
        vel_rot = gt.rotate(vel, theta=theta, axis=axis)

        # create particle instances
        particles = [None] * num_part

        for i in range(len(pos_rot[0])):
            particles[i] = Particle(
                [pos_rot[0][i], pos_rot[1][i], pos_rot[2][i]], [vel_rot[0][i], vel_rot[1][i], vel_rot[2][i]]
            )

        return particles

    def get_all_part(self):
        """
        Get a list of all the particles in the galaxy, including the black hole. This will return the black hole as the
        first object in the list, followed by the rest of the particles.
        :return: (list) list of all the particles in the system, beginning with the black hole.
        """
        # preallocate particles array
        n = self.get_num_part() + 1
        particles = [None] * n

        # iterate over all particles and store in particles array
        particles[0] = self.black_hole

        i = 1  # note: have to do it this way because the rings are different sizes
        for ring in self._rings:
            for part in ring:
                particles[i] = part
                i += 1

        return particles

    def get_rings(self):
        """
        Get the galaxy's rings.
        :return : (list) list of all rings, with each ring being a list of Particle objects
        """
        return self._rings

    def get_num_part(self):
        """
        Get number of particles in the galaxy.
        :return : (int) number of particles in galaxy
        """
        num = 0

        for i in self._rings:
            num += len(i)

        return num

    def add_init_vel(self, vx, vy, vz):
        """
        Add initial velocities to all particles in the galaxy (both in massless ring particles and the galactic core)

        :param vx: (float) initial velocity along x-axis
        :param vy: (float) initial velocity along y-axis
        :param vz: (float) initial velocity along z-axis
        """
        particles = self.get_all_part()

        for part in particles:
            part.vel = part.vel + np.array([vx, vy, vz])

    def move_galaxy(self, dx, dy, dz):
        """
        Translate the galaxy in 3D Cartesian space.
        :param dx: (float) amount to translate the galaxy along the x-axis.
        :param dy: (float) amount to translate the galaxy along the y-axis.
        :param dz: (float) amount to translate the galaxy along the z-axis.
        :return:
        """
        # move the black hole
        self.black_hole.move_part(dx, dy, dz)

        # move all the particles in the rings
        for ring in self._rings:
            for part in ring:
                part.move_part(dx, dy, dz)

    def plot_galaxy(self, fig, ax):
        """
        Plot the galaxy.
        :param fig  : (matplotlib.figure) matplotlib figure to plot on
        :param ax   : (matplotlib.axes.Axes) axes object to plot on
        """

        # plot rings
        for ring in self._rings:
            for part in ring:
                ax.scatter(part.pos[0], part.pos[1], part.pos[2], c="r", s=10)

        # plot black hole
        ax.scatter(
            self.black_hole.pos[0], self.black_hole.pos[1], self.black_hole.pos[2], c="b", s=100
        )

        # make graph nice
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
