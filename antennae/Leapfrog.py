import numpy as np
from tqdm import tqdm


def main(galaxy_1, galaxy_2):
    # create array with pointers to all particles
    all_particles = galaxy_1.get_all_part() + galaxy_2.get_all_part()

    # calculate total number of particles
    n = len(all_particles)

    # next, set integration parameters
    nstep = 1500  # total number of steps
    nout = 2  # intervals between outputs
    dt = 0.03  # time step in 1e8 years
    tnow = -11.85  # set initial time

    # preallocate for speed
    x = np.zeros((3, n))  # position
    v = np.zeros((3, n))  # velocity
    m = np.zeros(n)  # mass

    # copy values over from particle objects to arrays above
    for i in range(n):
        x[0, i] = all_particles[i].get_pos()[0]
        x[1, i] = all_particles[i].get_pos()[1]
        x[2, i] = all_particles[i].get_pos()[2]
        v[0, i] = all_particles[i].get_vel()[0]
        v[1, i] = all_particles[i].get_vel()[1]
        v[2, i] = all_particles[i].get_vel()[2]
        m[i] = all_particles[i].get_mass()

    # open text file to store all values
    f = open("data/leapstep_latest.csv", "w")

    # now, loop performing integration
    for step in tqdm(range(nstep)):  # loop nstep times
        if step % nout == 0:  # if time to output state
            for i in range(n):  # loop over all points...
                f.write(f"{tnow},{i},{x[0][i]},{x[1][i]},{x[2][i]},{v[0][i]},{v[1][i]},{v[2][i]}\n")
        leapstep(x, v, n, m, dt)  # take integration step
        tnow += dt  # and update value of time
    if nstep % nout == 0:  # if last output wanted
        for i in range(n):  # loop over all points...
            f.write(f"{tnow},{i},{x[0][i]},{x[1][i]},{x[2][i]},{v[0][i]},{v[1][i]},{v[2][i]}\n")

    # close file
    f.close()

def leapstep(x, v, n, m, dt):
    """LEAPSTEP: take one step using the leapfrog integrator, formulated
    as a mapping from t to t + dt.  WARNING: this integrator is not
    accurate unless the timestep dt is fixed from one call to another.

    Args:
        x (np.array): positions of all points
        v (np.array): velocities of all points
        n (int): number of points
        m (int): mass of points
        dt (float): timestep for integration
    """
    a = accel(x, n, m)
    v += 0.5 * dt * a  # advance vel by half-step
    x += dt * v  # advance pos by full-step
    a = accel(x, n, m)
    v += 0.5 * dt * a  # and complete vel. step

def accel(x, n, m):
    """ACCEL: compute accelerations for orbiting particles

    Args:
        x (np.array): positions of all points
        n (int): number of points
        m (int): mass of points
    """
    rmin = 25
    G = 1  # 7.0291146e-36  # gravitational constant in units of kpc, 1e8 years, solar mass
    eps = 0.2 * rmin  # softening parameter

    a = np.zeros((3, n))

    # calculate particle--core interactions
    # r: 3 x n
    # x: 3 x n
    for i_core in [0, 343]:
        r = x - x[:, i_core][..., np.newaxis]
        a += -G * m[i_core] / (((r**2).sum(axis=0) + eps**2) ** (3 / 2)) * r

    return a


if __name__ == "__main__":
    main()
