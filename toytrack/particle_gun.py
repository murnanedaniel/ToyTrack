import numpy as np
from typing import Union, Tuple, Optional, List

class Particle:
    """
    A class that represents a particle.

    Parameters
    ----------
    pt: float
        The transverse momentum of the particle.
    dimension: int
        The dimension of the space in which the particle is generated. This can be 2 or 3.
    vx: float
        The creation vertex x-coordinate of the particle.
    vy: float
        The creation vertex y-coordinate of the particle.
    vz: float
        The creation vertex z-coordinate of the particle.
    pphi: float
        The local phi momentum of the particle at the creation vertex.
    pz: float
        The local z momentum of the particle at the creation vertex.
    particle_id: int or str
        The identifier of the particle.
    """
    def __init__(self, vx: float, 
                 vy: float, 
                 pt: float, 
                 pphi: float, 
                 dimension: int = None, 
                 vz: float = None,
                 pz: float = None,
                 charge: int = None,
                 particle_id: Union[int, str] = None):
        """
        Initialize the Particle with the given parameters.
        """
        self.vx = vx
        self.vy = vy
        self.pt = pt
        self.pphi = pphi
        self.dimension = dimension
        self.charge = charge
        self.particle_id = particle_id

        if dimension == 2:
            self.vz = 0 
            self.pz = 0
        elif dimension == 3:
            self.vz = vz
            self.pz = pz
        else:
            raise ValueError("Dimension must be either 2 or 3.")
        
        self.calculate_track_parameters()

    def calculate_track_parameters(self):
        if self.dimension == 2:
            self.d0, self.phi = self.calculate_track_parameters_2d()
        elif self.dimension == 3:
            raise NotImplementedError("3D track parameters not implemented yet.")
        
    def calculate_track_parameters_2d(self):
        r = 1 / self.pt
        x0 = self.vx - self.charge * r * np.cos(self.pphi)
        y0 = self.vy - self.charge * r * np.sin(self.pphi)
        
        r0_magnitude = np.sqrt(x0**2 + y0**2)
        
        lambda_ = r / r0_magnitude
        
        Px = x0 * (1 - lambda_)
        Py = y0 * (1 - lambda_)
        
        d0 = np.sqrt(Px**2 + Py**2)
        phi = np.arctan2(Py, Px)
        
        return d0, phi
        
    def __repr__(self):
        return f"Particle(vx={self.vx}, vy={self.vy}, vz={self.vz}, pt={self.pt}, pphi={self.pphi}, pz={self.pz}, dimension={self.dimension}, charge={self.charge}, particle_id={self.particle_id})"


class ParticleGun:
    """
    A class that generates particles with a given kinematic distribution.

    Parameters
    ----------
    dimension: int
        The dimension of the space in which the particles are generated. This can be 2 or 3.
    pt: float, tuple
        The transverse momentum of the particles to be generated. This can be specified in three ways:
        1. A float: All particles will be generated with this transverse momentum.
        2. A tuple (min, max): Particles will be generated uniform-random with a transverse momentum in this range.
        3. A tuple (float, float, dist_type): Particles will be generated with a transverse momentum in the range 
        specified by the first two elements. The third element, dist_type, specifies the distribution type and can 
        be either 'uniform' or 'normal'. If uniform, then particles will be generated uniform-random in the range.
        If normal, then the first entry is the mean, the second entry is the standard deviation, and particles will
        be generated with a normal distribution.
    vx: float, tuple
        The creation vertex x-coordinate of the particles to be generated. This can be specified in the same ways as pt.
    vy: float, tuple
        The creation vertex y-coordinate of the particles to be generated. This can be specified in the same ways as pt.
    vz: float, tuple
        The creation vertex z-coordinate of the particles to be generated. This can be specified in the same ways as pt.
    pphi: float, tuple
        The local phi momentum of the particles to be generated. This can be specified in the same ways as pt.
    """

    def __init__(self, dimension: int,
                 pt: Union[float, Tuple[float, float], Tuple[float, float, str]],
                 pphi: Union[float, Tuple[float, float], Tuple[float, float, str]],
                 vx: Union[float, Tuple[float, float], Tuple[float, float, str]],
                 vy: Union[float, Tuple[float, float], Tuple[float, float, str]],
                 vz: Optional[Union[float, Tuple[float, float], Tuple[float, float, str]]] = None):
        """
        Initialize the ParticleGun with the given parameters.
        """
        self.dimension = dimension
        self.pt = pt
        self.pphi = pphi
        self.vx = vx
        self.vy = vy
        if dimension == 2:
            self.vz = 0.
        elif dimension == 3:
            self.vz = vz
        else:
            raise ValueError("Dimension must be either 2 or 3.")

    def generate_particles(self, num_particles: int = 1) -> List[Particle]:
        """
        Generate a list of particles based on the initialized parameters.
        """
        pt = self._generate_values(self.pt, num_particles)
        pphi = self._generate_values(self.pphi, num_particles)
        vx = self._generate_values(self.vx, num_particles)
        vy = self._generate_values(self.vy, num_particles)
        vz = self._generate_values(self.vz, num_particles)
        charge = np.random.choice([-1, 1], size=num_particles)
        
        # Create a Particle instance with the generated values
        particles = [Particle(vx = vx[i],
                                vy = vy[i],
                                vz = vz[i],
                                pt = pt[i],
                                pphi = pphi[i],
                                dimension = self.dimension,
                                charge = charge[i],
                                particle_id = i) for i in range(num_particles)]

        return particles

    def _generate_values(self, value: Union[float, Tuple[float, float], Tuple[float, float, str]], size: int) -> np.ndarray:
        """
        Helper method to generate an array of values based on the input which can be a float or a tuple.
        """
        if isinstance(value, float) or isinstance(value, int):
            return np.full(size, value, dtype=float)
        elif isinstance(value, tuple):
            if len(value) == 2:
                return np.random.uniform(*value, size=size)
            elif len(value) == 3 and value[2] == 'uniform':
                return np.random.uniform(*value[:2], size=size)
            elif len(value) == 3 and value[2] == 'normal':
                return np.random.normal(*value[:2], size=size)
        else:
            raise ValueError("Value must be either a float or a tuple.")
            
    def __repr__(self):
        return f"ParticleGun(dimension={self.dimension}, pt={self.pt}, pphi={self.pphi}, vx={self.vx}, vy={self.vy}, vz={self.vz})"