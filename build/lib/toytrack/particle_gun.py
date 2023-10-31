import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List

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

    def generate_particles(self, num_particles: int = 1) -> pd.DataFrame:
        """
        Generate a DataFrame of particles based on the initialized parameters.
        """
        pt = self._generate_values(self.pt, num_particles)
        pphi = self._generate_values(self.pphi, num_particles)
        vx = self._generate_values(self.vx, num_particles)
        vy = self._generate_values(self.vy, num_particles)
        vz = self._generate_values(self.vz, num_particles)
        charge = np.random.choice([-1, 1], size=num_particles)
        
        # Create a DataFrame with the generated values
        particles = pd.DataFrame({
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'pt': pt,
            'pphi': pphi,
            'dimension': self.dimension,
            'charge': charge,
            'particle_id': range(num_particles)
        })

        # Calculate track parameters
        particles = self.calculate_track_parameters(particles)

        return particles

    def calculate_track_parameters(self, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate track parameters for a DataFrame of particles.
        """
        if self.dimension == 2:
            particles['d0'], particles['phi'] = self.calculate_track_parameters_2d(particles)
        elif self.dimension == 3:
            raise NotImplementedError("3D track parameters not implemented yet.")
        
        return particles

    def calculate_track_parameters_2d(self, particles: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        r = 1 / particles['pt']
        x0 = particles['vx'] - particles['charge'] * r * np.cos(particles['pphi'])
        y0 = particles['vy'] - particles['charge'] * r * np.sin(particles['pphi'])
        
        r0_magnitude = np.sqrt(x0**2 + y0**2)
        
        lambda_ = r / r0_magnitude
        
        Px = x0 * (1 - lambda_)
        Py = y0 * (1 - lambda_)
        
        d0 = np.sqrt(Px**2 + Py**2)
        phi = np.arctan2(Py, Px)
        
        return d0, phi

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