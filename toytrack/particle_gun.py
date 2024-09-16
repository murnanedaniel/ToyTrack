import numpy as np
import pandas as pd
from typing import Union, List, Optional

class ParticleGun:
    """
    A class that generates particles with a given kinematic distribution.

    Parameters
    ----------
    num_particles: float, list
        The number of particles to generate. This can be specified in three ways:
        - A single float: This will generate a fixed number of particles.
        - A pair of floats: This will generate a random number of particles with a uniform distribution.
        - A list of three floats: This will generate a random number of particles. The third entry of the list
        is the distribution type (uniform, normal or poisson), and the first two entries are the parameters of the
        distribution. For example, if the third entry is 'uniform', then the first two entries are the minimum and
        maximum of the uniform distribution. If the third entry is 'normal', then the first two entries are the mean
        and standard deviation of the normal distribution. If the third entry is 'poisson', then the first entry is lambda
        and the second is omitted.
    pt: float, list
        The transverse momentum of the particles to be generated. This can be specified in three ways:
        1. A float: All particles will be generated with this transverse momentum.
        2. A list [min, max]: Particles will be generated uniform-random with a transverse momentum in this range.
        3. A list [float, float, dist_type]: Particles will be generated with a transverse momentum in the range 
        specified by the first two elements. The third element, dist_type, specifies the distribution type and can 
        be either 'uniform' or 'normal'. If uniform, then particles will be generated uniform-random in the range.
        If normal, then the first entry is the mean, the second entry is the standard deviation, and particles will
        be generated with a normal distribution.
    pphi: float, list
        The local phi momentum of the particles to be generated. This can be specified in the same ways as pt.
    vx: float, list
        The creation vertex x-coordinate of the particles to be generated. This can be specified in the same ways as pt.
    vy: float, list
        The creation vertex y-coordinate of the particles to be generated. This can be specified in the same ways as pt.
    vz: float, list
        The creation vertex z-coordinate of the particles to be generated. This can be specified in the same ways as pt.
    dimension: int
        The dimension of the space in which the particles are generated. This can be 2 or 3.
    """

    def __init__(self,
                 num_particles: Union[float, List[float], List[Union[float, str]]],
                 pt: Union[float, List[float], List[Union[float, str]]],
                 pphi: Union[float, List[float], List[Union[float, str]]],
                 vx: Union[float, List[float], List[Union[float, str]]],
                 vy: Union[float, List[float], List[Union[float, str]]],
                 vz: Optional[Union[float, List[float], List[Union[float, str]]]] = None,
                 dimension: int = 2):
        """
        Initialize the ParticleGun with the given parameters.
        """
        self.dimension = dimension
        self.num_particles = num_particles
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

    def generate_particles(self) -> pd.DataFrame:
        """
        Generate a DataFrame of particles based on the initialized parameters.
        """
        # Generate the number of particles
        num_particles = max(int(self._generate_values(self.num_particles, size=None)), 1)

        pt = self._generate_values(self.pt, num_particles)
        pphi = self._generate_values(self.pphi, num_particles)
        vx = self._generate_values(self.vx, num_particles)
        vy = self._generate_values(self.vy, num_particles)
        vz = self._generate_values(self.vz, num_particles)
        charge = np.random.choice([-1, 1], size=num_particles)

        # make sure that pt is positive
        pt = np.clip(pt, 0, None)
        
        # Create a DataFrame with the generated values
        particles = pd.DataFrame({
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'pt': pt,
            'pphi': pphi,
            'dimension': self.dimension,
            'charge': charge,
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

    def calculate_track_parameters_2d(self, particles: pd.DataFrame) -> List[pd.Series]:
        r = 1 / particles['pt']
        x0 = particles['vx'] - particles['charge'] * r * np.cos(particles['pphi'])
        y0 = particles['vy'] - particles['charge'] * r * np.sin(particles['pphi'])
        
        r0_magnitude = np.hypot(x0, y0)
        
        lambda_ = r / r0_magnitude
        
        Px = x0 * (1 - lambda_)
        Py = y0 * (1 - lambda_)
        
        d0 = np.hypot(Px, Py)
        phi = np.arctan2(Py, Px)
        
        return [d0, phi]

    def _generate_values(self, value: Union[float, List[float], List[Union[float, str]]], size: Optional[int]) -> np.ndarray:
        """
        Helper method to generate an array of values based on the input which can be a float or a list.
        """
        if isinstance(value, float) or isinstance(value, int):
            return np.full(size, value, dtype=float)
        elif isinstance(value, list):
            if len(value) == 2:
                return np.random.uniform(*value, size=size)
            elif len(value) == 3 and value[2] == 'uniform':
                return np.random.uniform(*value[:2], size=size)
            elif len(value) == 3 and value[2] == 'normal':
                return np.random.normal(*value[:2], size=size)
            elif len(value) == 3 and value[2] == 'poisson':
                return np.random.poisson(value[0], size=size)
        else:
            raise ValueError("Value must be either a float or a list.")
            
    def __repr__(self):
        return f"ParticleGun(dimension={self.dimension}, pt={self.pt}, pphi={self.pphi}, vx={self.vx}, vy={self.vy}, vz={self.vz})"