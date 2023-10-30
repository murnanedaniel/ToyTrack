from typing import Tuple, Union, List
import numpy as np

from .particle_gun import ParticleGun, Particle
from .detector_geometry import Detector, Hit

class Event:
    """
    Class to store an event for the tracking micro challenges.

    Parameters
    ----------
    particles: list
        A list of Particle instances.
    hits: list
        A list of Hit instances.
    """
    
    def __init__(self, particles: list, hits: list):
        """
        Initialize the Event with the given parameters.
        """
        self.particles = particles
        self.hits = hits

class EventGenerator:
    """
    Class to generate events for the tracking micro challenges.

    Parameters
    ----------
    particle_gun: ParticleGun
        The particle gun to use to generate particles.
    detector: Detector
        The detector to use to generate hits.
    num_particles: float, tuple
        The number of particles to generate. This can be specified in three ways:
        - A single float: This will generate a fixed number of particles.
        - A pair of floats: This will generate a random number of particles with a uniform distribution.
        - A tuple of three floats: This will generate a random number of particles. The third entry of the tuple
        is the distribution type (uniform, normal or poisson), and the first two entries are the parameters of the
        distribution. For example, if the third entry is 'uniform', then the first two entries are the minimum and
        maximum of the uniform distribution. If the third entry is 'normal', then the first two entries are the mean
        and standard deviation of the normal distribution. If the third entry is 'poisson', then the first two entries
        are the mean and standard deviation of the poisson distribution.
    """

    def __init__(self, particle_gun: ParticleGun, detector: Detector, num_particles: Union[float, Tuple[float, float], Tuple[float, float, str]]):
        """
        Initialize the EventGenerator with the given parameters.
        """
        self.particle_gun = particle_gun
        self.detector = detector
        self.num_particles = num_particles

    def generate_event(self):
        """
        Generate an event based on the initialized parameters.
        """
        # Generate the number of particles
        num_particles = self._generate_value(self.num_particles)

        # Generate the particles
        particles = [self.particle_gun.generate_particle() for _ in range(num_particles)]

        # Generate the hits
        hits = [self.detector.generate_hits(particle) for particle in particles]

        # Create an Event instance with the generated particles and hits
        event = Event(particles, hits)

        return event


    def _generate_value(self, value: Union[float, Tuple[float, float], Tuple[float, float, str]]) -> float:
        """
        Helper method to generate a value based on the input which can be a float or a tuple.
        """
        if isinstance(value, float):
            return value
        elif isinstance(value, tuple):
            if len(value) == 2:
                return np.random.uniform(*value)
            elif len(value) == 3 and value[2] == 'uniform':
                return np.random.uniform(*value[:2])
            elif len(value) == 3 and value[2] == 'normal':
                return np.random.normal(*value[:2])
