from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .particle_gun import ParticleGun
from .detector_geometry import Detector

class Event:
    """
    Class to store an event for the tracking micro challenges.

    Parameters
    ----------
    particles: pd.DataFrame
        A DataFrame of particles
    hits: pd.DataFrame
        A DataFrame of hits
    tracks: np.ndarray
        A 2xN numpy array of the truth tracks.
    detector: Detector
        The detector used to generate the hits.
    """
    
    def __init__(self, particles: pd.DataFrame, hits: pd.DataFrame, tracks: np.ndarray, detector: Detector):
        """
        Initialize the Event with the given parameters.
        """
        self.particles = particles
        self.hits = hits
        self.tracks = tracks
        self.detector = detector


    def __repr__(self):
        return f"Event(particles={self.particles}, hits={self.hits}), tracks={self.tracks}, detector={self.detector})"

    def display(self):
        fig, ax = plt.subplots()

        # Radii for the cylindrical layers
        radii = [layer["radius"] for layer in self.detector.layers if self.detector.dimension == 2 and layer["shape"] == "cylinder"]

        # Plot each cylindrical layer
        for radius in radii:
            circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--')
            ax.add_patch(circle)

        # Plot the tracks
        ax.plot(self.hits.x.values[self.tracks], self.hits.y.values[self.tracks], color='k', linewidth=1, alpha=0.3, label='Truth Tracks')

        # Separate noise and non-noise hits
        noise_hits = self.hits[self.hits["particle_id"] == -1]
        non_noise_hits = self.hits[self.hits["particle_id"] != -1]

        # Plot the noise hits
        ax.scatter(noise_hits['x'], noise_hits['y'], color='gray', s=10, label='Noise')

        # Plot the non-noise hits
        cmap = plt.get_cmap('jet')  # Get the 'jet' colormap
        colors = cmap(non_noise_hits["particle_id"]/non_noise_hits["particle_id"].max())  # Apply the colormap to your normalized particle IDs
        ax.scatter(non_noise_hits['x'], non_noise_hits['y'], color=colors, s=10, label='Hits')

        max_radius = max(radii)

        ax.set_aspect('equal')
        ax.set_xlim(-max_radius-1, max_radius+1)
        ax.set_ylim(-max_radius-1, max_radius+1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()



class EventGenerator:
    """
    Class to generate events for the tracking micro challenges.

    Parameters
    ----------
    particle_gun: ParticleGun or List[ParticleGun]
        The particle gun(s) to use to generate particles.
    detector: Detector
        The detector to use to generate hits.
    noise: float, list
        The amount of noise to add to the hits. This can be specified in three ways:
        - A single float: This will add a fixed fraction of noise hits.
        - A pair of floats: This will add a random fraction of noise hits with a uniform distribution.
        - A list of three elements: This will add a random fraction of noise hits. The third entry of the list
        is the distribution type (uniform, normal or poisson), and the first two entries are the parameters of the
        distribution.
        If the values are given as non-zero integers, then the noise is added as a fixed number of hits.
        If the values are given as floats between 0 and 1, then the noise is added as a fraction of the number of non-noise hits.
    """

    def __init__(self, 
                 particle_gun: Union[ParticleGun, List[ParticleGun]], 
                 detector: Detector, 
                 noise: Union[float, List[float], List[Union[float, str]], int, List[int], List[Union[int, str]]] = None,
                ):
        """
        Initialize the EventGenerator with the given parameters.
        """
        self.particle_gun = particle_gun if isinstance(particle_gun, list) else [particle_gun]
        self.detector = detector
        self.noise = noise

    def generate_event(self):
        """
        Generate an event based on the initialized parameters.
        """
        # Generate particles from all guns and concatenate them
        particles = pd.concat([gun.generate_particles() for gun in self.particle_gun], ignore_index=True)
        particles["particle_id"] = particles.index

        # Generate the hits
        hits, particles = self.detector.generate_hits(particles)

        # Add noise to the hits
        if self.noise is not None:
            num_noise = self._get_num_noise(self.noise, hits)
            hits = self.detector.generate_noise(hits, num_noise)

        # Generate the truth tracks
        tracks = self._generate_truth_tracks(particles, hits)

        # Create an Event instance with the generated particles and hits
        event = Event(particles, hits, tracks, self.detector)

        return event

    def _get_num_noise(self, noise: Union[float, List[float], List[Union[float, str]], int, List[int], List[Union[int, str]]], hits: pd.DataFrame) -> int:
        """
        Helper method to generate the number of noise hits based on the input which can be a float, list or int.
        If the inputs are floats, then first convert to a raw number of hits, by multiplying by the number of hits.
        """

        num_hits = len(hits)
        if isinstance(noise, float):
            noise = int(noise*num_hits)
        elif isinstance(noise, list) and len(noise) == 2 and isinstance(noise[0], float):
            noise = [int(noise[0]*num_hits), int(noise[1]*num_hits)]
        elif isinstance(noise, list) and len(noise) == 3 and isinstance(noise[0], float):
            noise = [int(noise[0]*num_hits), int(noise[1]*num_hits), noise[2]]
        
        return int(self._generate_value(noise))
        
    
    def _generate_truth_tracks(self, particles: pd.DataFrame, hits: pd.DataFrame) -> np.ndarray:
        """
        Generates the 2xN list of edges for the truth tracks, as a numpy array

        Parameters
        ----------
        particles: pd.DataFrame
            The particles DataFrame.
        hits: pd.DataFrame
            The hits DataFrame.

        Returns
        -------
        truth_tracks: np.ndarray
            The truth tracks as a 2xN numpy array.
        """

        # Remove noise hits
        non_noise_hits = hits[hits["particle_id"] != -1].reset_index()
        non_noise_hits.rename(columns={'index': 'hit_id'}, inplace=True)

        # Sort the hits by particle ID and then by R
        merged_hits = non_noise_hits.merge(particles, on='particle_id')
        merged_hits['R'] = np.sqrt((merged_hits["x"] - merged_hits["vx"])**2 + (merged_hits["y"] - merged_hits["vy"])**2)
        sorted_hits = merged_hits.sort_values(by=['particle_id', 'R'])

        # Get the edges of the tracks
        track_edges = np.stack([
            sorted_hits.hit_id.values[:-1],
            sorted_hits.hit_id.values[1:]
        ], axis=0)

        # Remove edges that don't have the same particle ID, just in case
        track_edges = track_edges[:, hits.particle_id.values[track_edges[0]] == hits.particle_id.values[track_edges[1]]]

        return track_edges


    def _generate_value(self, value: Union[float, List[float], List[Union[float, str]]]) -> float:
        """
        Helper method to generate a value based on the input which can be a float or a list.
        """
        if isinstance(value, int) or isinstance(value, float):
            return value
        elif isinstance(value, list):
            if len(value) == 2:
                return np.random.uniform(*value)
            elif len(value) == 3 and value[2] == 'uniform':
                return np.random.uniform(*value[:2])
            elif len(value) == 3 and value[2] == 'normal':
                return np.random.normal(*value[:2])
            elif len(value) == 3 and value[2] == 'poisson':
                return np.random.poisson(value[0])
