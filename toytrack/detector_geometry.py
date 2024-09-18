import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List

from .particle_gun import ParticleGun



class Hit:
    """
    A class that represents a hit.
    
    Parameters
    ----------
    x: float
        The x-coordinate of the hit.
    y: float
        The y-coordinate of the hit.
    z: float
        The z-coordinate of the hit.
    r: float
        The radial coordinate of the hit.
    phi: float
        The azimuthal angle of the hit.
    layer: int
    module: int
    """

    def __init__(self, x: float = None, y: float = None, z: float = None, r: float = None, phi: float = None, layer: int = None, module: int = None):
        """
        Initialize the Hit with the given parameters.
        The coordinates can be either cartesian (x, y, z) or cylindrical (r, phi, z).
        If cartesian coordinates are provided, cylindrical coordinates are calculated and vice versa.
        """
        if x is not None and y is not None and z is not None:
            self.x = x
            self.y = y
            self.z = z
            self.r = (x**2 + y**2)**0.5
            self.phi = np.arctan2(y, x)
        elif r is not None and phi is not None and z is not None:
            self.r = r
            self.phi = phi
            self.z = z
            self.x = r * np.cos(phi)
            self.y = r * np.sin(phi)
        else:
            raise ValueError("Either cartesian (x, y, z) or cylindrical (r, phi, z) coordinates must be provided.")
        self.layer = layer
        self.module = module


class Detector:
    """
    A class that represents a detector geometry.

    Parameters
    ----------
    dimension: int
        The dimension of the space in which the detector is located. This can be 2 or 3.
    shape: str
        The shape of the detector. This can be 'plane', 'cylinder' or 'sphere'.
    radius: float
        The radius of the detector. This is only used if shape is 'cylinder' or 'sphere'.
    length: float
        The length of the detector. This is only used if shape is 'cylinder' or 'plane'.
    layer_spacing: float
        The spacing between layers of the detector.
    number_of_layers: int
        The number of layers in the detector.      
    hole_inefficiency: int, float
        If specified: If a float, then this is the inefficiency of each layer - i.e. the
        probability that a hit will not be recorded. If an int, then this is the number of
        holes that will be guaranteed per particle.
    """

    def __init__(self, dimension: int, hole_inefficiency: Union[int, float] = 0, layer_safety_guarantee: bool = False):
        """
        Initialize the Detector with the given parameters.
        """
        self.dimension = dimension
        self.hole_inefficiency = hole_inefficiency
        self.layer_safety_guarantee = layer_safety_guarantee
        self.layers = []

    def add_from_template(self, template: str = None, **kwargs):
        """
        Add a detector from a template.
        """
        if template == 'barrel':
            self.add_barrel(**kwargs)
        elif template == 'endcap':
            self.add_endcap(**kwargs)
        else:
            raise ValueError("Template must be 'barrel' or 'endcap'.")

        return self

    def add_barrel(self, min_radius: float, max_radius: float, number_of_layers: int, length: float = None):
        """
        Add a barrel detector. Layers are spaced evenly between min_radius and max_radius.
        """
        
        barrel_layers = []
        radius_spacing = (max_radius - min_radius) / (number_of_layers - 1)
        for layer in range(number_of_layers):
            layer_radius = min_radius + layer * radius_spacing
            barrel_layers.append({
                'shape': 'cylinder',
                'radius': layer_radius,
                'length': length,
            })

        self.layers.extend(barrel_layers)

    def add_endcap(self, min_radius: float, max_radius: float, min_z, max_z, layer_spacing: float, number_of_layers: int):
        """
        Add an endcap detector. This is a series of annuli, evenly spaced in z.
        """
        endcap_layers = []
        for layer in range(number_of_layers):
            z = min_z + layer * layer_spacing
            endcap_layers.append({
                'shape': 'annulus',
                'min_radius': min_radius,
                'max_radius': max_radius,
                'z': z,
            })           

        self.layers.extend(endcap_layers)
        

    def generate_hits(self, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a DataFrame of hits based on the given particles.
        """
        if self.layer_safety_guarantee and self.hole_inefficiency > 0:
            raise ValueError("Cannot have both layer safety guarantee and hole inefficiency.")

        # Generate the hits
        hits_df = self._generate_hits(particles)

        if self.layer_safety_guarantee:
            hits_df, particles = self._apply_simple_layer_safety_guarantee(hits_df, particles)
        elif self.hole_inefficiency > 0:
            hits_df = self._generate_holes(hits_df)

        return hits_df, particles

    def generate_noise(self, hits_df: pd.DataFrame, num_noise: int) -> pd.DataFrame:
        """
        Generate a DataFrame of noise hits.
        """
        # Generate the noise
        hits_df = self._generate_noise(hits_df, num_noise)

        return hits_df

    def _generate_noise(self, hits_df: pd.DataFrame, num_noise: int) -> pd.DataFrame:
        """
        Helper method to generate noise hits. Samples a random set of points that satisfy the detector geometry.
        """

        # Generate a random angle for each noise hit
        phi = np.random.uniform(0, 2 * np.pi, num_noise)

        # Generate a random layer index for each noise hit
        layer_indices = np.random.randint(0, len(self.layers), num_noise)

        # Create a lookup array for the radii of the layers
        radii = np.array([layer['radius'] for layer in self.layers])

        # Get the radius of the selected layer for each noise hit
        r = radii[layer_indices]

        # Convert the polar coordinates to Cartesian coordinates
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        # Create a DataFrame of the noise hits
        noise_df = pd.DataFrame({
            'x': x,
            'y': y,
            'particle_id': -1,
        })

        # Mix together hits and noise randomly
        hits_df = pd.concat([hits_df, noise_df])
        hits_df = hits_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return hits_df

    
    def _generate_hits(self, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to generate a list of hits based on the given particle.
        """
        # Calculate the intersection points of the particle trajectory with the detector
        hits = self._calculate_intersection_points(particles)

        # Reset the index on the hits
        hits.reset_index(drop=True, inplace=True)
        
        return hits

    def _generate_holes(self, hits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to generate holes in the hits DataFrame.
        """
        # Calculate the number of holes to generate
        if isinstance(self.hole_inefficiency, float):
            hits_df = self._generate_holes_probabilistic(hits_df)
        elif isinstance(self.hole_inefficiency, int):
            hits_df = self._generate_holes_deterministic(hits_df)

        return hits_df
    
    def _generate_holes_probabilistic(self, hits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Each hit is given a random number between 0 and 1. If this number is less than the hole inefficiency,
        then the hit is removed.
        """
        random_hole_probability = np.random.uniform(size=len(hits_df))
        hits_df = hits_df[random_hole_probability > self.hole_inefficiency]

        return hits_df
    
    def _generate_holes_deterministic(self, hits_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each particle, remove a random number of hits equal to the hole inefficiency.
        """

        # Remove a random number of hits for each particle
        holes = hits_df.groupby('particle_id').sample(n=self.hole_inefficiency, random_state=42)
        
        if self.hole_inefficiency > 1:
            holes = holes.droplevel(0)

        # Drop these holes from the hits_df
        hits_df = hits_df.drop(holes.index).reset_index(drop=True)

        return hits_df
    
    def _calculate_intersection_points(self, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to calculate the intersection points of the particle trajectory with the detector.
        """
        # Calculate the intersection points of the particle trajectory with the detector
        if self.dimension == 2:
            intersection_points = self._calculate_intersection_points_2d(particles)
        elif self.dimension == 3:
            intersection_points = self._calculate_intersection_points_3d(particles)
        else:
            raise ValueError("Dimension must be 2 or 3.")

        return intersection_points
    
    def _calculate_intersection_points_2d(self, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to calculate the intersection points of the particle trajectory with the detector in 2D.
        """
        # Calculate the intersection points of the particle trajectory with the detector
        intersection_points = []
        intersection_points.append(self._calculate_intersection_points_2d_cylinder(particles))
        
        intersection_points = pd.concat(intersection_points)

        return intersection_points

    def _calculate_intersection_points_3d(self, particles: pd.DataFrame) -> List[Tuple[float, float, float]]:
        raise NotImplementedError("3D intersection points not implemented yet.")
    
    def _calculate_intersection_points_2d_plane(self, particles: pd.DataFrame) -> List[Tuple[float, float, float]]:
        raise NotImplementedError("2D plane intersection points not implemented yet.")
    
    def _calculate_intersection_points_2d_cylinder(self, particles_df: pd.DataFrame) -> List[Tuple[float, float, float]]:
        """
        Helper method to calculate the intersection points of the particle trajectory with the detector in 2D.
        """
        layers_df = pd.DataFrame(self.layers)
        # Narrow to only the layers that are cylinders
        layers_df = layers_df[layers_df['shape'] == 'cylinder']

        particle_layer_pairs = pd.merge(particles_df, layers_df, how='cross')

        # Find intersections
        self._find_intersections_df(particle_layer_pairs)

        # Filter valid intersections
        self._filter_points_on_segment_df(particle_layer_pairs)

        hits = pd.concat([
            particle_layer_pairs[particle_layer_pairs['valid1']][['x1', 'y1', 'particle_id']].rename(columns={'x1': 'x', 'y1': 'y'}),
            particle_layer_pairs[particle_layer_pairs['valid2']][['x2', 'y2', 'particle_id']].rename(columns={'x2': 'x', 'y2': 'y'}),
        ])

        return hits

    def _find_intersections_df(self, df):
        r = df['pt']
        x0 = df['vx'] - df['charge'] * r * np.cos(df['pphi'])
        y0 = df['vy'] - df['charge'] * r * np.sin(df['pphi'])

        # distance between circles R
        R = np.sqrt(x0**2 + y0**2)

        a = (r**2 - df['radius']**2) / (2 * R)
        b1 = np.sqrt(
            (r**2 + df['radius']**2)/2 - 
            (r**2 - df['radius']**2)**2 / (4 * R**2) -
            R**2 / 4
            )
        b2 = -b1

        # Calculate the intersection points
        x1 = x0/2 - a * x0/R + b1 * y0/R
        y1 = y0/2 - a * y0/R - b1 * x0/R
        x2 = x0/2 - a * x0/R + b2 * y0/R
        y2 = y0/2 - a * y0/R - b2 * x0/R

        df["x0"], df["y0"], df["x1"], df["y1"], df["x2"], df["y2"] = x0, y0, x1, y1, x2, y2

    def _filter_points_on_segment_df(self, df):
        """
        A vectorized version of the above loop. Takes a list of 
        (x1, y1), (x2, y2) and creates two new columns, with a
        true or false entry for each point.
        """
    
        # Calculate the angle of the intersection point relative to the trajectory's starting point
        angle1 = np.arctan2(df['y1'] - df['y0'], df['x1'] - df['x0'])
        angle2 = np.arctan2(df['y2'] - df['y0'], df['x2'] - df['x0'])

        # Adjust the angle based on the charge
        lower_bound = np.where(df['charge'] == -1, df['pphi'] + np.pi - np.pi/2, df['pphi'])
        upper_bound = np.where(df['charge'] == -1, df['pphi'] + np.pi, df['pphi'] + np.pi/2)

        # Check if the angle falls within the delta_theta range
        df['valid1'] = ((lower_bound <= angle1) & (angle1 <= upper_bound)) | ((lower_bound <= angle1 + 2*np.pi) & (angle1 + 2*np.pi <= upper_bound))
        df['valid2'] = ((lower_bound <= angle2) & (angle2 <= upper_bound)) | ((lower_bound <= angle2 + 2*np.pi) & (angle2 + 2*np.pi <= upper_bound))

    def _apply_simple_layer_safety_guarantee(self, hits_df: pd.DataFrame, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that each particle has exactly one hit per layer by removing particles
        that don't meet this criterion.
        """
        # Count the number of hits per particle
        hits_per_particle = hits_df.groupby('particle_id').size()

        # Identify particles with the correct number of hits
        valid_particles = hits_per_particle[hits_per_particle == len(self.layers)].index

        # Keep only hits from valid particles and noise hits
        valid_hits = hits_df[hits_df['particle_id'].isin(valid_particles) | (hits_df['particle_id'] == -1)]

        # Keep only valid particles
        valid_particles_df = particles[particles.index.isin(valid_particles)]

        return valid_hits.reset_index(drop=True), valid_particles_df

    def __repr__(self):
        return f"Detector(dimension={self.dimension}), layers: {self.layers}"
