import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List

from .particle_gun import ParticleGun, Particle



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
    """

    def __init__(self, dimension: int):
        """
        Initialize the Detector with the given parameters.
        """
        self.dimension = dimension
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
        

    def generate_hits(self, particles: List[Particle]) -> pd.DataFrame:
        """
        Generate a DataFrame of hits based on the given particles.
        """
        # Generate the hits
        hits_df = self._generate_hits(particles)

        return hits_df
    
    def _generate_hits(self, particles: List[Particle]) -> pd.DataFrame:
        """
        Helper method to generate a list of hits based on the given particle.
        """
        # Calculate the intersection points of the particle trajectory with the detector
        hits = self._calculate_intersection_points(particles)

        return hits
    
    def _calculate_intersection_points(self, particles: List[Particle]) -> pd.DataFrame:
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
    
    def _calculate_intersection_points_2d(self, particles: List[Particle]) -> pd.DataFrame:
        """
        Helper method to calculate the intersection points of the particle trajectory with the detector in 2D.
        """
        # Calculate the intersection points of the particle trajectory with the detector
        intersection_points = []
        intersection_points.append(self._calculate_intersection_points_2d_cylinder(particles))
        
        intersection_points = pd.concat(intersection_points)

        return intersection_points

    def _calculate_intersection_points_3d(self, particle: Particle) -> List[Tuple[float, float, float]]:
        raise NotImplementedError("3D intersection points not implemented yet.")
    
    def _calculate_intersection_points_2d_plane(self, particle: Particle) -> List[Tuple[float, float, float]]:
        raise NotImplementedError("2D plane intersection points not implemented yet.")
    
    def _calculate_intersection_points_2d_cylinder(self, particles: List[Particle]) -> List[Tuple[float, float, float]]:
        """
        Helper method to calculate the intersection points of the particle trajectory with the detector in 2D.
        """
        # Convert particle list to a pandas dataframe
        particles_df = pd.DataFrame(vars(particle) for particle in particles)
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

    def __repr__(self):
        return f"Detector(dimension={self.dimension}), layers: {self.layers}"
