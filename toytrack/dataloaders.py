import numpy as np
import pandas as pd
from toytrack import ParticleGun, Detector, EventGenerator, Event
from typing import Optional, Union, List, Dict, Iterator, Tuple, Callable
from .transforms import TrackletPatchify

try:
    import torch
    from torch.utils.data import IterableDataset
    from torch.nn.utils.rnn import pad_sequence
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class TracksDataset(IterableDataset):
        """An iterable dataset for generating toy tracking data.

        This dataset generates events with particle tracks and detector hits based on the provided configuration.
        It can output data in either a hitwise or trackwise structure.

        Args:
            config (Dict): A dictionary containing configuration parameters for the dataset.
            transform (Optional[Callable]): An optional transform to apply to each generated sample.

        Attributes:
            config (Dict): The configuration dictionary.
            detector (Detector): The detector object used for generating hits.
            particle_guns (List[ParticleGun]): List of particle gun objects for generating particles.
            event_generator (EventGenerator): The event generator object.
            structure (str): The output structure, either 'hitwise' or 'trackwise'.
            transform (Optional[Callable]): The transform to apply to each sample.

        Example:
            >>> config = {
            ...     'detector': {'dimension': 2, 'min_radius': 0.5, 'max_radius': 3.0},
            ...     'particle_guns': [{'num_particles': [1, 5], 'pt': [1, 5]}],
            ...     'noise': 0.1,
            ...     'structure': 'trackwise'
            ... }
            >>> dataset = TracksDataset(config)
            >>> for sample in dataset:
            ...     # Process the sample
            ...     pass
        """

        def __init__(self, config: Dict, transform: Optional[Callable] = None):
            self.config = config
            self.detector = self._create_detector()
            self.particle_guns = self._create_particle_guns()
            self.event_generator = self._create_event_generator()
            self.structure = config.get('structure', 'hitwise')
            self.transform = transform

        def _create_detector(self) -> Detector:
            detector_config = self.config.get('detector', {})
            return Detector(
                dimension=detector_config.get('dimension', 2),
                layer_safety_guarantee=detector_config.get('layer_safety_guarantee', False),
                hole_inefficiency=detector_config.get('hole_inefficiency', 0)
            ).add_from_template(
                'barrel', 
                min_radius=detector_config.get('min_radius', 0.5),
                max_radius=detector_config.get('max_radius', 3.),
                number_of_layers=detector_config.get('number_of_layers', 10),
            )

        def _create_particle_guns(self) -> List[ParticleGun]:
            gun_configs = self.config.get('particle_guns', [])
            return [self._create_single_particle_gun(gun_config) for gun_config in gun_configs]

        def _create_single_particle_gun(self, gun_config: Dict) -> ParticleGun:
            return ParticleGun(
                dimension=gun_config.get('dimension', 2),
                num_particles=gun_config.get('num_particles', [1, None, 'poisson']),
                pt=gun_config.get('pt', [1, 5]),
                pphi=gun_config.get('pphi', [-np.pi, np.pi]),
                vx=gun_config.get('vx', [0, self.config.get('d0', 0.1) * 0.5**0.5, 'normal']),
                vy=gun_config.get('vy', [0, self.config.get('d0', 0.1) * 0.5**0.5, 'normal']),
                vz=gun_config.get('vz', 0) if gun_config.get('dimension', 2) == 3 else None
            )

        def _create_event_generator(self) -> EventGenerator:
            return EventGenerator(
                particle_gun=self.particle_guns,
                detector=self.detector,
                noise=self.config.get('noise')
            )

        def __iter__(self) -> Iterator[Dict]:
            """
            This function generates events and yields them as dictionaries.
            The structure of the output dictionary depends on the 'structure' parameter in the config.
            If 'structure' is 'hitwise', then:
            - 'x' is a tensor of shape (num_hits, 2) containing the x and y coordinates of the hits.
            - 'mask' is a tensor of shape (num_hits,) containing the mask of the hits.
            - 'pids' is a tensor of shape (num_hits,) containing the particle IDs of the hits.
            - 'event' is the event object.

            If 'structure' is 'trackwise', then hits are grouped into tracks:
            - 'x' is a tensor of shape (num_tracks, num_hits_max_per_track, 2) containing the x and y coordinates of the hits.
            - 'mask' is a tensor of shape (num_tracks, num_hits_max_per_track) containing the mask of the hits.
            - 'pids' is a tensor of shape (num_tracks,) containing the particle IDs of the tracks.
            - 'event' is the event object.
            """

            while True:
                event = self.event_generator.generate_event()
                output = {}

                if self.structure == 'hitwise':
                    output['x'] = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
                    output['mask'] = torch.ones(len(event.hits), dtype=torch.bool)
                    output['pids'] = torch.tensor(event.hits.particle_id, dtype=torch.long)
                    output['event'] = event

                elif self.structure == 'trackwise':
                    tracks_x, tracks_mask, tracks_pids = self.group_hits_into_tracks(event)
                    output['x'] = tracks_x
                    output['mask'] = tracks_mask
                    output['pids'] = tracks_pids
                    output['event'] = event

                if self.transform:
                    output = self.transform(output)

                yield output

        def group_hits_into_tracks(self, event: Event) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            This function groups hits into tracks.
            """
            hits = event.hits
            particles = event.particles

            # 1. Remove noise hits
            non_noise_hits = hits[hits["particle_id"] != -1].reset_index()
            non_noise_hits.rename(columns={'index': 'hit_id'}, inplace=True)
            non_noise_hits = non_noise_hits.merge(particles, on="particle_id")

            # 2. Calculate R
            non_noise_hits['R'] = np.sqrt((non_noise_hits["x"] - non_noise_hits["vx"])**2 + (non_noise_hits["y"] - non_noise_hits["vy"])**2)

            # 3. Sort hits by particle ID and R
            sorted_hits = non_noise_hits.sort_values(by=['particle_id', 'R'])

            # 4. Group hits by particle ID, and apply a conversion to a torch tensor
            grouped_hits = sorted_hits.groupby('particle_id')
            grouped_hits_positions = grouped_hits[['x', 'y']].apply(lambda x: torch.tensor(x.values, dtype=torch.float))

            # 5. Convert to pytorch
            # Get particle ids, which are the index of the grouped hits
            trackwise_particle_ids = torch.tensor(grouped_hits_positions.index.values, dtype=torch.long)

            # Pad the hit positions
            trackwise_hit_positions = pad_sequence(grouped_hits_positions.values, batch_first=True, padding_value=0)

            # Create masks (True if hit position is not 0)
            trackwise_masks = (trackwise_hit_positions != 0).any(dim=-1)

            return trackwise_hit_positions, trackwise_masks, trackwise_particle_ids

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict:
            collated = {}
            for key in batch[0].keys():
                if key == 'event':
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = torch.nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True)
            return collated

else:
    def _torch_not_installed_error(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Please install ToyTrack with PyTorch support: pip install toytrack[torch]")

    TracksDataset = _torch_not_installed_error