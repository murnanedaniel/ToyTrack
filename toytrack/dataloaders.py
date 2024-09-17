import numpy as np
from toytrack import ParticleGun, Detector, EventGenerator, Event
from typing import Optional, Union, List, Dict, Iterator, Tuple

try:
    import torch
    from torch.utils.data import IterableDataset
    from torch.nn.utils.rnn import pad_sequence
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class TracksDataset(IterableDataset):
        def __init__(self, config: Dict):
            self.config = config
            self.detector = self._create_detector()
            self.particle_guns = self._create_particle_guns()
            self.event_generator = self._create_event_generator()
            self.outputs = config.get('outputs', {})

        def _create_detector(self) -> Detector:
            detector_config = self.config.get('detector', {})
            return Detector(
                dimension=detector_config.get('dimension', 2),
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
            while True:
                event = self.event_generator.generate_event()
                output = {}

                if self.outputs.get('x', False):
                    output['x'] = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()

                if self.outputs.get('mask', False):
                    output['mask'] = torch.ones(len(event.hits), dtype=torch.bool)

                if self.outputs.get('pids', False):
                    output['pids'] = torch.tensor(event.hits.particle_id, dtype=torch.long)

                if self.outputs.get('event', False):
                    output['event'] = event

                yield output

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