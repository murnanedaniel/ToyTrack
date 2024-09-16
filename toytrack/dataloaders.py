import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import numpy as np
from toytrack import ParticleGun, Detector, EventGenerator
from typing import Optional, Union, List

# danieltm/ExaTrkX/ExaLibraries/ToyTrack/toytrack/dataset.py

from typing import Dict, List, Union, Optional, Iterator, Tuple
import torch
from torch.utils.data import IterableDataset
import numpy as np
import pandas as pd

from .particle_gun import ParticleGun
from .detector_geometry import Detector
from .event_generator import EventGenerator, Event

class TracksDataset(IterableDataset):
    def __init__(self, config: Dict):
        self.config = config
        self.detector = self._create_detector()
        self.particle_guns = self._create_particle_guns()
        self.event_generator = self._create_event_generator()

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

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Event]]:
        while True:
            event = self.event_generator.generate_event()
            x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
            mask = torch.ones(x.shape[0], dtype=torch.bool)
            
            # Decide what to yield as the third element based on config
            if self.config.get('yield_particle_ids', False):
                third_element = torch.tensor(event.hits.particle_id.values, dtype=torch.long)
            else:
                # For variable tracks, we'll yield a single label for the entire event
                third_element = torch.tensor([0], dtype=torch.float)  # You may want to adjust this based on your needs

            yield x, mask, third_element, event

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Event]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Event]]:
        x, mask, third_element, events = zip(*batch)
        return (
            torch.nn.utils.rnn.pad_sequence(x, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(mask, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(third_element, batch_first=True) if third_element[0].dim() > 0 else torch.cat(third_element),
            list(events)
        )

class TracksDatasetVariable(IterableDataset):
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            minbias_lambda: Optional[float] = 50,
            pileup_lambda: Optional[float] = 45,
            hard_proc_lambda: Optional[float] = 5,
            minbias_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            pileup_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            hard_proc_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
            min_radius: Optional[float] = 0.5,
            max_radius: Optional[float] = 3.,
        ):
        super().__init__()

        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.minbias_lambda = minbias_lambda
        self.pileup_lambda = pileup_lambda
        self.hard_proc_lambda = hard_proc_lambda
        self.minbias_pt_dist = minbias_pt_dist
        self.pileup_pt_dist = pileup_pt_dist
        self.hard_proc_pt_dist = hard_proc_pt_dist
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __iter__(self):
        return _TrackIterableVariable(
            self.hole_inefficiency,
            self.d0,
            self.noise,
            self.minbias_lambda,
            self.pileup_lambda,
            self.hard_proc_lambda,
            self.minbias_pt_dist,
            self.pileup_pt_dist,
            self.hard_proc_pt_dist,
            self.min_radius,
            self.max_radius
        )
    
class _TrackIterableVariable:
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            minbias_lambda: Optional[float] = 50,
            pileup_lambda: Optional[float] = 45,
            hard_proc_lambda: Optional[float] = 5,
            minbias_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            pileup_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            hard_proc_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
            min_radius: Optional[float] = 0.5,
            max_radius: Optional[float] = 3.,
        ):
        
        detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel', 
            min_radius=min_radius, 
            max_radius=max_radius, 
            number_of_layers=10,
        )
        
        self.minbias_gun = ParticleGun(
            dimension=2, 
            num_particles=[minbias_lambda, None, "poisson"], 
            pt=minbias_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.pileup_gun = ParticleGun(
            dimension=2, 
            num_particles=[pileup_lambda, None, "poisson"],
            pt=pileup_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.hard_proc_gun = ParticleGun(
            dimension=2, 
            num_particles=[hard_proc_lambda, None, "poisson"],
            pt=hard_proc_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.minbias_gen = EventGenerator(self.minbias_gun, detector, noise)
        self.hard_proc_gen = EventGenerator([self.pileup_gun, self.hard_proc_gun], detector, noise)
        
        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.minbias_lambda = minbias_lambda
        self.pileup_lambda = pileup_lambda
        self.hard_proc_lambda = hard_proc_lambda
        self.minbias_pt_dist = minbias_pt_dist
        self.pileup_pt_dist = pileup_pt_dist
        self.hard_proc_pt_dist = hard_proc_pt_dist
        self.y = np.random.rand() < 0.5
    
    def __next__(self):
        
        self.y = not self.y
        
        if self.y:
            event = self.hard_proc_gen.generate_event()
        else:
            event = self.minbias_gen.generate_event()
            
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        mask = torch.ones(x.shape[0], dtype=bool)

        return x, mask, torch.tensor([self.y], dtype=torch.float), event
    
def collate_fn_variable(ls):
    x, mask, y, events = zip(*ls)
    return pad_sequence(x, batch_first=True), pad_sequence(mask, batch_first=True), torch.cat(y), list(events)


class TracksDatasetFixed(IterableDataset):
    
    def __init__(
        self,
        hole_inefficiency: Optional[float] = 0,
        d0: Optional[float] = 0.1,
        noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
        num_particles: Optional[float] = 5,
        pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
        min_radius: Optional[float] = 0.5,
        max_radius: Optional[float] = 3.,
    ):
        super().__init__()

        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.num_particles = num_particles
        self.pt_dist = pt_dist
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __iter__(self):
        return _TrackIterableFixed(
            self.hole_inefficiency,
            self.d0,
            self.noise,
            self.num_particles,
            self.pt_dist,
            self.min_radius,
            self.max_radius
        )
    
class _TrackIterableFixed:
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            num_particles: Optional[float] = 5,
            pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
            min_radius: Optional[float] = 0.5,
            max_radius: Optional[float] = 3.,
        ):

        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.num_particles = num_particles
        self.pt_dist = pt_dist
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel', 
            min_radius=min_radius, 
            max_radius=max_radius, 
            number_of_layers=8,
        )

        self.particle_gun = ParticleGun(
            dimension=2, 
            pt=pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )
        self.particle_gen = EventGenerator(particle_gun = self.particle_gun, 
                                           detector = detector, 
                                           num_particles = self.num_particles, 
                                           noise = self.noise)
        
        

    def __next__(self):
        
        event = self.particle_gen.generate_event()
            
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        mask = torch.ones(x.shape[0], dtype=bool)
        pids = torch.tensor([event.hits.particle_id], dtype=torch.long).squeeze()

        return x, mask, pids, event

def collate_fn_fixed(ls):
    x, mask, pids, events = zip(*ls)
    return pad_sequence(x, batch_first=True), pad_sequence(mask, batch_first=True), pad_sequence(pids, batch_first=True), list(events)