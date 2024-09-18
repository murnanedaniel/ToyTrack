# ToyTrack

[![Documentation Status](https://readthedocs.org/projects/toytrack/badge/?version=latest)](https://toytrack.readthedocs.io/en/latest/?badge=latest)

ToyTrack is a Python library for generating toy tracking events for particle physics. 

**The goal**: To produce a "*good-enough*" event simulation, in as few lines as possible (currently 3 lines), as quickly as possible (currently 0.07 seconds for a 10,000-particle event).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ToyTrack.

```bash
pip install toytrack
```

Optionally, there are Pytorch dataloaders available for convenience. These require the `torch` package.

```bash
pip install toytrack[torch]
```

# Usage

## Vanilla Event

```python
from toytrack import ParticleGun, Detector, EventGenerator

# Initialize a particle gun which samples uniformly from pt between 2 and 20 GeV, 
# initial direction phi between -pi and pi, and creation vertex vx and vy between -0.1 and 0.1 cm
# and which fires a normally-distributed number of particles with mean 20 and standard deviation 5
particle_gun = ParticleGun(num_particles=[20, 5, 'normal'], pt=[2, 20], pphi=[-np.pi, np.pi], vx=[-0.1, 0.1], vy=[-0.1, 0.1])

# Initialize a detector, a barrel-like detector with inner radius of 0.5 cm, and outer radius of 3 cm,
# with 10 layers
detector = Detector(dimension=2).add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

# Initialize an event generator and generate an event
event = EventGenerator(particle_gun, detector).generate_event()

# Access the particles, hits and tracks as needed
particles = event.particles
hits = event.hits
tracks = event.tracks

# Plot the event
event.display()
```

![Example Event](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/example_event_vanilla.png)

## Event with Track Holes

```python
... # as above

# Initialize a detector that randomly drops 10% of hits. If an int N is given, then exactly
# N hits per track are dropped
detector = Detector(dimension=2, hole_inefficiency=0.1).add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

... # as above

# Plot the event
event.display()
```

![Example Event with Holes](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/example_event_holes.png)

## Event with Noise Hits

```python
... # as above

# Generate event with between 30% and 70% noise hits per real hits. E.g. If the event has 100 
# real hits, then between 30 and 70 noise hits will be generated. If an int N is given, then
# the absolute value of N noise hits are generated
event = EventGenerator(particle_gun, detector, noise=[0.3, 0.7]).generate_event()

... # as above

# Plot the event
event.display()
```

![Example Event with Noise](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/example_event_noise.png)

## Event with Multiple Particle Guns

```python
# Initialize one particle gun which samples uniformly from pt between 2 and 3 GeV, 
# initial direction phi between -pi and pi, and creation vertex vx and vy between -0.1 and 0.1 cm
particle_gun_1 = ParticleGun(num_particles=[20, 5, 'normal'], pt=[2, 3], pphi=[-np.pi, np.pi], vx=[-0.1, 0.1], vy=[-0.1, 0.1])

# Initialize another particle gun which samples uniformly from pt between 100 and 200 GeV, 
# initial direction phi between -pi and pi, and creation vertex vx and vy between -0.1 and 0.1 cm
particle_gun_2 = ParticleGun(num_particles=1, pt=[100, 200], pphi=[-np.pi, np.pi], vx=[-0.1, 0.1], vy=[-0.1, 0.1])

... # as above

# Initialize an event generator with a list of particle guns
event = EventGenerator([particle_gun_1, particle_gun_2], detector).generate_event()

... # as above

# Plot the event
event.display()
```

![Example Event with Multiple Particle Guns](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/example_event_multigun.png)

## Detector with Layer Safety Guarantee

We can ensure that each particle has exactly one hit per layer by setting the `layer_safety_guarantee` flag to `True`.

Consider an event with low pT particles:

```python
particle_gun = ParticleGun(num_particles=[20, 5, 'normal'], pt=[1, 3], pphi=[-np.pi, np.pi], vx=[-0.1, 0.1], vy=[-0.1, 0.1])

detector = Detector(dimension=2, layer_safety_guarantee=False).add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

event = EventGenerator(particle_gun, detector).generate_event()

event.display()
```

![Example Event with Low pT Particles](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/example_event_lowpt.png)

```python
... # as above

# Initialize a detector WITH layer safety guarantee
detector = Detector(dimension=2, layer_safety_guarantee=True).add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

... # as above

event.display()
```

![Example Event with Layer Safety Guarantee](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/example_event_lowpt_layer_safety.png)

Observe that particles with such low pT that they "curled" before the final layer are now removed.

# Performance

ToyTrack is designed to be fast. The following benchmarks were performed on a 64-core AMD EPYC 7763 (Milan) CPU. 

![Scaling Study](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/time_scaling.png)

# Data Loading

## Pytorch Dataset

The `TracksDataset` class is a Pytorch dataset which can be used with a Pytorch dataloader.

```python
config = {
    "detector": {
        "dimension": 2,
        "hole_inefficiency": 0,
        "min_radius": 0.5,
        "max_radius": 3.,
        "number_of_layers": 10
    },
    "particle_guns": [
        {
            "num_particles": [20, 5, 'normal'],
            "pt": [2, 20],
            "pphi": [-3.14159, 3.14159],
            "vx": [-0.1, 0.1],
            "vy": [-0.1, 0.1]
        }
    ],
    "outputs": {
        "x": True,
        "mask": True,
        "pids": True,
        "event": True
    }
}

# initialize dataloader
dataset = TracksDataset(config)

# iterate over dataset
for batch in dataset:
    x, mask, pids, event = batch["x"], batch["mask"], batch["pids"], batch["event"]
    # Do something with the batch!
```

## Pytorch DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=100, collate_fn=collate_fn)

# iterate over dataloader
for batch in dataloader:
    x, mask, pids, event = batch["x"], batch["mask"], batch["pids"], batch["event"]
    # Do something with the batch (which now has an extra batch dimension of size 100)
```
