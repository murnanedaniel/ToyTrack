# ToyTrack

[![Documentation Status](https://readthedocs.org/projects/toytrack/badge/?version=latest)](https://toytrack.readthedocs.io/en/latest/?badge=latest)

ToyTrack is a Python library for generating toy tracking events for particle physics. 

**The goal**: To produce a "*good-enough*" event simulation, in as few lines as possible (currently 3 lines), as quickly as possible (currently 0.07 seconds for a 10,000-particle event).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ToyTrack.

```bash
pip install toytrack
```

# Usage

## Vanilla Event

```python
from toytrack import ParticleGun, Detector, EventGenerator

# Initialize a particle gun which samples uniformly from pt between 10 and 20 GeV, 
# initial direction phi between -pi and pi, and creation vertex vx and vy between -0.1 and 0.1 cm
particle_gun = ParticleGun(dimension=2, pt=[2, 20], pphi=[-np.pi, np.pi], vx=[-0.1, 0.1], vy=[-0.1, 0.1])

# Initialize a detector
detector = Detector(dimension=2).add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

# Initialize an event generator, which generates an event with a number of particles given by a normal
# distribution, with mean 10 and standard deviation 5
event = EventGenerator(particle_gun, detector, num_particles=[20, 5, 'normal']).generate_event()

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
event = EventGenerator(particle_gun, detector, num_particles=[20, 5, 'normal'], noise=[0.3, 0.7]).generate_event()

... # as above

# Plot the event
event.display()
```

![Example Event with Noise](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/example_event_noise.png)

## Performance

ToyTrack is designed to be fast. The following benchmarks were performed on a 64-core AMD EPYC 7763 (Milan) CPU. 

![Scaling Study](https://raw.githubusercontent.com/murnanedaniel/ToyTrack/main/docs/imgs/time_scaling.png)

