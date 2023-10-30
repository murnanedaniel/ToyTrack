# ToyTrack

ToyTrack is a Python library for generating toy tracking events for particle physics. It provides classes for particles, hits, detectors, and events, and allows for the generation of events with a specified number of particles.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ToyTrack.

```bash
pip install toytrack
```

## Usage
```python
from toytrack import ParticleGun, Detector, EventGenerator
Initialize a particle gun
particle_gun = ParticleGun(dimension=2, pt=(0.1, 1.0, 'uniform'), pphi=(0, 2np.pi, 'uniform'), vx=0, vy=0)

# Initialize a detector
detector = Detector(dimension=2)
detector.add_from_template('barrel', min_radius=10, max_radius=100, number_of_layers=10)

# Initialize an event generator
event_generator = EventGenerator(particle_gun, detector, num_particles=(1, 10, 'uniform'))

# Generate an event
event = event_generator.generate_event()

# Access the particles and hits
particles = event.particles
hits = event.hits
``````