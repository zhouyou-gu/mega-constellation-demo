# Mega-Constellation Simulation

A 3D visualization of Starlink satellites with Earth rotation, built with Skyfield + SGP4 for orbit propagation and Vispy for rendering.

## Features
- Loads Starlink TLE data from Celestrak
- Propagates satellite positions/velocities in real time
- Visualizes satellites, Earth texture, and LISL links

## Requirements
- Python 3.9+ recommended
- GPU/OpenGL-capable environment for Vispy rendering

## Setup
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure the Earth texture file exists next to `simulation.py`:
   - `population_density_texture.png`

## Run
- `python simulation.py`

## Notes
- The simulation loads TLEs from Celestrak at runtime.
- Increase/decrease `Simulation.TIME_SCALE` to adjust simulation speed.
- LISL distance/angle thresholds are configured in `Simulation` class constants.
