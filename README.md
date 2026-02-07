# Mega-Constellation Simulation

A 3D visualization of Starlink satellites with Earth rotation, built with Skyfield + SGP4 for orbit propagation and Vispy for rendering.

This repository is intended as a research-grade visualization and prototyping tool for inter-satellite link (LISL) concepts and mega‑constellation behavior.

**Author:** Zhouyou Gu, research fellow at Singapore University of Technology and Design (SUTD), supervised by Prof. Jihong Park.

## Features
- Loads Starlink TLE data from Celestrak
- Propagates satellite positions/velocities in real time
- Visualizes satellites, Earth texture, and LISL links
- Computes candidate LISL edges using view constraints and a greedy matching heuristic

## Papers
Relevant papers by the author:

1) https://arxiv.org/abs/2601.21921
2) https://arxiv.org/abs/2601.21914

BibTeX:

```bibtex
@article{gu2026dualityguided,
   title={Duality-Guided Graph Learning for Real-Time Joint Connectivity and Routing in LEO Mega-Constellations},
   author={Gu, Zhouyou and Choi, Jinho and Quek, Tony Q. S. and Park, Jihong},
   journal={arXiv preprint arXiv:2601.21921},
   year={2026}
}

@article{gu2026jointlaser,
   title={Joint Laser Inter-Satellite Link Matching and Traffic Flow Routing in LEO Mega-Constellations via Lagrangian Duality},
   author={Gu, Zhouyou and Park, Jihong and Choi, Jinho},
   journal={arXiv preprint arXiv:2601.21914},
   year={2026}
}
```

## Repository Layout
- [simulation.py](simulation.py): Main simulation entry point, numerical kernels, and visualization loop
- [requirements.txt](requirements.txt): Python dependencies
- population_density_texture.png: Earth texture image used for the sphere (place next to simulation.py)

## Requirements
- Python 3.9+ recommended
- GPU/OpenGL-capable environment for Vispy rendering

## Setup
1) Install dependencies:
   - pip install -r [requirements.txt](requirements.txt)
2) Ensure the Earth texture file exists next to [simulation.py](simulation.py):
   - population_density_texture.png

## Run
- /path/to/python [simulation.py](simulation.py)

## Configuration
The simulation uses a centralized configuration object. Default values live in SimulationConfig.

Key parameters:
- for_theta_deg: LISL pointing half-angle threshold
- lisl_max_distance_km: Maximum distance for LISL candidate edges
- time_scale: Simulation time scaling factor
- earth_radius_km: Earth radius used for normalization
- plot_potential_lisl: Whether to draw all candidate LISLs
- texture_path: Path to Earth texture image

## Notes
- The simulation loads TLEs from Celestrak at runtime.
- The visualization loop can be heavy on CPU and memory for large TLE sets.
- Adjust parameters in SimulationConfig for performance/visual clarity.

## Troubleshooting
- If you see a blank window, verify OpenGL support and that Vispy can access the GPU.
- If TLE loading is slow, check network connectivity or try a smaller constellation.
- If performance is slow, reduce time_scale, LISL distance, or switch off plot_potential_lisl.

## Citation
If you use this code in your work, please cite the paper listed above and acknowledge this repository.

## License
MIT License. See [LICENSE](LICENSE).
