# Mega-Constellation Digital Twin: A 3D Starlink LISL Visualizer.

A 3D digital twin of Starlink satellites with Earth rotation, built with Skyfield + SGP4 for orbit propagation and Vispy for rendering. The project is also a data‑oriented digital twin: the core LISL pipeline is organized as batch numerical kernels over arrays to support large‑scale constellation analysis.

This repository is intended as a research‑grade digital twin and prototyping tool for inter‑satellite link (LISL) concepts and mega‑constellation behavior.

**Author:** Zhouyou Gu, research fellow at Singapore University of Technology and Design (SUTD), supervised by Prof. Jihong Park.

## Features
- Loads Starlink TLE data from Celestrak
- Propagates satellite positions/velocities in real time
- Visualizes satellites, Earth texture, and LISL links
- Computes candidate LISL edges using view constraints and a greedy matching heuristic
- Data-oriented digital twin pipeline with array-first kernels for scalability
- Uses Numba for JIT-compiled numerical kernels

## Quick Glossary
- TLE: Two-Line Element set used for satellite orbit propagation.
- LISL: Laser inter-satellite link.
- LCT: Laser communication terminal.

## Papers
Relevant papers by the author:

1) https://arxiv.org/abs/2601.21921
2) https://arxiv.org/abs/2601.21914

BibTeX:

```bibtex
@article{gu2026duality,
   title={Duality-Guided Graph Learning for Real-Time Joint Connectivity and Routing in LEO Mega-Constellations},
   author={Gu, Zhouyou and Choi, Jinho and Quek, Tony Q. S. and Park, Jihong},
   journal={arXiv preprint arXiv:2601.21921},
   year={2026}
}

@article{gu2026joint,
   title={Joint Laser Inter-Satellite Link Matching and Traffic Flow Routing in LEO Mega-Constellations via Lagrangian Duality},
   author={Gu, Zhouyou and Park, Jihong and Choi, Jinho},
   journal={arXiv preprint arXiv:2601.21914},
   year={2026}
}
```

## Data-Oriented Digital Twin
The digital twin is structured around data-parallel, array-first computations rather than per-satellite object updates. This makes it easier to scale to thousands of satellites while keeping the pipeline inspectable for research.

Key design points:
- Positions/velocities are propagated as dense arrays each frame.
- Directional vectors, view constraints, and candidate LISL edges are computed in bulk using Numba JIT-compiled kernels.
- Filtering and expansion steps operate on contiguous arrays to minimize Python overhead.
- Visualization consumes the resulting arrays directly (scatter positions, arrow segments, link segments).

Why data-oriented:
- Traditional constellation simulators often model each satellite/link as an object and step the simulation via events or time steps. This can create rigid data structures and extra overhead when updating large, time-varying constellations.
- A data-oriented approach keeps the constellation state in contiguous arrays, enabling vectorized updates and compiled kernels to operate directly on the same data without serialization.
- The visualization pipeline can directly map these arrays into GPU buffers, keeping the rendered digital twin state synchronized with the real-time constellation output.

## Repository Layout
- [simulation.py](simulation.py): Main digital twin entry point, numerical kernels, and visualization loop
- [requirements.txt](requirements.txt): Python dependencies
- population_density_texture.png: Earth texture image used for the sphere (place next to simulation.py)

## Requirements
- Python 3.9+ recommended
- GPU/OpenGL-capable environment for Vispy rendering

## Quick Start (Beginner)
If you are new to Python, follow these steps exactly.

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Ensure the Earth texture file exists next to [simulation.py](simulation.py):
   - population_density_texture.png

   If you do not have it, download or place it in the same folder as [simulation.py](simulation.py).

## Run

```bash
python simulation.py
```

You should see:
- A rotating Earth sphere with a texture
- Black satellite markers distributed around Earth
- Colored LISL links appearing between satellites

## Configuration
The digital twin uses a centralized configuration object. Default values live in `DigitalTwinConfig` and `DEFAULT_CONFIG` in [simulation.py](simulation.py).

To change parameters, edit `DEFAULT_CONFIG` in [simulation.py](simulation.py), or pass a custom `DigitalTwinConfig` when constructing `DigitalTwin` in `main()`.

Key parameters (what they mean):
- `for_theta_deg`: LISL pointing half‑angle threshold (smaller = stricter)
- `lisl_max_distance_km`: Maximum distance for LISL candidate edges
- `time_scale`: Digital twin time scaling factor
- `earth_radius_km`: Earth radius used for normalization
- `plot_potential_lisl`: Whether to draw all candidate LISLs (may be slower)
- `texture_path`: Path to Earth texture image

## Notes
- The digital twin loads TLEs from Celestrak at runtime.
- The visualization loop can be heavy on CPU and memory for large TLE sets.
- The first run may be slower due to Numba JIT compilation of kernels.
- Adjust parameters in DigitalTwinConfig for performance/visual clarity.

## Tested Platform
- macOS @ MacBook Pro (M4, 2024) with FPS ~15-25 for full Starlink constellation 

## Troubleshooting
- If you see a blank window, verify OpenGL support and that Vispy can access the GPU.
- If TLE loading is slow, check network connectivity or try a smaller constellation.
- If performance is slow, reduce `time_scale`, LISL distance, or switch off `plot_potential_lisl`.

## Common Beginner Questions
**Q: The window opens but nothing shows up.**
A: Wait 30–60 seconds on the first run. Numba compiles kernels and can stall rendering initially.

**Q: I got a “module not found” error.**
A: Make sure you activated the virtual environment and ran `pip install -r requirements.txt`.

**Q: It is too slow on my laptop.**
A: Reduce `time_scale`, increase `for_theta_deg`, or set `plot_potential_lisl = False` in [simulation.py](simulation.py).

## Citation
If you use this code in your work, please cite the paper listed above and acknowledge this repository.

## License
MIT License. See [LICENSE](LICENSE).
