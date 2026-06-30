---
name: joon-robotics-competition-project
description: What the /home/user/joon project is — robotics competition perception on Jetson Orin Nano
metadata: 
  node_type: memory
  type: project
  originSessionId: c1b83ff7-86ff-4467-adfb-c2717d5ef135
---

`/home/user/joon` is a perception pipeline for a **robotics competition**: a robot in a 4×4 m arena autonomously finds, identifies, picks up, and places target objects, all computed onboard an **NVIDIA Jetson Orin Nano**.

- **Two separate Sets** (do NOT merge into one model): **Set 1** = white 3D-printed polyhedra (cube/octahedron/dodecahedron/icosahedron); **Set 2** = cubes with fruit textures. Only one target shape is announced on competition day; load only that set's models.
- **Set 1 architecture (built)**: two-stage = high-recall YOLO11n detector (single class `polyhedron`) → conservative MobileNetV3-Small shape classifier (5 classes incl. `unknown`) → IoU tracker → conservative decision policy (`SEARCHING`/`TARGET_CONFIRMED`/`PICKUP_READY`/`GIVE_UP`). Never pick on an ambiguous shape; dodecahedron-vs-icosahedron is the key confusion.
- **Sensors**: 2D LiDAR @0.30 m (nav only, not shape), two **NUROUM V11** RGB cameras (1280×720, 90° HFOV, fx=fy=640) @~0.20 m, −20° pitch, forward-outward ±30°. Synthetic data must be from these robot viewpoints.
- **Environment**: Isaac Sim 5.1 runs on **Windows** (not WSL); training/runtime in the WSL `yolo/` venv (ultralytics 8.4.78, torch 2.11 CUDA, RTX 4070 Ti Super). User: snurobotteam2@gmail.com.
- Source STLs: 6C1=cube, 8C1=octahedron, 12C1_Fixed=dodecahedron, 20C1=icosahedron.

Set 1 is implemented end-to-end (see `README_SET1.md`). Set 2 is a future separate pipeline reusing shared `sim/ deployment/ runtime/`. See [[isaac-sim-51-gotchas]].
