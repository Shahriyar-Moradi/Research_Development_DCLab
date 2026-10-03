# Autonomous driving and road-map graphs

**Status: proposed, safety-critical research.** A road network graph describes static topology; a driving-scene graph describes actors, traffic controls, and their interactions over time. Combining these may support prediction and explanations, but a benchmark prototype is not evidence of vehicle safety.

## Candidate graph

- Map nodes: lane segments, intersections, crosswalks, merge points.
- Map edges: legal connectivity, successor/predecessor, adjacency, merge relations.
- Actor nodes: vehicles, pedestrians, cyclists, with position, velocity, class, and uncertainty over time.
- Actor-map edges: occupies lane, approaches crossing, yields to, conflicts with.
- Temporal links: actor tracks and predicted state transitions.

## First bounded study

Use a labeled public driving dataset and one scenario, such as pedestrian crossing at an intersection. Compare map-only, actors-only, lane/actor graph, and graph-plus-temporal model for trajectory or event prediction. Split by scene/log/road location to reduce leakage. Report miss rate, displacement at fixed horizons, collision-proxy events, calibration, rare-scenario coverage, and runtime. Preserve raw scene references so graph errors can be audited.

Do not claim autonomous driving readiness. Real deployment requires a defined operational domain, extensive scenario and fault analysis, independent safety engineering, and compliance work beyond this repository's current scope.
