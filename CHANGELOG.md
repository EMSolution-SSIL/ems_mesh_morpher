# Changelog

## 0.13.0 - 2026-09-24

- Add planar triangle/quad skin-layer generation with multiple layers and
  Weighted Laplace core morphing.
- Reject mixed 2D/3D inputs that would create nonconforming interfaces.
- Add line/plane-selected symmetry-boundary exclusions for the 2D generator.
- Constrain active/excluded junctions and excluded-boundary nodes to their
  symmetry planes.
- Add thin-plate and toothed stator validation cases and quality regressions.

## 0.12.0 - 2026-09-09

- Add reference-based repeated translation and rotation schedules.
- Reuse prepared Weighted Laplace sparse factorization across frames.
- Add element-size and aspect-ratio-aware Laplace weighting.
- Add common relative quality reports for triangle, quad, tetrahedron, pyramid,
  wedge, and hexahedron cells.
- Add quality-driven displacement limiting.
- Add repeat-motion benchmarks for block translation and annular rotation.

## 0.10.0 - 2026-09-08

- Add pyemsol external DEFORM schema v1/v2 adapter.
- Add IDW, RBF, and Weighted Laplace coupling backends.
- Add explicit rigid XYZ translation and fixed-boundary support.

## 0.9.0 - 2026-09-06

- Add multi-layer tetrahedral and hexahedral Skin Layer Generator.
- Add local thickness limits, collision policies, and cross-format output.
