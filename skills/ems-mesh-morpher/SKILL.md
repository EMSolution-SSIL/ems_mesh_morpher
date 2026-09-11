---
name: ems-mesh-morpher
description: Deform EMSolution and CAE meshes, propagate prescribed motion with IDW, RBF, or Weighted Laplace, generate conductor skin layers, and connect coordinate morphing to pyemsol DEFORM workflows.
---

# EMS Mesh Morpher

Use this skill when a user needs to move an existing finite-element mesh,
create static or dynamic eccentricity, propagate actuator displacement, add
volumetric conductor skin layers, or supply deformed coordinates to pyemsol.
Choose the workflow from the requested output; ordinary morphing preserves
topology, while skin-layer generation intentionally creates new cells.

## Prerequisites

Python 3.10 or newer is required. Install the package and its mesh-format
converter dependency from PyPI:

```powershell
python -m pip install ems-mesh-morpher
```

Before changing a mesh, identify its coordinate units, dimensionality,
Property IDs, supported element types, fixed boundaries, and output format.
Do not infer units or Property IDs from names alone.

## Choose a workflow

1. Use the `ems-mesh-morph` CLI for configured rigid motion and coordinate-only
   morphing of an existing mesh.
2. Use `generate_skin_layer` for tetrahedral or hexahedral conductor skin
   layers, sharp rectangular corners, and excluded symmetry planes.
3. Use `PyEMSolMorphingAdapter` when pyemsol supplies DEFORM regions, topology,
   and per-step rigid-node targets.
4. Use the lower-level IDW, RBF, or Weighted Laplace APIs when another solver
   supplies control-node displacements or destination coordinates directly.

## Configured mesh motion

Run a JSON configuration with:

```powershell
ems-mesh-morph INPUT_MESH OUTPUT_MESH --config CONFIG_JSON
```

Use mode `motor_eccentricity_2d` for the radial air-gap workflow and mode
`mesh_motion` for general motion. General motion supports translation,
rotation, scaling, and prescribed point displacements. Select `moving`,
`deformable`, `fixed`, `protected`, and optional `morphing_zone` regions by
Property ID, point index, or geometry. Available geometric selectors are
`all`, `box`, `circle`, `annulus`, `sphere`, and finite `cylinder`.

Region semantics are strict:

- `moving` receives the exact requested motion.
- `deformable` receives an interpolated displacement.
- `fixed` and `protected` stay at zero displacement.
- `morphing_zone` limits deformable nodes only.

Choose propagation according to the model:

- `none`: move only the selected moving region.
- `distance_blend`: inexpensive transition between moving and fixed controls.
- `idw`: lightweight scattered-control interpolation; useful for general
  actuator displacement and modest motion.
- `rbf`: smooth compact-support interpolation with local neighbors.
- `weighted_laplace` or `laplace`: connectivity-aware propagation; prefer it
  when element quality and repeated translation/rotation matter.

For EMSolution/pyemsol, set `output.file_format` to `gmsh4`. Inspect the printed
`result.summary()` and retain the exact configuration with generated results.

## Skin-layer generation

The current generator accepts linear tetrahedral and hexahedral conductor
meshes. It creates wedge cells from triangular boundary faces and hexahedral
cells from quadrilateral faces. Set total `thickness`, `layer_count`, and
`growth_ratio`; a ratio of `1.0` creates equal layers and a ratio greater than
`1.0` makes outer layers finer.

Use the reviewed samples to verify an installation:

```powershell
python .\examples\skin_layer_sample\generate_samples.py `
  --layer_count 3 `
  --growth_ratio 1.0 `
  --collision_policy fail
```

For one sample, `--sample NAME --thickness VALUE` overrides its total
thickness. Prefer the underscore option spellings shown above. Outputs and
`reports.json` are written under `generated/skin_layer_sample` unless
`--output_dir` is supplied.

In the Python API, define conductor `volume_property_ids`, output Property IDs,
excluded planes through `BoundaryRoleConfig`, and `CoreMorphingConfig(method="auto")`.
Keep collision policy `fail` unless the user deliberately chooses local
thickness reduction. Treat an inversion, collision, or quality-gate failure as
a request to inspect units, thickness, constraints, or the core morphing method;
do not bypass it silently.

## pyemsol DEFORM coupling

pyemsol coupling requires both the deformation description and deformation
mesh topology. A representative runner is:

```powershell
python .\examples\pyemsol_deform_sample\run_external_morphing.py `
  C:\path\to\model `
  --input_file input.json `
  --method weighted_laplace
```

Supported runner methods are `linear`, `idw`, `rbf`, and
`weighted_laplace`. Keep quality checks and zero-displacement anchor inference
enabled unless a diagnosed model condition requires otherwise. This interface
updates coordinates only; mesh replacement and local remeshing are not part of
the current release.

## Validation and interpretation

- Evaluate triangle, quad, tetrahedron, pyramid, wedge, and hexahedron quality
  where present. Reject orientation flips, degeneracy, and insufficient signed
  measure or scaled Jacobian.
- For repeated motion, evaluate every frame from the original reference mesh to
  prevent accumulated drift.
- Round-trip the result through its target format. For EMSolution, verify Gmsh
  4.x output and retained Physical/Entity metadata.
- Record the input path, output path, exact command or config, selected Property
  IDs, units, quality summary, and any locally reduced skin thickness.

Current skin-layer limitations include all-prism input, higher-order elements,
concave-corner repair, full offset-surface intersection repair, local
remeshing, explicit per-layer thickness lists, and automatic skin-depth-based
thickness. Use only the public `data/skin_layer_sample` models in reproducible
examples; never publish private solver or customer meshes.
