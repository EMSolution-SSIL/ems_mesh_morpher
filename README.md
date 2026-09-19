# ems_mesh_morpher

`ems_mesh_morpher` provides reusable mesh motion and local morphing-zone tools
for EMSolution workflows. It supports translation, rotation, scaling,
prescribed displacement, property/geometrical selection, radial, distance, IDW,
compact-support RBF, or weighted Laplace propagation, basic 2D quality reporting, and
metadata-preserving output through `ems_file_format_converter`, including Gmsh
4.x. The Phase 1 motor eccentricity interface remains available as a
backward-compatible convenience wrapper.

This project is source-available under the
[PolyForm Perimeter License 1.0.1](LICENSE). It permits broad use, modification,
and redistribution, but does not permit providing a product that competes with
this software. It is therefore not an OSI-approved open-source license.

## Setup

Install the published package from PyPI:

```powershell
python -m pip install ems-mesh-morpher
```

For editable development installation, use this repository:

```powershell
cd C:\EMSolution\build\MeshMorphingWorkspace\ems_mesh_morpher
conda run -n ems-py311 python -m pip install -e .
```

The command line entry point is `ems-mesh-morph`.

```powershell
conda run -n ems-py311 ems-mesh-morph --help
```

## Basic Usage

```powershell
conda run -n ems-py311 ems-mesh-morph INPUT_MESH OUTPUT_MESH --config CONFIG_JSON
```

Two configuration modes are supported:

- `motor_eccentricity_2d`: the existing Phase 1 radial air-gap workflow.
- `mesh_motion`: the general Phase 2/3 motion and morphing-zone workflow.

The Python API is also available:

```python
from ems_mesh_morpher import apply_mesh_motion, load_config

config = load_config("examples/general/local_zone_translation.json")
result = apply_mesh_motion(mesh, config)
print(result.summary())
```

For EMSolution / pyemsol, keep the output format as Gmsh 4.x. The GL80 example
configs already set this with:

```json
{
  "output": {
    "file_format": "gmsh4"
  }
}
```

## Config Structure

The motor-eccentricity example configs are under `examples/GL80`. Motor mesh
and solver input files are not distributed in this public repository; provide
your own model and adjust property IDs to match it.

- `static_stator_eccentricity_*.json`
  - Static eccentricity case.
  - The stator mesh is moved/deformed, and the rotor mesh remains fixed.
- `dynamic_rotor_eccentricity_*.json`
  - Dynamic eccentricity case.
  - The rotor mesh is moved/deformed, and the stator mesh remains fixed.

The eccentricity amount is controlled by `motion.displacement`.

```json
"motion": {
  "type": "translation",
  "displacement": [0.00005, 0.0]
}
```

The GL80 meshes use meters, so:

- 0.01 mm = `0.00001`
- 0.05 mm = `0.00005`
- 0.1 mm = `0.0001`

## General Motion and Morphing Zones

General examples are under `examples/general`.

Supported motion types:

- `translation`: `displacement` with two or three components.
- `rotation`: `center`, `axis`, and `angle_degrees`.
- `scaling`: `center` and a scalar or two/three-axis `scale`.
- `prescribed`: per-point entries in `point_displacements`.

Each region accepts `property_ids`, `point_indices`, and one `geometry` or a
list of `geometries`. Criteria can be combined with `union` or `intersection`.
Available geometries are `all`, `box`, `circle`, `annulus`, `sphere`, and a
finite `cylinder`.

```json
{
  "mode": "mesh_motion",
  "regions": {
    "moving": {"property_ids": [10]},
    "deformable": {"property_ids": [20]},
    "fixed": {"property_ids": [30]},
    "protected": {"property_ids": [40]},
    "morphing_zone": {
      "geometry": {
        "type": "box",
        "minimum": [-0.01, -0.01],
        "maximum": [0.01, 0.01]
      }
    }
  },
  "motion": {
    "type": "translation",
    "displacement": [0.001, 0.0]
  },
  "morphing": {
    "type": "distance_blend",
    "blend": "smoothstep"
  }
}
```

`morphing_zone` limits deformable nodes only. Moving nodes retain their
prescribed motion, while fixed and protected nodes remain unchanged. A
`distance_blend` requires at least one moving node and one fixed/protected node.
Protected nodes have the highest assignment priority.

`morphing.type` can also be set to `"idw"`. Moving nodes are non-zero control
points, fixed/protected nodes are zero-displacement controls, and IDW is evaluated
only for deformable nodes. The default uses the nearest 32 controls and a power
of `2.0`.

```json
"morphing": {
  "type": "idw",
  "power": 2.0,
  "neighbors": 32,
  "radius": null,
  "smoothing": 0.0,
  "outside_policy": "nearest"
}
```

For direct integration with actuator results, the generic API accepts either
control-node displacements or destination coordinates:

```python
import numpy as np

from ems_file_format_converter import write_mesh
from ems_mesh_morpher import IDWConfig, build_idw_displacement_field

field = build_idw_displacement_field(
    mesh.points,
    control_indices=[10, 20],
    control_target_points=np.array([
        [0.001, 0.000, 0.000],
        [0.010, 0.002, 0.000],
    ]),
    fixed_indices=[30, 31],
    config=IDWConfig(power=2.0, neighbors=32),
)
morphed_mesh = field.apply_to_mesh(mesh)
write_mesh("actuator_deformed.msh", morphed_mesh, file_format="gmsh4")
```

`field.displacement` and `field.target_points` are full arrays in original node
order. Applying the field preserves connectivity, property/material data, and
node/element IDs. The same mesh can therefore be written through
`ems_file_format_converter` in a format supported by the target EMSolution
workflow. `apply_mesh_motion` continues to accept a full external displacement
array as well. See `examples/general/idw_prescribed_displacement.json` for the
JSON/CLI form.

The same generic contract is available with a sparse compact-support RBF. It uses
the Wendland C2 kernel and an affine polynomial term by default. The affine term
improves reproduction of translation and approximately linear displacement fields.

```python
from ems_mesh_morpher import RBFConfig, build_rbf_displacement_field

field = build_rbf_displacement_field(
    mesh.points,
    control_indices=[10, 20],
    control_displacements=np.array([
        [0.001, 0.000, 0.000],
        [0.010, 0.002, 0.000],
    ]),
    fixed_indices=[30, 31],
    config=RBFConfig(
        radius=None,
        neighbors=32,
        radius_scale=1.25,
        regularization=1.0e-12,
        polynomial_degree=1,
    ),
)
```

With `radius=None`, a support radius is estimated from the control and query point
spacing. For large models, setting `radius` explicitly provides more predictable
sparsity and run time. `polynomial_degree=-1` disables the polynomial term, `0`
uses a constant term, and `1` uses the default affine term. See
`examples/general/rbf_prescribed_displacement.json` for JSON/CLI input.

Weighted Laplace uses the actual mesh connectivity rather than only spatial
distance. It supports line, triangle, quadrilateral, tetrahedral, hexahedral,
wedge, and pyramid cells. When volume and boundary cells coexist, repeated edges
are deduplicated so boundary faces do not bias the weights while independent
surface regions remain available.

```python
from ems_mesh_morpher import LaplaceConfig, build_laplace_displacement_field

field = build_laplace_displacement_field(
    mesh,
    control_indices=[10, 20],
    control_displacements=np.array([
        [0.001, 0.000, 0.000],
        [0.010, 0.002, 0.000],
    ]),
    fixed_indices=[30, 31],
    query_indices=deformable_node_indices,
    config=LaplaceConfig(
        weighting="inverse_distance",
        distance_power=1.0,
    ),
)
```

Use `morphing.type = "weighted_laplace"` (or the shorter alias `"laplace"`) in
JSON/CLI input. `weighting="uniform"` is available for comparison. Every
connected deformable component must contain a moving or fixed control boundary;
unanchored components are reported as an error. See
`examples/general/weighted_laplace_prescribed_displacement.json`.

For meshes containing small or high-aspect-ratio elements,
`weighting="inverse_distance_element"` increases graph stiffness using incident
element size and edge aspect ratio. `element_size_power` and
`element_aspect_power` control the two contributions. The prepared API builds and
factorizes the sparse matrix once and accepts new control values for every frame.

```python
from ems_mesh_morpher import LaplaceConfig, prepare_laplace_interpolator

prepared = prepare_laplace_interpolator(
    mesh,
    query_indices=deformable_node_indices,
    control_indices=control_node_indices,
    config=LaplaceConfig(weighting="inverse_distance_element"),
)
displacement = prepared.solve(control_displacements)
```

## Repeated motion and quality control

`RepeatedMotionRunner` always evaluates every frame from the unchanged reference
mesh. This prevents accumulated coordinate drift. Weighted Laplace matrix
factorization is reused when the region definition and mesh topology stay fixed.

```python
from ems_mesh_morpher import MotionSchedule, RepeatedMotionRunner

schedule = MotionSchedule.translations(
    times=[0.0, 0.5, 1.0],
    displacements=[[0.0, 0.0], [0.001, 0.0], [0.0, 0.0]],
)
run = RepeatedMotionRunner(reference_mesh, config).run(schedule)
print(run.summary())
```

The common quality API evaluates triangle, quad, tetrahedron, pyramid, wedge,
and hexahedron cells. It reports orientation flips, degeneracy, minimum signed
measure ratio, minimum scaled Jacobian, and the worst element. A mesh-motion
configuration can reject an invalid result or reduce the requested displacement
with a bounded line search.

```json
{
  "quality": {
    "enabled": true,
    "minimum_measure_ratio": 0.05,
    "minimum_scaled_jacobian": 0.01,
    "failure_policy": "scale",
    "minimum_scale": 0.1
  }
}
```

Run the translation/rotation benchmark for IDW, RBF, and element-aware Weighted
Laplace with:

```powershell
python .\examples\repeated_motion\benchmark.py --model all --frames 21 `
  --output .\generated\repeated_motion_benchmark.json
```

## pyemsol External DEFORM Coupling

Version 0.12 can consume the dictionaries returned by pyemsol's
`describe_deformation()` and `describe_deformation_mesh()`. The static topology is
validated once and each call to `evaluate()` returns the absolute-coordinate payload
accepted by `CoupledSession.step(deformation=...)`. pyemsol is not a required package
dependency because the adapter itself only exchanges Python dictionaries.

```python
from ems_mesh_morpher import PyEMSolMorphingAdapter

description = session.describe_deformation()
topology = session.describe_deformation_mesh()
adapter = PyEMSolMorphingAdapter.prepare(
    description,
    topology=topology,
    method="weighted_laplace",  # linear, idw, rbf, weighted_laplace
)

for step_no, (time, delta_time, position) in enumerate(steps, start=1):
    deformation = adapter.evaluate(position, mesh_revision=step_no)
    result = session.step(
        time=time,
        delta_time=delta_time,
        deformation=deformation,
    )
```

The adapter maps every external node ID to a mesh-local index and keeps the output in
`return_node_ids` order. Rigid nodes are prescribed controls, deformable nodes are
interpolated, and fixed shared-boundary nodes remain at their reference coordinates.
Some existing models expose fixed outer-boundary nodes as `deformable` in schema v2.
By default, nodes whose schema v1 endpoint displacement is zero are inferred as fixed
anchors. This can be disabled with
`PyEMSolConstraintConfig(infer_zero_displacement_anchors=False)`.

For rotations and other non-linearly parameterized rigid motion, pass explicit global
XYZ targets keyed by external node ID. Every rigid node in the region must be present.
Independent XYZ translation can be passed more simply with `rigid_translations`.

```python
deformation = adapter.evaluate(
    rigid_targets={
        1: {
            1001: [x1, y1, z1],
            1002: [x2, y2, z2],
        }
    },
    mesh_revision=step_no,
)

deformation = adapter.evaluate(
    rigid_translations={1: [x_displacement, y_displacement, z_displacement]},
    mesh_revision=step_no,
)
```

Before returning a payload, the adapter checks relative signed area or volume for
triangle, quad, tetra, prism/wedge, hexahedron, and pyramid elements. Orientation
flips, degenerate elements, and an insufficient measure ratio are rejected before
EMSolution advances the step.

Run the adapter example with a pyemsol model directory that contains the
solver input and mesh files:

```powershell
python .\examples\pyemsol_deform_sample\run_external_morphing.py `
  C:\path\to\model --input_file input.json --method weighted_laplace
```

The interface updates coordinates only. Mesh replacement and local remeshing are not
part of this version.

## Phase 10C-10E layered generator

Version 0.9 creates one or more volumetric skin layers on tetrahedral or hexahedral
conductors. Triangle boundary faces produce wedge cells and quadrilateral faces
produce hexahedral cells. Rectangular sharp corners and excluded symmetry planes
are supported. Phase 10E adds opposing-surface collision checks and explicit
local thickness control.

```python
from ems_file_format_converter import read_mesh
from ems_mesh_morpher.skin_layer import (
    BoundaryRoleConfig,
    CoreMorphingConfig,
    PlaneConstraint,
    PlaneSelector,
    SkinLayerCollisionConfig,
    SkinLayerConfig,
    SkinLayerQualityConfig,
    generate_skin_layer,
)

mesh = read_mesh("conductor.neu")
symmetry = PlaneSelector(
    PlaneConstraint(normal=(1.0, 0.0, 0.0), offset=0.0, tolerance=1.0e-9)
)
config = SkinLayerConfig(
    thickness=0.0001,
    volume_property_ids=(100,),
    layer_count=3,
    growth_ratio=1.0,
    boundary_roles=BoundaryRoleConfig(excluded_planes=(symmetry,)),
    core_morphing=CoreMorphingConfig(method="auto"),
    collision=SkinLayerCollisionConfig(policy="fail"),
    quality=SkinLayerQualityConfig(
        minimum_core_scaled_jacobian=1.0e-8,
        minimum_layer_scaled_jacobian=1.0e-8,
        minimum_core_measure_ratio=0.1,
    ),
    skin_layer_property_id=200,
    inner_surface_property_id=201,
)
result = generate_skin_layer(mesh, config)
result.write("conductor_with_skin.neu")
print(result.report.to_dict())
```

`thickness` is the total skin thickness. `layer_count` controls the number of
layers and `growth_ratio` is the adjacent-layer thickness ratio from the outer
surface toward the core. A ratio of `1.0` gives equal layers. A ratio greater
than `1.0` makes the outer layers finer; for example, three layers with a ratio
of `2.0` divide the total thickness in proportions `1:2:4`.

`SkinLayerCollisionConfig.policy` controls Phase 10E behavior:

- `"fail"` is the default and stops before generation when the requested
  thickness exceeds a local edge-length or opposing-surface limit.
- `"reduce"` explicitly permits node-local reduction. The default opposing
  surface factor is `0.4`, leaving at least 20% of a narrow section between
  layers generated from both sides.
- `"allow"` records the risk without changing the thickness; the normal 3D
  quality gate still rejects inverted or excessively degraded elements.

The node-wise requested, maximum, and applied values are available in
`result.local_thickness`. The summary report includes the minimum, maximum,
and mean applied thickness, reduced-node count, opposing-surface risk count,
minimum detected clearance, and a warning whenever reduction is used. Local
reductions are smoothed along surface edges using a bounded thickness gradient.

An existing symmetry surface property can be excluded without a plane equation:

```python
config = SkinLayerConfig(
    thickness=0.0001,
    volume_property_ids=(100,),
    boundary_roles=BoundaryRoleConfig(excluded_surface_property_ids=(210,)),
)
result = generate_skin_layer(mesh, config)
```

The output format is selected by the extension or can be passed explicitly.

```python
result.write("conductor_with_skin.atl")
result.write("conductor_with_skin.unv")
result.write("conductor_with_skin.neu")
result.write("conductor_with_skin.msh", file_format="gmsh4")
```

`CoreMorphingConfig.method` accepts `"auto"`, `"distance_blend"`, `"idw"`,
`"rbf"`, `"laplace"`, or `"none"`. `"auto"` first tries distance blending and retries with the sparse
Laplace solver when the 3D quality gate rejects the result. The report records the
selected method, requested and actual normal thickness, element counts, minimum
core volume ratio, scaled Jacobians, and warnings from rejected attempts.
IDW and RBF are currently selected explicitly so that introducing them does not
change existing `"auto"` results before comparative quality and performance
evaluation.
The Laplace backend uses `CoreMorphingConfig(laplace=LaplaceConfig(...))` and
shares the same weighted solver as general mesh motion.

The lower-level Phase 10B APIs remain available when only surface extraction,
boundary classification, or offset targets are needed:

```python
from ems_mesh_morpher.skin_layer import (
    classify_boundary_faces,
    compute_inward_offset,
    detect_surface_features,
    extract_boundary_surface,
)

surface = extract_boundary_surface(mesh, volume_property_ids=[100])
roles = classify_boundary_faces(
    mesh,
    surface,
    BoundaryRoleConfig(excluded_planes=(symmetry,)),
)
features = detect_surface_features(mesh, surface, feature_angle_degrees=30.0)
offset = compute_inward_offset(
    mesh,
    surface,
    thickness=0.0001,
    classification=roles,
    features=features,
)
```

Current limitations: linear tetrahedron/hexahedron input only; explicit
per-layer thickness lists and skin-depth-based thickness calculation are not yet supported;
concave corners fail explicitly by default. Phase 10E uses local edge-length and
inward-ray/opposing-surface checks; full offset-surface intersection repair and
local remeshing are not yet supported.

## Skin-layer sample data

The supplied Femap and ATLAS samples can be generated together from the project
root:

```powershell
conda run -n ems-py311 python .\examples\skin_layer_sample\generate_samples.py --layer_count 3 --growth_ratio 1.0 --collision_policy fail
```

Outputs and `reports.json` are written to `generated/skin_layer_sample`. The
sample configuration treats the domain boundary planes `x=0`, `y=0`, and `z=0`
as excluded surfaces, so no skin volume is created on those planes.

`--thickness` overrides the total thickness for one selected sample. For
example, this generates three equal layers with a total thickness of `0.012`:

```powershell
conda run -n ems-py311 python .\examples\skin_layer_sample\generate_samples.py --sample block_hexa --thickness 0.012 --layer_count 3 --growth_ratio 1.0 --output_dir .\generated\skin_layer_custom
```

Available sample names are `block_hexa`, `block_tetra`,
`atlas_block_conductor`, and `atlas_curved_coil`. To prevent an accidental
cross-model override, `--thickness` requires exactly one `--sample`. Without
`--thickness`, each selected sample uses the total thickness listed below.
The former hyphenated spellings remain accepted for compatibility, but the
underscore spellings are the documented form.

| Sample | Target property | Total thickness | Three-layer output |
|---|---:|---:|---|
| `post_geom_hexa.neu` | 1 | 0.01 | 375 hexahedra |
| `post_geom_tetra.neu` | 1 | 0.01 | 750 wedges |
| `post_geom.atl` block conductor | 1 | 0.0025 | 375 hexahedra |
| `post_geom.atl` curved coil | 3 | 0.002 | 585 hexahedra |

All four cases pass the 3D quality gate and round-trip through their original
format. The curved coil's actual face-normal thickness is 0.0019754 to 0.002;
the small variation comes from smooth normal averaging along the arc. The
all-prism reference mesh is read correctly but is intentionally rejected by the
current tetrahedron/hexahedron-only generator.

## 日本語版

日本語版は [`README_ja.md`](README_ja.md) を参照してください。
