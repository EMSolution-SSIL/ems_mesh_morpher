# EMS Mesh Morpher agent guidance

## Scope

This repository provides finite-element mesh motion, displacement propagation,
skin-layer generation, and pyemsol external DEFORM coupling. Keep package
changes under `src/ems_mesh_morpher`, examples under `examples`, and regression
coverage under `tests`.

## Source and dependency boundaries

- `ems-file-format-converter` is a separate project and is required at version
  0.6.0 or newer. Use its structured mesh I/O instead of copying parsers here.
- Preserve format metadata, Property IDs, node IDs, and supported cell data
  when reading and writing. Use Gmsh 4.x for EMSolution/pyemsol output.
- The public release targets linear elements. Do not claim higher-order element
  support unless it is implemented and tested.
- This project is source-available under PolyForm Perimeter 1.0.1. Do not
  describe it as OSI-approved open-source software.

## Mesh-motion invariants

- Ordinary morphing changes coordinates only. Preserve point order, element
  connectivity, cell ordering, and Property assignments.
- Moving nodes must receive the prescribed rigid or pointwise displacement.
  Fixed and protected nodes must remain unchanged; protected selection has the
  highest assignment priority. Propagate displacement only to deformable nodes.
- Reject ambiguous control definitions, incomplete pyemsol DEFORM payloads,
  topology/order mismatches, and unsupported cells instead of guessing.
- Repeated-motion runs must evaluate each frame from the reference mesh rather
  than the previous frame so coordinate drift does not accumulate.
- Keep quality checks enabled by default. Reject orientation flips, degenerate
  cells, inverted cells, or results below configured measure/Jacobian limits.
  Do not silently write a failed result.

## Skin-layer invariants

- The 2-D generator accepts 2D-only planar linear triangle/quad meshes and
  creates quad layers on their exterior edges. Reject any input containing 3-D
  volume cells because their interfaces would not receive matching layers.
- For 2-D symmetry/model cuts, use `SkinLayer2DConfig.excluded_planes` with
  `PlaneSelector`. Keep the default `slip_plane` constraint unless the boundary
  is intentionally fixed; active layers must terminate conformingly on the cut.
- The 3-D generator accepts linear tetrahedral or hexahedral conductor input.
  Triangle boundary faces create wedge layers and quadrilateral faces create
  hexahedral layers; all-prism input is reference data, not supported generator
  input.
- Skin-layer generation intentionally adds points, cells, and Property data.
  Preserve the remaining source topology and metadata.
- Preserve excluded symmetry/boundary planes. Do not create skin volume on an
  excluded face, and keep its constrained points on the specified plane. In a
  3-D mixed-region model, exclude only a true outer or symmetry boundary; every
  interface adjoining a retained volume region must be layered to stay conforming.
- Keep collision policy `fail` by default. `reduce` may lower local thickness;
  `allow` may record a risk but must still pass the normal 3-D quality gate.
- Concave-corner repair, full offset-surface intersection repair, local
  remeshing, explicit per-layer thickness lists, and skin-depth-derived
  thickness are not implemented in the current release.
- The current 2-D generator supports plane-selected excluded edges. Explicit
  edge IDs, boundary Property selectors, and local 2-D remeshing are not yet
  implemented.

## Data policy

Only the reviewed samples under `data/skin_layer_sample` may be committed.
GL80 meshes, pyemsol DEFORM models, solver inputs/results, and customer or
internal meshes must stay local. Generated meshes and reports belong under
ignored `generated` or temporary directories.

## Validation

Run the relevant tests after changes; for a normal package change run:

```powershell
python -m pytest -q
python -m build
python -m twine check dist/*
```

For mesh I/O or skin-layer changes, also round-trip representative public
samples through the affected formats and assert that the 2-D/3-D quality gate
reports no inverted cells. Add regression tests for new selection, propagation,
constraint, collision, or topology rules.

Do not publish to PyPI, push to GitHub, change licenses, or add model data unless
the user explicitly requests that external action.
