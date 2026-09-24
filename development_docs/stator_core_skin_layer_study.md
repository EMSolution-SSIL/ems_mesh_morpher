# Stator Core 2D Skin-Layer Study

## Scope

This study evaluates the current `generate_skin_layer_2d` implementation on:

- input: `data/skin_layer_sample/Dmodel/pre_geom2D.neu`
- conductor region: Property 1
- output layers: 3 equal-thickness quadrilateral layers, Property 101
- core propagation: Weighted Laplace with element-aware inverse-distance weights

The source contains 1,344 points and 2,530 triangular elements. Property 1
contains 1,156 triangles. Its boundary is one closed, non-branched loop with
326 edges; the minimum boundary edge length is approximately `0.444 mm` when
the source coordinates are interpreted as metres.

## Important algorithm detail

The current 2D generator does not contract points toward a region centroid.
It computes an inward offset from the two oriented boundary-edge normals at
each boundary node, applies a bounded corner miter, and propagates the boundary
displacement through the connected core with Weighted Laplace. Outer arcs,
slot arcs, tooth sides, and tooth tips can therefore be handled as parts of the
same oriented boundary without identifying a separate centroid for each tooth.

## Results

All tests used `layer_count=3`, `growth_ratio=1.0`, excluded the `x=0` and
`y=0` model cuts, and included an additional non-adjacent boundary-segment
intersection check for every generated contour. Of 326 boundary edges, 38 were
excluded and 288 received skin layers.

| Total thickness | Result | Minimum core area ratio | Minimum layer scaled Jacobian |
|---:|---|---:|---:|
| 0.10 mm | Passed | 0.6108 | 0.6876 |
| 0.20 mm | Passed | 0.3170 | 0.6876 |
| 0.25 mm | Passed | 0.1914 | 0.6876 |
| 0.30 mm | Passed | 0.1086 | 0.6876 |
| 0.35 mm | Rejected | - | - |
| 0.40 mm | Rejected | - | - |
| 0.50 mm | Rejected | - | - |
| 1.00 mm | Rejected | - | - |
| 2.00 mm | Rejected | - | - |

The successful cases had no detected contour intersections, inverted layer
elements, or flipped core elements. At `0.35 mm` and above, the existing core
quality gate rejected flipped/degenerate triangles and an insufficient area
ratio. The practical margin should remain below the last passing value; for
this mesh, `0.20 mm` is substantially healthier than `0.30 mm`.

The generated `0.20 mm` mesh contains 2,530 original triangles and 864 new
quadrilateral skin elements.

## Model cut boundaries

`SkinLayer2DConfig.excluded_planes` accepts the existing `PlaneSelector` API.
Boundary edges whose two endpoints match a selector are omitted. With the
default `node_constraint="slip_plane"`, active/excluded junctions and the
shortened core boundary remain on the selected symmetry plane.

The current implementation supports plane-selected exclusions. Future selector
extensions may add:

1. Existing boundary Property IDs.
2. Explicit edge or node IDs.
3. Geometric boxes, circles, or angular ranges.

## Assessment of cylindrical subregions

Cylindrical coordinates are useful for geometric classification, but are not
required for the basic offset direction. A global centroid contraction would
indeed be unsuitable, while boundary-normal offsets already provide the local
direction needed around each slot and tooth.

Splitting every tooth into an independent morphing region is not recommended as
the default. Teeth and back yoke form one connected material region, and an
independent solve can create displacement discontinuities at artificial
tooth/yoke interfaces. If subregions are introduced as a fallback, they should
share constrained interface nodes and be solved as a coupled system. No skin
elements should be created on those artificial internal interfaces.

The back yoke is also not only an outer cylindrical boundary. Depending on the
physical skin definition, its outer circumference and its slot-side surfaces
are both material/air interfaces and may require layers. Radius and angular
range are better used as selectors for boundary roles and solver weighting than
as replacements for topology-based boundary extraction.

## Recommended implementation order

1. Add 2D opposing-segment/local-feature-size checks and optional local
   thickness reduction, analogous to Phase 10E in 3D.
2. Add explicit contour-intersection checks to the normal 2D quality gate.
3. Evaluate variable-stiffness Weighted Laplace near thin teeth and tooth roots.
4. Use coupled tooth/back-yoke subregions only where the global solve fails.
5. Use local remeshing when the requested thickness cannot be represented by
   the original core mesh without inversion.

## Reproduction

Run:

```powershell
python .\generated\skin_layer_sample\Dmodel\test_stator_skin_layer.py
python .\generated\skin_layer_sample\Dmodel\plot_stator_skin_layer.py
```

The scripts write meshes, plots, and `stator_skin_layer_report.json` under
`generated/skin_layer_sample/Dmodel`.
