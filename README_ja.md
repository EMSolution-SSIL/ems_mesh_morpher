# ems_mesh_morpher

`ems_mesh_morpher` は、EMSolution のワークフローで再利用できるメッシュ移動および局所モーフィング領域の機能を提供します。平行移動、回転、スケーリング、規定変位、Property／幾何形状による選択、radial、distance、IDW、compact-support RBF、weighted Laplace による変位伝播、基本的な2次元メッシュ品質評価、および `ems_file_format_converter` を利用したメタデータ保持付き出力（Gmsh 4.x を含む）に対応しています。Phase 1 で実装したモータ偏心インターフェースも、後方互換性を保つ簡便なラッパーとして引き続き利用できます。

本プロジェクトは [PolyForm Perimeter License 1.0.1](LICENSE) に基づく Source-Available ソフトウェアです。このライセンスでは広範な利用、改変、再配布が認められていますが、本ソフトウェアと競合する製品を提供することは認められていません。そのため、OSI が承認したオープンソースライセンスではありません。

## セットアップ

PyPI から公開パッケージをインストールします。

```powershell
python -m pip install ems-mesh-morpher
```

editable install で開発する場合は、このリポジトリを使用します。

```powershell
cd C:\EMSolution\build\MeshMorphingWorkspace\ems_mesh_morpher
conda run -n ems-py311 python -m pip install -e .
```

コマンドラインのエントリーポイントは `ems-mesh-morph` です。

```powershell
conda run -n ems-py311 ems-mesh-morph --help
```

## 基本的な使い方

```powershell
conda run -n ems-py311 ems-mesh-morph INPUT_MESH OUTPUT_MESH --config CONFIG_JSON
```

2種類の設定モードに対応しています。

- `motor_eccentricity_2d`: Phase 1 で実装した、半径方向エアギャップを対象とするモータ偏心ワークフロー
- `mesh_motion`: Phase 2/3 で実装した、汎用メッシュ移動およびモーフィング領域ワークフロー

Python API も利用できます。

```python
from ems_mesh_morpher import apply_mesh_motion, load_config

config = load_config("examples/general/local_zone_translation.json")
result = apply_mesh_motion(mesh, config)
print(result.summary())
```

EMSolution / pyemsol で使用する場合は、出力形式を Gmsh 4.x にしてください。GL80 のサンプル設定では、すでに以下のように設定されています。

```json
{
  "output": {
    "file_format": "gmsh4"
  }
}
```

## 設定ファイルの構造

モータ偏心のサンプル設定は `examples/GL80` 以下にあります。モータのメッシュおよびソルバー入力ファイルは、この公開リポジトリには含まれていません。各自のモデルを用意し、そのモデルに合わせて Property ID を調整してください。

- `static_stator_eccentricity_*.json`
  - 静的偏心（static eccentricity）
  - ステータメッシュを移動／変形し、ロータメッシュは固定
- `dynamic_rotor_eccentricity_*.json`
  - 動的偏心（dynamic eccentricity）
  - ロータメッシュを移動／変形し、ステータメッシュは固定

偏心量は `motion.displacement` で指定します。

```json
"motion": {
  "type": "translation",
  "displacement": [0.00005, 0.0]
}
```

GL80 のメッシュ単位は m です。

- 0.01 mm = `0.00001`
- 0.05 mm = `0.00005`
- 0.1 mm = `0.0001`

## 汎用メッシュ移動とモーフィング領域

汎用サンプルは `examples/general` 以下にあります。

対応している motion type:

- `translation`: 2成分または3成分の `displacement`
- `rotation`: `center`、`axis`、`angle_degrees`
- `scaling`: `center` と、スカラーまたは2軸／3軸の `scale`
- `prescribed`: `point_displacements` に各点の変位を指定

各 region では `property_ids`、`point_indices`、および1つの `geometry` または複数の `geometries` を指定できます。複数条件は `union` または `intersection` で組み合わせることができます。利用可能な geometry は `all`、`box`、`circle`、`annulus`、`sphere`、有限長の `cylinder` です。

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

`morphing_zone` は変形可能節点だけを制限します。moving 節点には指定された移動量がそのまま与えられ、fixed および protected 節点は移動しません。`distance_blend` を使用するには、少なくとも1つの moving 節点と、1つの fixed または protected 節点が必要です。節点の割り当てでは protected が最優先されます。

`morphing.type` には `"idw"` も指定できます。moving 節点は非ゼロ変位の制御点、fixed/protected 節点はゼロ変位の制御点となり、IDW は deformable 節点に対してのみ評価されます。デフォルトでは、最近傍32個の制御点と `2.0` の power を使用します。

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

アクチュエータ解析結果などと直接連携する場合、汎用 API には制御節点の変位または移動先座標のどちらでも与えることができます。

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

`field.displacement` と `field.target_points` は、元の節点順序に対応する全節点配列です。変位場を適用しても、要素接続、Property／Material データ、Node ID／Element ID は保持されます。そのため、同一メッシュを `ems_file_format_converter` で、対象の EMSolution ワークフローがサポートする形式に出力できます。`apply_mesh_motion` は、従来どおり全節点分の外部変位配列も受け取れます。JSON/CLI 形式については `examples/general/idw_prescribed_displacement.json` を参照してください。

同じ汎用インターフェースは、疎な compact-support RBF でも利用できます。デフォルトでは Wendland C2 kernel と affine polynomial term を使用します。affine term により、平行移動およびほぼ線形な変位場の再現性が向上します。

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

`radius=None` の場合、support radius は制御点と評価点の間隔から推定されます。大規模モデルでは `radius` を明示的に指定することで、疎性と実行時間をより予測しやすくできます。`polynomial_degree=-1` では polynomial term を無効化し、`0` では定数項、`1` ではデフォルトの affine term を使用します。JSON/CLI 入力については `examples/general/rbf_prescribed_displacement.json` を参照してください。

Weighted Laplace は空間距離だけでなく、実際のメッシュ接続関係を使用します。line、triangle、quadrilateral、tetrahedral、hexahedral、wedge、pyramid の各セルに対応しています。体積要素と境界要素が混在する場合、重複するエッジを除去することで境界面による重みの偏りを防ぎつつ、独立した表面領域は保持します。

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

JSON/CLI 入力では `morphing.type = "weighted_laplace"`（または短い alias の `"laplace"`）を指定します。比較用に `weighting="uniform"` も利用できます。連結した各 deformable component には、moving または fixed の制御境界が含まれている必要があります。アンカーされていない component はエラーとして報告されます。`examples/general/weighted_laplace_prescribed_displacement.json` を参照してください。

小さな要素やアスペクト比の大きな要素を含むメッシュでは、`weighting="inverse_distance_element"` を指定すると、隣接要素サイズとエッジのアスペクト比を利用してグラフ剛性を高めます。`element_size_power` と `element_aspect_power` でそれぞれの寄与を調整します。prepared API では疎行列の構築と factorization を1回だけ行い、各フレームでは新しい制御値を与えて解くことができます。

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

## 繰り返し移動と品質管理

`RepeatedMotionRunner` は、各フレームを常に未変更の参照メッシュから評価します。これにより、座標変位の累積誤差を防ぎます。region 定義とメッシュトポロジーが同じ場合、Weighted Laplace の行列 factorization は再利用されます。

```python
from ems_mesh_morpher import MotionSchedule, RepeatedMotionRunner

schedule = MotionSchedule.translations(
    times=[0.0, 0.5, 1.0],
    displacements=[[0.0, 0.0], [0.001, 0.0], [0.0, 0.0]],
)
run = RepeatedMotionRunner(reference_mesh, config).run(schedule)
print(run.summary())
```

共通の品質評価 API は triangle、quad、tetrahedron、pyramid、wedge、hexahedron の各セルを評価します。要素の反転、縮退、最小 signed measure ratio、最小 scaled Jacobian、最悪要素を報告します。メッシュ移動設定では、不正な結果を棄却するか、制限付き line search により要求変位を縮小することができます。

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

IDW、RBF、要素情報を考慮した Weighted Laplace の平行移動／回転 benchmark は以下で実行できます。

```powershell
python .\examples\repeated_motion\benchmark.py --model all --frames 21 `
  --output .\generated\repeated_motion_benchmark.json
```

## pyemsol External DEFORM 連成

Version 0.12 では、pyemsol の `describe_deformation()` および `describe_deformation_mesh()` が返す dictionary を利用できます。静的な topology は最初に1回だけ検証され、`evaluate()` を呼び出すたびに `CoupledSession.step(deformation=...)` が受け付ける絶対座標形式の payload を返します。adapter 自体は Python dictionary のみを受け渡すため、pyemsol は必須依存パッケージではありません。

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

adapter は、各 external node ID を mesh-local index に対応付け、出力順序を `return_node_ids` の順序に保ちます。rigid 節点は prescribed control、deformable 節点は補間対象、固定された共有境界節点は参照座標のままとなります。既存モデルの一部では、schema v2 で固定外周境界節点が `deformable` として公開される場合があります。デフォルトでは、schema v1 の endpoint displacement がゼロの節点を fixed anchor として推定します。この挙動は `PyEMSolConstraintConfig(infer_zero_displacement_anchors=False)` で無効にできます。

回転など、パラメータに対して非線形な rigid motion を与える場合は、external node ID をキーとした global XYZ target を明示的に指定します。region 内のすべての rigid 節点を指定する必要があります。XYZ 各方向の独立した平行移動だけであれば、より簡単に `rigid_translations` を利用できます。

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

payload を返す前に、adapter は triangle、quad、tetra、prism/wedge、hexahedron、pyramid 要素の relative signed area または volume をチェックします。要素の反転、縮退、measure ratio の不足が検出された場合は、EMSolution が次の step に進む前にエラーとして棄却されます。

pyemsol のモデルディレクトリ（solver input と mesh file を含む）を指定して、adapter のサンプルを実行できます。

```powershell
python .\examples\pyemsol_deform_sample\run_external_morphing.py `
  C:\path\to\model --input_file input.json --method weighted_laplace
```

このインターフェースが更新するのは座標のみです。mesh replacement と local remeshing はこの version の対象外です。

## Phase 10C-10E 多層生成機能

Version 0.9 では、四面体または六面体導体の表面に1層以上の体積 skin layer を生成できます。三角形境界面からは wedge 要素、四角形境界面からは hexahedral 要素を生成します。矩形の鋭角 corner および除外対象の symmetry plane に対応しています。Phase 10E では、対向面との collision check および明示的な local thickness control が追加されています。

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

`thickness` は skin layer 全体の厚さです。`layer_count` は層数、`growth_ratio` は外表面から core 側へ向かう隣接層間の厚さ比を指定します。`1.0` の場合は等厚です。`1.0` より大きい場合は外側の層ほど薄くなります。たとえば3層で ratio が `2.0` の場合、全厚を `1:2:4` の比率で分割します。

`SkinLayerCollisionConfig.policy` は Phase 10E の挙動を制御します。

- `"fail"`: デフォルト。要求厚さが局所エッジ長または対向面による制限を超える場合、生成前に停止
- `"reduce"`: 節点ごとの局所的な厚さ低減を明示的に許可。デフォルトの opposing surface factor は `0.4` で、両側から layer を生成する狭い領域に少なくとも20%の空間を残します
- `"allow"`: 厚さを変更せずにリスクのみを記録。通常の3次元 quality gate では、反転要素や過度に劣化した要素を引き続き棄却

節点ごとの要求値、最大許容値、実際の適用値は `result.local_thickness` で取得できます。summary report には、適用厚さの最小値、最大値、平均値、厚さを低減した節点数、対向面リスク数、検出された最小 clearance、および厚さ低減を使用した場合の warning が含まれます。局所的な低減値は、表面エッジに沿って制限付き thickness gradient を用いて平滑化されます。

既存の symmetry surface Property は、平面方程式を指定せずに除外できます。

```python
config = SkinLayerConfig(
    thickness=0.0001,
    volume_property_ids=(100,),
    boundary_roles=BoundaryRoleConfig(excluded_surface_property_ids=(210,)),
)
result = generate_skin_layer(mesh, config)
```

出力形式は拡張子から選択されます。明示的に指定することもできます。

```python
result.write("conductor_with_skin.atl")
result.write("conductor_with_skin.unv")
result.write("conductor_with_skin.neu")
result.write("conductor_with_skin.msh", file_format="gmsh4")
```

`CoreMorphingConfig.method` には `"auto"`、`"distance_blend"`、`"idw"`、`"rbf"`、`"laplace"`、`"none"` を指定できます。`"auto"` は最初に distance blending を試し、3次元 quality gate で棄却された場合に sparse Laplace solver で再試行します。report には、選択された method、要求した normal thickness と実際の thickness、要素数、最小 core volume ratio、scaled Jacobian、棄却された試行からの warning が記録されます。IDW と RBF は、比較による品質・性能評価が完了する前に既存の `"auto"` の結果を変えないよう、現時点では明示的に選択した場合のみ使用されます。Laplace backend は `CoreMorphingConfig(laplace=LaplaceConfig(...))` を使用し、汎用メッシュ移動と同じ weighted solver を共有します。

表面抽出、境界分類、offset target のみが必要な場合は、低レベルの Phase 10B API も引き続き利用できます。

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

現在の制限事項: 入力は一次の tetrahedron/hexahedron のみ。層ごとの厚さを明示するリスト指定、および skin depth に基づく厚さ計算には未対応です。concave corner はデフォルトで明示的にエラーとなります。Phase 10E では局所エッジ長および inward-ray／opposing-surface check を使用します。offset surface 全体の交差修復および local remeshing にはまだ対応していません。

## Skin-layer サンプルデータ

付属の Femap および ATLAS サンプルは、プロジェクト root からまとめて生成できます。

```powershell
conda run -n ems-py311 python .\examples\skin_layer_sample\generate_samples.py --layer_count 3 --growth_ratio 1.0 --collision_policy fail
```

出力と `reports.json` は `generated/skin_layer_sample` に保存されます。サンプル設定では、領域境界面 `x=0`、`y=0`、`z=0` を excluded surface として扱うため、これらの平面上には skin volume を生成しません。

`--thickness` を指定すると、選択した1つのサンプルに対して全厚を上書きできます。たとえば、以下では全厚 `0.012` の等厚3層を生成します。

```powershell
conda run -n ems-py311 python .\examples\skin_layer_sample\generate_samples.py --sample block_hexa --thickness 0.012 --layer_count 3 --growth_ratio 1.0 --output_dir .\generated\skin_layer_custom
```

利用可能なサンプル名は `block_hexa`、`block_tetra`、`atlas_block_conductor`、`atlas_curved_coil` です。誤って別モデルへ thickness override を適用することを防ぐため、`--thickness` を使用する場合は `--sample` を1つだけ指定する必要があります。`--thickness` を指定しない場合、選択された各サンプルでは以下に示す全厚を使用します。従来のハイフン区切りのサンプル名も互換性のため使用できますが、ドキュメントでは underscore 区切りを正式な表記としています。

| Sample | Target property | Total thickness | Three-layer output |
|---|---:|---:|---|
| `post_geom_hexa.neu` | 1 | 0.01 | 375 hexahedra |
| `post_geom_tetra.neu` | 1 | 0.01 | 750 wedges |
| `post_geom.atl` block conductor | 1 | 0.0025 | 375 hexahedra |
| `post_geom.atl` curved coil | 3 | 0.002 | 585 hexahedra |

4つのケースはすべて3次元 quality gate を通過し、元のファイル形式への round-trip も確認されています。curved coil では、実際の face-normal thickness は 0.0019754～0.002 です。このわずかな差は、円弧に沿った smooth normal averaging によるものです。all-prism の参照メッシュは正しく読み込まれますが、現在の generator は tetrahedron/hexahedron のみを対象としているため、意図的に棄却されます。

## 英語版README

英語版は [`README.md`](README.md) を参照してください。
