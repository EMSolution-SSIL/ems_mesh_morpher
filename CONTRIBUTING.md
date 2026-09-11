# Contributing

Use Python 3.10 or newer. Install the package and test dependencies in editable
mode:

```powershell
python -m pip install -e ".[test]"
```

Run the regression suite before submitting a change:

```powershell
python -m pytest tests -q
```

Keep public APIs backward compatible within a minor release. Add focused tests
for numerical behavior, mesh metadata preservation, and every newly supported
cell type or file format. Generated meshes and solver output should only be
committed when they are stable regression fixtures with documented provenance.
