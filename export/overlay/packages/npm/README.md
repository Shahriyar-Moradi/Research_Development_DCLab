# dclabai (npm launcher)

```bash
npx dclabai            # the studio in your browser (same as: dclab start)
npx dclabai init       # choose your model
npx dclabai doctor     # check the setup
```

This package runs the Python package [`dclabai`](https://pypi.org/project/dclabai/): through `uvx` when
[uv](https://docs.astral.sh/uv/) is installed, else `pipx`, else a virtual environment of its own in `~/.cache/dclabai/venv`
(made on the first run; it needs Python 3.10 or later). See the [project's README](https://github.com/Shahriyar-Moradi/dclabai).
