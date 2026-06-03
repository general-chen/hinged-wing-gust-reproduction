# Hinged-wing gust response reproduction

Standalone Python reproduction of the main results from:

> Stevenson et al. (2023), "Dynamics of hinged wings in strong upward gusts", *Royal Society Open Science* 10:221607. DOI: <https://doi.org/10.1098/rsos.221607>

This repo intentionally does **not** use the authors' supplied `.mat` files. It re-implements the model from the paper description and generates independent CSV/SVG outputs.

## What it implements

- 1-cosine upward gust profile from equation (2.15).
- Blade-element aerodynamic loads from equations (2.11)-(2.14).
- Linearized fixed-wing and hinged-wing equations from equations (2.5), (2.8), and (2.9).
- Triangular/linear spanwise wing mass distribution from equation (2.10).
- Linear lift curve (LLC) and soft-stall nonlinear lift curve (NLC).
- Parameter sweeps corresponding qualitatively to paper figures 3-6.

## Run

Use Python with NumPy installed:

```powershell
python reproduce_from_description.py
```

On the original Codex Windows workspace, the convenience launcher uses the bundled Python runtime:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_python_reproduction.ps1
```

## Outputs

The script writes generated files to `outputs_from_description/`:

- `figure3_from_description.svg` and `.csv`
- `figure4_from_description.svg` and `.csv`
- `figure5_from_description.svg` and `.csv`
- `figure6_from_description.svg` and `.csv`

These plots reproduce the paper's qualitative findings: hinged wings reduce fuselage reaction/velocity, soft stall improves rejection, heavier wings preserve centre-of-pressure alignment longer, and hinge stiffness degrades the rejection effect.

## Files

- `reproduce_from_description.py` - standalone solver and SVG/CSV generator.
- `run_python_reproduction.ps1` - Windows launcher for the Codex bundled Python runtime.
- `outputs_from_description/` - generated reproduction outputs.

## Notes

This is a compact reproducibility implementation rather than a line-for-line port of the authors' MATLAB supplement. Numerical values may differ slightly because this script uses a fixed-step RK4 integrator and hand-built SVG plotting, but the model structure, parameters, and qualitative dynamics follow the paper description.
