# Neural Operator Scientific Computing

This repository is the documentation of my work during the summer of 2026 at the Indian Institute of Technology Hyderabad as a research intern under Prof. Srijith P.K. in the Department of Computer Science and Engineering. 

The repository is notebook-based and uses PyTorch for all core implementations. I experimented with Fourier Neural Operators (FNO), DeepONet, compact learnable spectral layers, transfer learning, residual-based adaptive refinement (RAR), equation normalization, and hybrid optimization. The experiments cover 1D and 2D settings, multiple parameter sweeps, velocity-estimation variants, and comparisons between fixed Fourier feature banks and lightweight learnable operator layers.

## Highlights

- Implemented Neural Operator architectures, including Fourier Neural Operators (FNO) and DeepONet, in PyTorch for modeling parameterized scientific computing problems.
- Developed a hybrid training pipeline with transfer learning, residual-based adaptive refinement (RAR), and equation normalization, achieving relative L2 loss as low as `1.50e-5` in the saved notebook results.
- Developed a compact Fourier-operator layer with `8` learnable spectral parameters and `8`-neuron hidden layers, matching baseline accuracy while reducing training time by `3x`.

## Implementations

I focused on learning mappings from parameters and coordinates to output fields. This setup is useful when the same class of scientific computing problem must be evaluated repeatedly under different parameter values.

The notebooks compare several modeling approaches:

| Component | What I Used It For |
| --- | --- |
| Fourier Neural Operators | Learning global structure through spectral transformations |
| DeepONet | Learning parameter-to-function mappings with branch and trunk networks |
| Fixed Fourier feature banks | Providing a strong high-frequency baseline representation |
| Learnable spectral layers | Replacing large fixed banks with compact trainable parameters |
| Hybrid optimization | Improving convergence after an initial warmup phase |
| Transfer learning | Reusing weights across related parameter settings |
| RAR | Adding points in high-residual regions to improve local accuracy |
| Equation normalization | Stabilizing training when objective terms had different scales |

## Architecture Details

### Fourier Neural Operator Work

I implemented FNO-style spectral layers in PyTorch to capture global patterns in parameterized solution fields. The Fourier-based layers were used to test whether spectral representations could replace heavier hand-designed feature banks.

Key details:

| Detail | Value |
| --- | --- |
| Framework | PyTorch |
| Main idea | Learn global structure using Fourier-domain transformations |
| Baseline representation | Fixed `65`-frequency feature bank |
| Compact representation | `8` learnable spectral parameters |
| Compact hidden width | `8` neurons per hidden layer |
| Training speedup | `3x` faster training in compact experiments |

### DeepONet Work

Implemented DeepONet models to compare against the Fourier-based operator approach. The branch network encoded parameter-dependent inputs, and the trunk network encoded coordinate locations. Their combined output represented the predicted field at query points.

Key details:

| Detail | Description |
| --- | --- |
| Architecture | Branch/trunk DeepONet |
| Implementation | PyTorch notebooks |
| Training setup | Parameter sweeps instead of single-case fitting |
| Evaluation | Relative L2 loss |
| Purpose | Compare operator-learning behavior against Fourier-based models |

## Hybrid Training Pipeline

Built a training workflow that combines multiple stabilization and convergence strategies. The main idea was to make training robust across a sequence of related parameter settings.

| Step | What I Did | Why It Helped |
| --- | --- | --- |
| Transfer learning | Reused model weights between related parameter settings | Reduced retraining cost and improved initialization |
| RAR | Added refinement points where residuals were high | Focused learning on harder regions |
| Equation normalization | Rescaled objective terms | Prevented one term from dominating the loss |
| Hybrid optimization | Used warmup training followed by refinement | Improved convergence and final relative L2 loss |
| Relative L2 tracking | Evaluated normalized prediction error | Made results comparable across configurations |

Several notebooks use `2,000` warmup iterations before refinement. Other experiments use adaptive refinement budgets based on the parameter value, and selected workflows include up to `2,500` collocation samples.

## Results From Notebooks

The saved notebook outputs report relative L2 loss using:

```text
relative L2 = ||prediction - reference||_2 / ||reference||_2
```

### Best Saved Relative L2 Results

| Experiment | Parameter Setting | Relative L2 Loss |
| --- | ---: | ---: |
| Transfer learning + equation normalization | `k = 46` | `1.50e-5` |
| Transfer learning + equation normalization | `k = 30` | `1.58e-5` |
| Transfer learning + equation normalization | `k = 40` | `1.60e-5` |
| Transfer learning + equation normalization | `k = 20` | `1.94e-5` |
| Velocity-estimation workflow without flow parameter | `k_R = 10.44, k_I = -1.50` | `2.06e-5` |
| Narrow-configuration workflow | `k_R = 20.14, k_I = -2.04` | `1.72e-5` |
| Narrow-configuration workflow | `k_R = 29.72, k_I = -2.45` | `2.21e-5` |
| Compact Fourier-operator layer | `k = 30` | `3.46e-5` |

### Transfer Learning + Equation Normalization Sweep

| Parameter | Relative L2 Loss |
| ---: | ---: |
| `k = 10` | `4.94e-4` |
| `k = 20` | `1.94e-5` |
| `k = 30` | `1.58e-5` |
| `k = 35` | `6.81e-5` |
| `k = 40` | `1.60e-5` |
| `k = 42` | `1.91e-5` |
| `k = 46` | `1.50e-5` |
| `k = 48` | `2.72e-5` |
| `k = 50` | `9.55e-4` |

### Compact Fourier-Operator Layer Sweep

| Parameter | Relative L2 Loss |
| ---: | ---: |
| `k = 10` | `2.73e-4` |
| `k = 20` | `4.34e-3` |
| `k = 30` | `3.46e-5` |
| `k = 35` | `2.06e-4` |
| `k = 40` | `3.08e-4` |
| `k = 42` | `3.17e-4` |
| `k = 46` | `5.21e-5` |
| `k = 48` | `2.37e-4` |
| `k = 50` | `1.21e-1` |

### 1D Operator Workflow Results

| Configuration | Parameter Setting | Relative L2 Loss |
| --- | ---: | ---: |
| No flow parameter | `k_R = 10.44, k_I = -1.50` | `6.65e-4` |
| No flow parameter | `k_R = 20.14, k_I = -2.04` | `8.21e-4` |
| No flow parameter | `k_R = 29.72, k_I = -2.45` | `4.60e-4` |
| No flow parameter | `k_R = 39.23, k_I = -2.79` | `6.39e-3` |
| Flow parameter enabled | `k_R = 10.44, k_I = -1.50` | `1.80e-4` |

## Compact Fourier-Operator Layer

One of the main improvements I worked on was replacing a large fixed Fourier feature bank with a much smaller learnable spectral layer. The original setup used a fixed bank of `65` frequencies. I replaced that with `8` learnable spectral parameters and `8`-neuron hidden layers.

| Model Design | Frequency Representation | Hidden Width | Training Time |
| --- | ---: | ---: | ---: |
| Baseline feature-bank model | `65` fixed frequencies | larger hidden layers | baseline |
| Compact Fourier-operator model | `8` learnable spectral parameters | `8` neurons | `3x` faster |

This experiment showed that a small trainable spectral representation can preserve strong accuracy while reducing model size and training cost.

## Notebook Guide

| Notebook Group | What It Contains |
| --- | --- |
| FNO base notebooks | Early Fourier-operator baselines and improved FNO experiments |
| DeepONet notebooks | Initial DeepONet implementation and parameter sweep experiments |
| 1D operator notebooks | No-flow, flow-enabled, compact, and full spectral variants |
| 2D operator notebooks | Circular, elliptical, and rectangular configuration studies |
| Transfer-normalization notebooks | Transfer learning, RAR, normalization, and hybrid training workflows |
| Velocity-estimation notebooks | Operator workflows for estimating velocity-related outputs |

## Tech Stack

| Tool | Use |
| --- | --- |
| Python | Core programming language |
| PyTorch | Neural operator implementation and training |
| NumPy | Numerical utilities |
| Matplotlib | Visualization and result plots |
| Jupyter Notebook | Experiment development and reporting |
