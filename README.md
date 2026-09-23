# 1D Compressible Reacting-Flow Solver with NSCBC

A one-dimensional compressible reacting-flow solver implementing **Navier–Stokes Characteristic Boundary Conditions (NSCBC)** for reacting-flow simulations with non-reflecting inflow and outflow boundaries.

The solver is designed primarily for the numerical study of **autoignition, reacting compressible flows, pressure-wave propagation, and boundary–acoustic interactions**.

An optional machine-learning framework is also included for learning the relationship between NSCBC relaxation parameters and the resulting pressure response.

---

## Authors

**Nishant Vinod Bhorkar**
Indian Institute of Science (IISc)

**Aritra Roy Choudhury**
Indian Institute of Science (IISc)

---

## Overview

Artificial boundaries in compressible-flow simulations can generate spurious acoustic reflections that contaminate the solution, particularly when pressure waves, reaction fronts, or other strong disturbances reach the computational boundaries.

This project implements characteristic inflow and outflow boundary conditions based on the **Navier–Stokes Characteristic Boundary Condition (NSCBC)** methodology.

The implementation follows the characteristic-boundary formulation described by:

> C. S. Yoo and H. G. Im,
> “Characteristic boundary conditions for simulations of compressible reacting flows with multi-dimensional, viscous and reaction effects,”
> *Combustion Theory and Modelling*, Vol. 11, No. 2, pp. 259–286, 2007.
> DOI: **10.1080/13647830600898995**

The present repository contains a one-dimensional implementation intended for reacting-flow and autoignition calculations.

---

## Main Features

The solver includes:

* one-dimensional compressible reacting-flow equations;
* conservation of mass, momentum, total energy, and chemical species;
* finite-rate chemical kinetics through **Cantera**;
* temperature-dependent thermodynamic properties;
* viscosity and thermal conduction;
* mixture-averaged species diffusion;
* chemical reaction source terms;
* NSCBC inflow boundary conditions;
* NSCBC non-reflecting outflow boundary conditions;
* characteristic-wave treatment at the domain boundaries;
* relaxation of inlet temperature, velocity, and species composition;
* outlet pressure relaxation;
* automatic detection of ignition;
* pressure, momentum, and density probes;
* visualization of the evolving solution;
* optional machine-learning-based adjustment of NSCBC damping parameters.

---

# Governing Equations

The solver considers the one-dimensional compressible reacting Navier–Stokes equations.

The conservative state vector is of the form

$$
\mathbf{Q}
=
\begin{bmatrix}
\rho \\
\rho u \\
\rho E \\
\rho Y_1 \\
\vdots \\
\rho Y_{N_s}
\end{bmatrix},
$$

where

* \(\rho\) is the mixture density,
* \(u\) is the velocity,
* \(E\) is the specific total energy,
* \(Y_k\) is the mass fraction of species \(k\),
* \(N_s\) is the number of chemical species.

The equations contain convective transport, viscous transport, heat conduction, species diffusion, and chemical reaction source terms.

Thermodynamic, transport, and chemical-kinetic properties are evaluated using **Cantera**.

---

# Navier–Stokes Characteristic Boundary Conditions

A major component of this project is the implementation of **NSCBC** at the inlet and outlet.

Instead of prescribing all primitive variables directly at the boundaries, the governing equations are decomposed into characteristic waves.

For a one-dimensional flow, the characteristic information is associated with wave speeds involving

$$
u-c,\qquad u,\qquad u+c,
$$

where \(c\) is the local speed of sound.

The boundary treatment determines which characteristic waves enter the computational domain and which leave it.

The implementation includes contributions associated with:

* acoustic characteristics,
* entropy/density characteristics,
* species characteristics,
* viscous stresses,
* heat conduction,
* species diffusion,
* and chemical reactions.

The inlet uses relaxation toward specified target conditions, while the outlet uses pressure relaxation toward a prescribed far-field pressure.

The implementation is based on the NSCBC methodology described by Yoo and Im (2007).

---

# Autoignition Problem

The current configuration considers an initially premixed hydrogen–air reacting mixture.

The default initial conditions used by the solver are approximately

```text
Temperature      = 1100 K
Pressure         = 1 atm
Velocity         = 200 m/s
```

with initial mass fractions

```text
H2  = 0.00347816
O2  = 0.18401789
H2O = 0.05206297
N2  = 0.76044098
```

Autoignition is detected when the maximum temperature in the computational domain reaches

```text
Tmax >= 1400 K
```

The solver records both the autoignition iteration and corresponding physical time.

---

# Default Numerical Configuration

The current default computational setup is

```text
Domain length     : 0.05 m
Grid points       : 256
Time step         : 5.0e-9 s
```

The interior convective derivatives use high-order central finite-difference approximations, with appropriate one-sided/near-boundary formulas.

Viscous, thermal, and species-diffusion derivatives are also evaluated using finite differences.

A Savitzky–Golay filter is periodically applied after the initial transient to suppress high-frequency numerical oscillations.

The default NSCBC relaxation parameters are

```text
damp1 = 0.27
damp2 = 0.27
damp3 = 0.27
```

These parameters control the relaxation of the characteristic boundary conditions.

---

# Chemical Kinetic Mechanism

The chemistry is defined in

```text
chem_Lietal.yaml
```

and is loaded using Cantera.

The supplied mechanism contains the species

```text
H2
O2
O
OH
H2O
H
HO2
H2O2
N2
```

and uses an ideal-gas thermodynamic model with mixture-averaged transport.

The mechanism file must remain in the same directory as `1DSolver_autoignition.py`, unless the mechanism path in the solver is modified.

---

# Repository Structure

A recommended repository structure is

```text
project/
│
├── README.md
│
├── src/
│   ├── 1DSolver_autoignition.py
│   ├── chem_Lietal.yaml
│   ├── control_setup.py
│   ├── ml_controller.py
│   └── run_training_loop.py
│
└── results/
```

The individual source files have the following purposes.

### `1DSolver_autoignition.py`

Main one-dimensional reacting-flow solver.

It contains:

* flow initialization,
* compressible conservation equations,
* thermodynamic-property evaluation,
* species transport,
* chemical source terms,
* NSCBC inflow and outflow treatment,
* autoignition detection,
* pressure/momentum/density probes,
* plotting,
* and optional interaction with an external pressure controller.

### `chem_Lietal.yaml`

Cantera chemical mechanism containing thermodynamic, kinetic, and transport data for the hydrogen oxidation system.

### `control_setup.py`

Defines the parameter space used for generating ML training simulations.

The default training space varies:

```text
target pressure
damp1
damp2
damp3
```

The default configuration also defines parameters such as pressure-history length, prediction horizon, batch size, learning rate, and number of training epochs.

### `ml_controller.py`

Contains the neural-network pressure-response model.

The model learns a mapping of the form

```text
current flow/control state
        +
pressure history
        +
target pressure
        +
candidate NSCBC damping parameters

                ↓

predicted future pressure
```

The trained model can evaluate several candidate damping combinations and select parameters expected to drive the pressure response toward the target pressure.

### `run_training_loop.py`

Runs a sequence of reacting-flow simulations over the prescribed NSCBC parameter space and uses the resulting data to train the pressure-response model.

---

# Requirements

The code requires Python 3 and the following packages:

```text
numpy
pandas
scipy
matplotlib
cantera
torch
```

A Python virtual environment is recommended.

For example:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages using

```bash
pip install numpy pandas scipy matplotlib cantera torch
```

---

# Running the Base NSCBC Solver

Move into the source directory:

```bash
cd src
```

Run

```bash
python3 1DSolver_autoignition.py
```

The default execution calls

```python
run_simulation()
```

with the standard NSCBC parameters and simulation configuration.

---

# Simulation Function

The main solver can also be imported from another Python script.

The principal interface is

```python
run_simulation(
    p_target=None,
    damp_init=(0.27, 0.27, 0.27),
    controller=None,
    candidate_damps=None,
    control_interval=10,
    end_count=200000,
    write_files=True,
    make_plots=True,
    plot_dir='images_0.27',
    verbose=True,
)
```

This makes it possible to run the solver either as a standalone simulation or as part of an optimization/control framework.

---

# Simulation Outputs

The solver records pressure, momentum, and density at a probe located approximately at

$$
x = 0.1L.
$$

When file output is enabled, the histories are written to

```text
pressureProbe27.txt
momentumProbe27.txt
densityProbe27.txt
```

The returned `SimulationResult` also contains

```python
pressure_probe
momentum_probe
density_probe
max_temperature
damp_history
autoignition_step
autoignition_time
dt
```

When plotting is enabled, snapshots of the evolving solution are written to the specified image directory.

---

# Machine-Learning Pressure Controller

The repository also contains an experimental ML framework for learning how the NSCBC relaxation parameters affect the pressure response following autoignition.

The neural network uses information including

* recent pressure history,
* target pressure,
* current pressure error,
* pressure variation,
* maximum temperature,
* time since autoignition,
* current damping parameters,
* and candidate damping parameters.

The model predicts the future pressure at the probe.

Candidate NSCBC damping parameters can then be evaluated and the combination giving a predicted pressure closest to the requested target can be selected.

A penalty is also included to discourage unnecessarily large changes in the damping parameters.

---

# Training the ML Model

The default training parameter space is defined in

```text
control_setup.py
```

To start sequential training, run

```bash
cd src
python3 run_training_loop.py
```

For every training case, the script:

1. selects a target pressure and damping combination;
2. runs the 1D reacting-flow solver;
3. detects autoignition;
4. extracts post-ignition pressure histories;
5. constructs training samples;
6. trains the neural network;
7. saves the updated model.

The default trained model is stored as

```text
pressure_response_model.pt
```

The default parameter sweep uses target pressures between approximately

```text
0.95 atm and 1.15 atm
```

and varies each of the three damping parameters over the prescribed training range.

---

# Neural-Network Architecture

The current pressure-response model is a fully connected neural network with the approximate architecture

```text
Input
  ↓
64 neurons + Tanh
  ↓
64 neurons + Tanh
  ↓
32 neurons + Tanh
  ↓
1 output
```

The scalar output represents the predicted future pressure.

Training uses

```text
Optimizer : Adam
Loss      : Mean Squared Error
```

The model automatically uses CUDA when a compatible GPU is available and otherwise runs on the CPU.

---

# Important Note on the ML Component

The machine-learning portion of this repository should currently be considered an **experimental control framework**.

Its performance depends strongly on:

* the range and quality of generated training simulations,
* the selected NSCBC damping space,
* pressure targets,
* prediction horizon,
* pressure-history length,
* and the range of physical operating conditions represented during training.

A trained model should therefore not be assumed to generalize outside the parameter range represented by its training data without additional validation.

---

# Reference

If this repository or its NSCBC implementation is used in academic work, the underlying characteristic-boundary-condition methodology should be cited:

```bibtex
@article{YooIm2007,
  author  = {Yoo, C. S. and Im, H. G.},
  title   = {Characteristic boundary conditions for simulations of compressible reacting flows with multi-dimensional, viscous and reaction effects},
  journal = {Combustion Theory and Modelling},
  volume  = {11},
  number  = {2},
  pages   = {259--286},
  year    = {2007},
  doi     = {10.1080/13647830600898995}
}
```

---

# Usage and Permission

This code has been developed for **research and academic purposes**.

Any individual, research group, institution, or organization wishing to use, modify, reproduce, redistribute, or incorporate this code, either in whole or in part, into another project, publication, software package, or research work must obtain **prior permission from the authors**.

Permission should be obtained from:

* Nishant Vinod Bhorkar
* Aritra Roy Choudhury

Appropriate acknowledgement of the authors and citation of the relevant scientific literature, particularly the NSCBC methodology of Yoo and Im (2007), should accompany any authorized use of this code.

Redistribution of the code or presentation of the implementation as original work by another individual or organization without authorization is not permitted.

---

# Disclaimer

This repository is research software and is provided for scientific investigation and development.

Users are responsible for independently verifying numerical accuracy, physical assumptions, chemical mechanisms, boundary conditions, stability, and suitability of the solver for their intended application.

