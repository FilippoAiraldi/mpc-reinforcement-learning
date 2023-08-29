# Reinforcement Learning with Model Predictive Control

**mpcrl** is a library for training model-based Reinforcement Learning (RL) agents with Model Predictive Control (MPC) as function approximation. This framework, also known as MPC-based RL, was first proposed in [[1]](#1) and has so far been shown effective in various applications and with different learning algorithms, e.g., [[2](#2),[3](#3)].

[![PyPI version](https://badge.fury.io/py/mpcrl.svg)](https://badge.fury.io/py/mpcrl)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/casadi-nlp/blob/release/LICENSE)
![Python 3.9](https://img.shields.io/badge/python->=3.9-green.svg)

[![Tests](https://github.com/FilippoAiraldi/mpc-reinforcement-learning/actions/workflows/test-experimental.yml/badge.svg)](https://github.com/FilippoAiraldi/mpc-reinforcement-learning/actions/workflows/test-experimental.yml)
[![Downloads](https://static.pepy.tech/badge/mpcrl)](https://www.pepy.tech/projects/mpcrl)
[![Maintainability](https://api.codeclimate.com/v1/badges/9a46f52603d29c684c48/maintainability)](https://codeclimate.com/github/FilippoAiraldi/mpc-reinforcement-learning/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/9a46f52603d29c684c48/test_coverage)](https://codeclimate.com/github/FilippoAiraldi/mpc-reinforcement-learning/test_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Introduction

This framework merges two powerful control techinques into a single data-driven one

- MPC, a well-known control methodology that exploits a prediction model to predict the future behaviour of the environment and compute the optimal action

- and RL, a Machine Learning paradigm that showed many successes in recent years (with  games such as chess, Go, etc.) and is highly adaptable to unknown and complex-to-model environments.

<div align="center">
  <img src="https://raw.githubusercontent.com/FilippoAiraldi/mpc-reinforcement-learning/main/resources/mpcrl-diagram.png" alt="mpcrl-diagram" height="300">
</div>

The figure shows the main idea behind this learning-based control approach. The MPC controller, parametrized in $\vartheta$, acts both as policy provider (providing an action to the environment, given the current state) and as function approximation for the state and action value functions. Concurrently, an RL agent is employed to tune the parameters of the MPC in such a way to increase the controller's performance and achieve an (sub)optimal policy.

---

## Installation

To install the package, run

```bash
pip install mpcrl
```

**mpcrl** has the following dependencies

- [csnlp](https://pypi.org/project/csnlp/)
- [SciPy](https://scipy.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Numba](https://numba.pydata.org/)
- [typing_extensions](https://pypi.org/project/typing-extensions/)

For playing around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/mpc-reinforcement-learning.git
```

---

## Examples

Our [examples](https://github.com/FilippoAiraldi/mpc-reinforcement-learning/tree/main/examples) subdirectory contains an example application on a small linear time-invariant (LTI) system, tackled both with Q-learning and Deterministic Policy Gradient (DPG).

---

## License

The repository is provided under the MIT License. See the LICENSE file included with this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl” (Reinforcement Learning with Model Predictive Control) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.

---

## References

<a id="1">[1]</a>
S. Gros and M. Zanon, "Data-Driven Economic NMPC Using Reinforcement Learning," in _IEEE Transactions on Automatic Control_, vol. 65, no. 2, pp. 636-648, Feb. 2020, doi: 10.1109/TAC.2019.2913768.

<a id="2">[2]</a>
H. N. Esfahani, A. B. Kordabad and S. Gros, "Approximate Robust NMPC using Reinforcement Learning," _2021 European Control Conference (ECC)_, 2021, pp. 132-137, doi: 10.23919/ECC54610.2021.9655129.

<a id="3">[3]</a>
W. Cai, A. B. Kordabad, H. N. Esfahani, A. M. Lekkas and S. Gros, "MPC-based Reinforcement Learning for a Simplified Freight Mission of Autonomous Surface Vehicles," _2021 60th IEEE Conference on Decision and Control (CDC)_, 2021, pp. 2990-2995, doi: 10.1109/CDC45484.2021.9683750.
