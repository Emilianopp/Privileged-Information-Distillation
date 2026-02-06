# Privileged Information Distillation for Language Models

A Python package providing loss functions and training utilities for privileged information distillation in language models.

## Overview

This package implements the core loss functions from the paper:

> **Privileged Information Distillation for Language Models**  
> Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, Massimo Caccia  
> arXiv:2602.04942, 2026  
> [[Paper]](https://arxiv.org/abs/2602.04942)

Privileged information (PI) refers to additional information available during training but not at inference time (e.g., internal reasoning traces, privileged observations, or expert demonstrations). This package provides efficient implementations of loss functions for distilling knowledge from PI-conditioned models to standard models.

## Features

- **π-Distill Loss**: Joint teacher-student objective for training PI-conditioned teachers and unconditioned students simultaneously
- **On-Policy Self-Distillation (OPSD)**: RL-based approach with reverse KL penalty between student and PI-conditioned teacher
- **PPO Loss**: Proximal Policy Optimization with clipped importance sampling
- **TOPR Loss**: Training Objectives with Positive-negative Rewards combining SFT and TIS
- **KL Divergence**: Rao-Blackwellized KL divergence estimator for efficient computation

## Installation

```bash
pip install -e .
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{penaloza2026privileged,
  title={Privileged Information Distillation for Language Models},
  author={Penaloza, Emiliano and Vattikonda, Dheeraj and Gontier, Nicolas and Lacoste, Alexandre and Charlin, Laurent and Caccia, Massimo},
  journal={arXiv preprint arXiv:2602.04942},
  year={2026}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the torchtune library and adapted for standalone use with privileged information distillation methods.
