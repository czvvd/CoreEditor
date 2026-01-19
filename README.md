# CoreEditor: Consistent 3D Editing via Correspondence-constrained Diffusion

This repository contains the PyTorch implementation of these papers:

> [**CoreEditor: Consistent 3D Editing via Correspondence-constrained Diffusion**](https://arxiv.org/abs/2508.11603)           
> [Zhe Zhu](https://czvvd.github.io/homepage/), [Honghua Chen](https://chenhonghua.github.io/clay.github.io/), Peng Li, [Mingqiang Wei](https://scholar.google.com/citations?user=TdrJj8MAAAAJ)      
> **IEEE TVCG, Accept**

<!-- ![example](teaser.png) -->

## Abstract

> In this paper, we propose a novel network, SVDFormer, to tackle two specific challenges in point cloud completion: understanding faithful global shapes from incomplete point clouds and generating high-accuracy local structures. Current methods either perceive shape patterns using only 3D coordinates or import extra images with well-calibrated intrinsic parameters to guide the geometry estimation of the missing parts. However, these approaches do not always fully leverage the cross-modal self-structures available for accurate and high-quality point cloud completion. To this end, we first design a Self-view Fusion Network that leverages multiple-view depth image information to observe incomplete self-shape and generate a compact global shape. To reveal highly detailed structures, we then introduce a refinement module, called Self-structure Dual-generator, in which we incorporate learned shape priors and geometric self-similarities for producing new points. By perceiving the incompleteness of each point, the dual-path design disentangles refinement strategies conditioned on the structural type of each point.
SVDFormer absorbs the wisdom of self-structures, avoiding any additional paired information such as color images with precisely calibrated camera intrinsic parameters. Comprehensive experiments indicate that our method achieves state-of-the-art performance on widely-used benchmarks.

# Code will come soon!

## Citation
```
@article{zhu2025coreeditor,
  title={CoreEditor: Consistent 3D Editing via Correspondence-constrained Diffusion},
  author={Zhu, Zhe and Chen, Honghua and Li, Peng and Wei, Mingqiang},
  journal={arXiv preprint arXiv:2508.11603},
  year={2025}
}
```

## License

This project is open sourced under MIT license.


