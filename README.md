# CoreEditor: Consistent 3D Editing via Correspondence-constrained Diffusion

This repository contains the PyTorch implementation of these papers:

> [**CoreEditor: Consistent 3D Editing via Correspondence-constrained Diffusion**](https://arxiv.org/abs/2508.11603)           
> [Zhe Zhu](https://czvvd.github.io/homepage/), [Honghua Chen](https://chenhonghua.github.io/clay.github.io/), Peng Li, [Mingqiang Wei](https://scholar.google.com/citations?user=TdrJj8MAAAAJ)      
> **IEEE TVCG, Accept**

<!-- ![example](teaser.png) -->

## Abstract

> Text-driven 3D editing seeks to modify 3D scenes according to textual descriptions, and most existing approaches tackle this by adapting pre-trained 2D image editors to multi-view inputs. However, without explicit control over multi-view information exchange, they often fail to maintain cross-view consistency, leading to insufficient edits and blurry details. We introduce CoreEditor, a novel framework for consistent text-to-3D editing. The key innovation is a correspondence-constrained attention mechanism that enforces precise interactions between pixels expected to remain consistent throughout the diffusion denoising process. Beyond relying solely on geometric alignment, we further incorporate semantic similarity estimated during denoising, enabling more reliable correspondence modeling and robust multi-view editing. In addition, we design a selective editing pipeline that allows users to choose preferred results from multiple candidates, offering greater flexibility and user control. Extensive experiments show that CoreEditor produces high-quality, 3D-consistent edits with sharper details, significantly outperforming prior methods.


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


