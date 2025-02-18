# <p align="center">Sequential Prune</p>

<h3 align="center">
  <p>This repository contains the official implementation of <b>Sequential Prune: Progressive Visual Token Reduction for Improving the Inference Efficiency of Large Vision-Language Models</b></p>
</h3>


## Abstract
>Large Vision-Language Models face significant challenges due to the computational complexity and inference latency caused by processing numerous visual tokens. While token pruning method have been developed to address this issue, existing methods often remove a large number of visual tokens at specific layers, which can lead to the loss of important visual information and subsequent performance degradation. In this study, we propose a novel token pruning method called Sequential Prune to overcome these limitations. Sequential Prune progressively removes less important visual tokens at each layer based on their significance, thereby preserving critical visual information. Applying Sequential Prune to LVLMs, we achieved superior performance across various datasets compared to existing baselines, while maintaining similar levels of actual latency. These results confirm that Sequential Prune is an effective method for efficiently pruning visual tokens in LVLMs, enhancing computational efficiency without losing important information.


## Run Code

### Recommended Hardware

The experiments on the LLaVA-1.5-13B model were performed on a single A100-40GB GPU, and the other models were performed on a four A5000 24GB GPU.

### Installation

To clone this repository along with its submodules, use the following command:
    
    git clone https://github.com/LeeBumSeok/Tokenpruning

### Requirements

Ensure you have Python 3.10+ installed along with the required dependencies. Install the necessary libraries using the provided scripts:

    # LLaVa
    bash install_seqprune_llava.sh

    # QwenVL2
    bash install_seqprune_qwenvl2.sh


### Quick Start

After cloning the repository, you can easily run the Sequencial Prune method with the provided script. Use the following command to run:
    
    # LLaVa
    bash SequentialPrune/eval/VLMEvalKit/demo_llava.sh

    # LLaVa(MultiGPU)
    bash SequentialPrune/eval/VLMEvalKit/demo_multigpu_llava.sh

    # QwenVL2
    bash SequentialPrune/eval/VLMEvalKit/demo_qwenvl2.sh

    # QwenVL2(MultiGPU)
    bash SequentialPrune/eval/VLMEvalKit/demo_multigpu_qwenvl2.sh


## Related Work

This research is inspired by ideas from previous work on token pruning, particularly [FastV (Chen et al., 2024)](https://arxiv.org/pdf/2403.06764).

<!-- ## Citation
```bibtex
``` -->
