# DAMA: Data- and Model-aware Alignment of Multi-modal LLMs
Implementation of the ICML 2025 paper: [Data- and Model-aware Alignment of Multi-modal LLMs](https://arxiv.org/abs/2502.01943).

## Introduction
DAMA: \textbf{Da}ta- and \textbf{M}odel-\textbf{a}ware DPO is a new method to dynamically adjust the preference optimization process of Multi-modal LLMs, and it consists of the following two aspects: (1) a data-aware strategy that incorporates data hardness, and (2) a model-aware strategy that integrates real-time model responses. By combining the two strategies, DAMA enables the model to effectively adapt to data with varying levels of hardness.
Extensive experiments on five benchmarks demonstrate that DAMA not only significantly enhances the trustworthiness, but also improves the effectiveness over general tasks. For instance, on the Object HalBench, our DAMA-7B reduces response-level and mentioned-level hallucination by 90.0\% and 95.3\%, respectively.
<\br>
The data-aware strategy:
<div align=center>
  <img src="https://github.com/injadlu/DAMA/blob/main/Figure-1.svg">
</div>
The model-aware strategy:
<div align=center>
  <img src="https://github.com/injadlu/DAMA/blob/main/Figure-2.svg">
</div>
Code for DAMA: DAMA code is built upon [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) <br>

Dataset: We employ the LLaVA-v1.5-7B preference data from [RLAIF-V Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)
