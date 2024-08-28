# Unlearn_LLM

## get toxic dataset

*Dataset 1:* RealToxicityPrompts [Dataset Link](https://huggingface.co/datasets/allenai/real-toxicity-prompts) | [Paper](https://arxiv.org/pdf/2009.11462)

## Run ROME with GPT2

Notes:

- The entire codebase for ROME can be found in the directory `rome-main`. Cloned from [Locating and Editing Factual Associations in GPT (NeurIPS'22)](https://arxiv.org/pdf/2202.05262).
- The `rome-main/causal_main.py` is the main script to run a vanilla example of causal tracing on one of the datasets in the original paper.
- The directory `rome-main/dsets` contains the datasets they use. This is where we need to add the RealToxicityPrompts dataset and load it from for inference.

For William:

- Refer to the RealToxicityPrompts paper to determine which model they used and pick one of the GPT-2 variants to run ROME with the prompts from the dataset.