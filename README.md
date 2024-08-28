# Causal Representation Learning Approach for Unlearning Toxic Content in LLMs

## Dataset

**Dataset 1:** `RealToxicityPrompts` [[Dataset Link](https://huggingface.co/datasets/allenai/real-toxicity-prompts)] | [[Paper](https://arxiv.org/pdf/2009.11462)]

## Run ROME with GPT2

Notes:

- The entire codebase for `ROME` can be found in the directory `rome-main`. Cloned from [Locating and Editing Factual Associations in GPT (NeurIPS'22)](https://arxiv.org/pdf/2202.05262).
- The `rome-main/trace_main.py` is the main script to run a vanilla example of causal tracing on one of the datasets in the original paper.
- The directory `rome-main/dsets` contains the datasets they use. This is where we need to add the `RealToxicityPrompts` dataset and load it from for inference.

For William:

- Refer to the `RealToxicityPrompts` paper to determine which model they used and pick one of the GPT-2 variants to run `ROME` with the prompts from the dataset.
- Choose the "challenging" subset of prompts, i.e., where `dataset["challenging"] == true`
- For our use case, right now just the causal tracing part is enough, we don't need to worry about the editing part yet.