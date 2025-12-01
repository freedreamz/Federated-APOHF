# Federated Prompt Optimization with Human Feedback

This is the official implementation of the paper [Prompt Optimization with Human Feedback](https://arxiv.org/abs/2405.17346).

**Oral Presentation at ICML 2024 Workshop on Models of Human Feedback for AI Alignment**

📺 [Video Presentation](https://www.bilibili.com/video/BV13daQeuEgb)

## Overview

This project implements **Federated Contextual Bandits** for prompt optimization with human feedback. It supports three main applications:

| Application | Script | Description |
|-------------|--------|-------------|
| Prompt Optimization | `run_dbandits_po.py` | Optimizes instruction prompts for NLP tasks |
| Image Generation | `run_dbandits_image_gen.py` | Optimizes prompts for DALL-E image generation |
| Response Selection | `run_dbandits_response.py` | Selects optimal responses from candidates |

## Project Structure

```
Federated-APOHF/
├── Induction/
│   ├── automatic_prompt_engineer/    # Core APE modules
│   │   ├── llm.py                    # LLM interface
│   │   ├── evaluate.py               # Evaluation utilities
│   │   ├── template.py               # Prompt templates
│   │   └── config.py                 # Configuration
│   ├── experiments/
│   │   ├── Federated_neural_bandit.py    # Federated bandit algorithms
│   │   ├── LlamaForMLPRegression.py      # Neural network models
│   │   ├── run_dbandits_po.py            # Prompt optimization
│   │   ├── run_dbandits_image_gen.py     # Image generation
│   │   ├── run_dbandits_response.py      # Response selection
│   │   ├── configs/                      # Configuration files
│   │   ├── data/                         # Dataset
│   │   └── evaluation/                   # Evaluation scripts
│   ├── results/                      # Experiment results
│   └── environment.yml               # Conda environment
└── README.md
```

## Installation

### 1. Create Conda Environment

```bash
cd Induction
conda env create -f environment.yml
conda activate <env_name>
```

### 2. Prepare Data

Download the instruction induction data from [INSTINCT](https://github.com/xqlin98/INSTINCT/tree/main/Induction/experiments/data/instruction_induction) and place it under:

```
Induction/experiments/data/instruction_induction/
```

### 3. Configure API Key

Set your OpenAI API key in the experiment files:

```python
# In run_dbandits_po.py and run_dbandits_image_gen.py
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```

## Usage

### Prompt Optimization

```bash
cd Induction
bash experiments/run_dbandits_po.sh
```

Or run directly with Python:

```bash
python experiments/run_dbandits_po.py --task <task_name> --n_domain 500 --trial 0
```

### Image Generation

```bash
cd Induction
bash experiments/run_dbandits_image_gen.sh
```

### Response Selection

```bash
cd Induction
bash experiments/run_dbandits_response.sh
```

## Supported Tasks

The framework supports various instruction induction tasks:

- `antonyms` - Find antonyms of words
- `synonyms` - Find synonyms of words
- `sum` - Calculate sum of numbers
- `active_to_passive` - Convert active voice to passive
- `auto_categorization` - Categorize items
- `larger_animal` - Compare animal sizes
- `orthography_starts_with` - Find words starting with a letter
- And more...

## Algorithms

The project implements several federated bandit algorithms:

- **DoubleTS** - Double Thompson Sampling
- **LinearDBDiag** - Linear Diagonal Bandit
- **NeuralDBDiag** - Neural Diagonal Bandit

## Results

Experiment results are saved to `Induction/results/` in JSON format, including:
- Best instructions found per iteration
- Performance scores
- Training history

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{lin2024prompt,
  title={Prompt Optimization with Human Feedback},
  author={Lin, Xiaoqiang and Dai, Zhongxiang and Verma, Arun and Ng, See-Kiong and Jaillet, Patrick and Low, Bryan Kian Hsiang},
  journal={arXiv preprint arXiv:2405.17346},
  year={2024}
}
```

## Acknowledgments

This repo is based on the codebase of [INSTINCT](https://github.com/xqlin98/INSTINCT).

## License

Please refer to the original repositories for licensing information.
