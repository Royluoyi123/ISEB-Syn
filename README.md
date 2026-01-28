
## Prerequisites

Ensure you have the following environment configuration to maintain compatibility with the model's architecture:

* **Python:** 3.7.10
* **PyTorch:** 1.8.1
* **torch-geometric:** 1.7.0

## Getting Started

### 1. Data Generation
Before running the prediction, you need to preprocess the raw biological datasets (such as scRNA-seq or spatial transcriptomics) into a format suitable for the model.

Run the following command to generate the necessary data files:
```bash
python data.py
python main.py
