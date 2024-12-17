# AI-Based Code Vulnerability Detection Using Pairwise Learning to Rank

This project aims at detecting code vulnerabilities using the pairwise learning to rank method. 

## Description

### Goal:
The goal of this project is to 



### How it works:

1) Load pairs of vulnerable / benign functions from the [PrimeVul](https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK) paired dataset.



## Built With

- [Python (3.12)](https://www.python.org/)

### Dependencies

<!-- - [llama-cpp-python (0.3.2)](https://pypi.org/project/llama-cpp-python/) -->

<!-- - [langchain-huggingface (0.1.2)](https://python.langchain.com/docs/integrations/providers/huggingface/) -->
<!-- - [langchain_chroma (0.1.4)](https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html) -->
<!-- - [langchain_text_splitters (0.3.2)](https://api.python.langchain.com/en/latest/text_splitters_api_reference.html) -->

<!-- - [accelerate (1.1.1)](https://pypi.org/project/accelerate/) -->
- [transformers (4.46.3)](https://pypi.org/project/transformers/)
- [Pytorch (2.5.1)](https://pypi.org/project/torch/)
- [scikit-learn (1.5.2)](https://pypi.org/project/scikit-learn/)
- [numpy]
- [wandb (0.19.1)](): login with ```wandb login```
<!-- - [tensorboard (2.18.0)](): activate tensorboard with: ```tensorboard --logdir=runs``` -->
- For BigBird:
    - [tiktoken (0.8.0)](https://pypi.org/project/tiktoken/)
    - [sentencepiece (0.2.0)](https://pypi.org/project/sentencepiece/)
    - [protobuf (5.29.1)](https://protobuf.dev/getting-started/pythontutorial/)
<!-- - [pandas (2.2.3)](https://pandas.pydata.org/) -->
<!-- - [tqm (4.67.1)](https://tqdm.github.io/) -->


## Project Set-up

### Install Dependencies

To create a conda environment:
```console
conda env create -f environment.yml
```

In case of issues, please refer to the list of dependencies [above](#dependencies).

### Prerequisites

Prerequisites before running the project:

1. Download the [PrimeVul](https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK) paired dataset:
    - primevul_train_paired.jsonl
    - primevul_test_paired.jsonl
    - primevul_valid_paired.jsonl

## Run the project

1. Make sure to install all [dependencies](#install-dependencies) and follow the [prerequisite steps](#prerequisites).

2. Execute ```huggingface-cli login``` to sign in to your Hugging Face account to download models (one can refer to the [documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli)).

3. Set project location in ```main.py```, to be able to call project modules: ```sys.path.append("path/to/project/ai-vul-detect-pairwise-learning-to-rank")```.

4. Execute ```python main.py``` with all necessary arguments described in the file. 

    (Note: change log level from INFO to DEBUG in settings.py).

<!-- 5. Extract generated vulnerable dataset from the ```output/``` folder. -->


## Project Architecture
- ```src/load_data_pairs.py```: Load pairs of vulnerable and benign code dataset.

- ```src/settings.py```: Project settings, namely the logger.

- ```src/main.py```: Parse arguments, load data, train the model, evaluate results..

## Tested on:

- Ubuntu 20.04.3 LTS
- NVIDIA A10 (CUDA Version: 12.7)
- AMD EPYC 7282 16-Core Processor
