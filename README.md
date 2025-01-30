# Sentence Generalized Pooling

This library implements generalized pooling methods for sentence embeddings, based on the research paper ["Enhancing Sentence Embedding with Generalized Pooling"](https://aclanthology.org/C18-1154.pdf) by Qian Chen, Zhen-Hua Ling, and Xiaodan Zhu (COLING 2018).

## Installation

```bash
pip install sentence-generalized-pooling
```

## Usage

```python
from sentence_transformers import SentenceTransformer
from sentence_generalized_pooling import GeneralizedSentenceTransformerMaker

# Load a base model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a model with generalized pooling
maker = GeneralizedSentenceTransformerMaker(
    model_name=base_model,
    pooling_type=0,  # ADDITIVE pooling
    initalize=0,     # MEAN initialization
    device='cpu'
)

# Get the modified model
model = maker.get_model()

# Use the model
embeddings = model.encode(['Your sentence here'])
```

## Features

- Supports both additive and dot-product pooling mechanisms
- Multiple initialization strategies: mean, noised, and random
- Multi-head attention mechanism
- Compatible with all sentence-transformer models

## Citation

If you use this library in your research, please cite:

```bibtex
@inproceedings{chen-etal-2018-enhancing,
    title = "Enhancing Sentence Embedding with Generalized Pooling",
    author = "Chen, Qian  and
      Ling, Zhen-Hua  and
      Zhu, Xiaodan",
    booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
    year = "2018",
    url = "https://aclanthology.org/C18-1154",
    pages = "1815--1826",
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.