# Trallie - Transfer Learning for Information Extraction

<p align="center">
  <img src="assets/trallie.png" alt="Image description" style="width:250px; height:auto;">
</p>

[![Apache 2.0 License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/school/pischool/)
[![Stars](https://img.shields.io/github/stars/PiSchool/trallie?style=flat&logo=github&cacheSeconds=3600)](https://github.com/PiSchool/trallie/stargazers)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-red?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N3TuH5-IHw9ZQfk4cNflZlCGFbJHTXf6?usp=sharing)

*Trallie (“Transfer learning for information extraction”) boosts Information Extraction (IE) for search among textual asset descriptions by doing away with costly human annotation, instead leveraging LLM capabilities to follow NL guidelines, understand labels, and manipulate NL like it does for code.*

**Problem**: Natural language descriptions of assets and resources are here to stay, both as legacy or as flexible catch-alls. Clustering and categorizing them to run structured search queries traditionally requires information extraction (IE), with some partial solutions offered by RAG and dense embedding matching. This often is bottlenecked by costly human annotation, if only to provide few-shot examples of categories. 

**Ambition**: Trallie brings transfer learning and world understanding afforded by LLM to make information extraction agile. We deliver multilingual, IE-fine-tuned checkpoints of various open model architectures; and for reproducibility, our full fine-tuning recipe including prompt templates.

**Impact**: Transfer learning and natural language input imply impact on legacy and low-resource scenarios, improving discoverability of hidden asset collections, plurality of sources through easier access to search tools, improved trust and privacy.

**Team**: At Pi School, our experience of rapid prototyping in AI, acquired over >100 AI projects, gives us an advantage in exploiting the rapidly moving SOTA.

## Getting Started 
1. Install the "trallie" package. 
```
pip install trallie
```
2. Run the demo script from the repository.
```
python main_pipeline.py
```
3. Alternatively, you can use the [Demo Notebook](https://colab.research.google.com/drive/1N3TuH5-IHw9ZQfk4cNflZlCGFbJHTXf6?usp=sharing) on Google Colab. 

