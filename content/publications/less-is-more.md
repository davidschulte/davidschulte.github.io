---
title: "Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning"
date: 2024-10-19
draft: false
description: "Efficiently search for Intermediate Tasks for Fine-Tuning using Embedding Space Maps"
publication: "EMNLP 2024"
authors: "David Schulte, Felix Hamborg, Alan Akbik"
url_pdf: "https://aclanthology.org/2024.emnlp-main.529.pdf"
url_publication: "https://aclanthology.org/2024.emnlp-main.529/"
url_arxiv: "https://arxiv.org/abs/2410.15148"
url_code: "https://github.com/davidschulte/hf-dataset-selector"
citation: >
 @inproceedings{schulte-etal-2024-less,
    title = "Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning",
    author = "Schulte, David  and
      Hamborg, Felix  and
      Akbik, Alan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.529/",
    doi = "10.18653/v1/2024.emnlp-main.529",
    pages = "9431--9442",
    abstract = "Intermediate task transfer learning can greatly improve model performance. If, for example, one has little training data for emotion detection, first fine-tuning a language model on a sentiment classification dataset may improve performance strongly. But which task to choose for transfer learning? Prior methods producing useful task rankings are infeasible for large source pools, as they require forward passes through all source language models. We overcome this by introducing Embedding Space Maps (ESMs), light-weight neural networks that approximate the effect of fine-tuning a language model. We conduct the largest study on NLP task transferability and task selection with 12k source-target pairs. We find that applying ESMs on a prior method reduces execution time and disk space usage by factors of 10 and 278, respectively, while retaining high selection performance (avg. regret@5 score of 2.95)."
 }
---

Intermediate task transfer learning can greatly improve model performance. If, for example, one has little training data for emotion detection, first fine-tuning a language model on a sentiment classification dataset may improve performance strongly. But which task to choose for transfer learning? Prior methods producing useful task rankings are infeasible for large source pools, as they require forward passes through all source language models. We overcome this by introducing Embedding Space Maps (ESMs), light-weight neural networks that approximate the effect of fine-tuning a language model. We conduct the largest study on NLP task transferability and task selection with 12k source-target pairs. We find that applying ESMs on a prior method reduces execution time and disk space usage by factors of 10 and 278, respectively, while retaining high selection performance (avg. regret@5 score of 2.95).
