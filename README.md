# Causally Modeling the Linguistic and Social Factors that Predict Email Response

This repository contains the code and resources for our NAACL 2025 paper: **[Causally Modeling the Linguistic and Social Factors that Predict Email Response](https://aclanthology.org/2025.naacl-long.594/)**. In this study, we introduce SIZZLER, a dataset of 1,800 emails annotated with email intent and expectation labels. Using this dataset, we benchmark various models, including feature-based logistic regression and zero-shot LLMs, to analyze how linguistic and social factors influence email response behaviors. Our analysis spans over 11.3 million emails from the GMANE archive. We provide insights into the causal effects of social status, argumentation, and social connections on whether an email receives a reply.

## Project Structure

- `data/`: Processed datasets and annotation files  
- `notebooks/`: Jupyter notebooks for exploratory analysis  
- `src/`: All code scripts  
  - `src/data/`: Data loading and preprocessing  
  - `src/features/`: Feature extraction  
  - `src/models/`: Modeling and evaluation  
  - `src/visualization/`: Plotting and figure generation  
- `requirements.txt`: Python dependencies  
- `README.md`: Project description and usage

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{xu-etal-2025-causally,
    title = "Causally Modeling the Linguistic and Social Factors that Predict Email Response",
    author = "Xu, Yinuo  and
      Chen, Hong  and
      Rakshit, Sushrita  and
      Ananthasubramaniam, Aparna  and
      Yadav, Omkar  and
      Zheng, Mingqian  and
      Jiang, Michael  and
      Zhang, Lechen  and
      Yi, Bowen  and
      Alkiek, Kenan  and
      Israeli, Abraham  and
      Shu, Bangzhao  and
      Shen, Hua  and
      Pei, Jiaxin  and
      Zhang, Haotian  and
      Schirmer, Miriam  and
      Jurgens, David",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.594/",
    pages = "11842--11866",
    ISBN = "979-8-89176-189-6",
    abstract = "Email is a vital conduit for human communication across businesses, organizations, and broader societal contexts. In this study, we aim to model the intents, expectations, and responsiveness in email exchanges. To this end, we release SIZZLER, a new dataset containing 1800 emails annotated with nuanced types of intents and expectations. We benchmark models ranging from feature-based logistic regression to zero-shot prompting of large language models. Leveraging the predictive model for intent, expectations, and 14 other features, we analyze 11.3M emails from GMANE to study how linguistic and social factors influence the conversational dynamics in email exchanges. Through our causal analysis, we find that the email response rates are influenced by social status, argumentation, and in certain limited contexts, the strength of social connection."
}
