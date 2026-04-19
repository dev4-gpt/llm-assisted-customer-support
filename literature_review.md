# Literature Review: LLM-Augmented Customer Support Triage and Quality Monitoring

Customer support automation is fundamentally an NLP problem in which free-form text must be transformed into reliable operational decisions. The literature shows progress across intent detection, intent discovery, multi-intent classification, and summarization, but most work remains component-specific rather than integrated end-to-end [1]-[6]. To align with project requirements, each study below is summarized with five required elements: citation, pursued work, tools and techniques (including architecture), reported performance/findings, and what we learned for implementation.

### Study 1: Semi-supervised multi-task intent classification
**Citation:** Dong et al., *A Semi-supervised Multi-task Learning Approach to Classify Customer Contact Intents* [1]. **What was pursued:** the authors target customer-contact intent classification in realistic settings with noisy and incomplete labels. **Tools and techniques / architecture:** ALBERT-based transformer encoder with a semi-supervised multi-task setup using unlabeled data and auxiliary objectives. **Reported performance and findings:** the paper reports strong gains over supervised baselines, including notable AUC-ROC improvements and better robustness on difficult intent categories. **What we learned:** our project should not rely on only a single supervised model; we should include a semi-supervised training path for low-frequency and ambiguous intents.

### Study 2: Intent mining from historical conversations
**Citation:** Chatterjee and Sengupta, *Intent Mining from past conversations for Conversational Agent* [2]. **What was pursued:** discovering latent intent structure from historical dialogue logs before defining final intent labels. **Tools and techniques / architecture:** Universal Sentence Encoder embeddings combined with iterative density-based clustering (ITER-DBSCAN style pipeline). **Reported performance and findings:** the method improves clustering quality relative to simpler clustering baselines and reveals hidden intent groupings not captured by manual taxonomy. **What we learned:** we should add an offline intent-discovery stage on archived tickets prior to supervised label finalization.

### Study 3: Intent detection at scale
**Citation:** *Intent Detection at Scale: Tuning a Generic Model using Relevant Intents* [3]. **What was pursued:** scalable intent detection where model focus is narrowed to relevant intent subsets. **Tools and techniques / architecture:** transfer learning pipeline with relevant-intent filtering followed by targeted fine-tuning. **Reported performance and findings:** better precision/recall/F1 trade-offs are reported compared with naive full-taxonomy modeling, especially as the number of intents grows. **What we learned:** our system should use a two-stage architecture: coarse intent-family router first, then fine-grained intent specialist.

### Study 4: Multi-intent detection for support tickets
**Citation:** *Multi-Intent Detection in Customer Support Queries Using AI* [4]. **What was pursued:** handling tickets that contain multiple concurrent intents in one message. **Tools and techniques / architecture:** multi-label intent classification with threshold calibration instead of single-label argmax prediction. **Reported performance and findings:** multi-label evaluation metrics (e.g., micro/macro F1 and subset-style accuracy) show improved handling of compound queries. **What we learned:** our API schema should support `intents: list[str]` and confidence per intent, not only one category label.

### Study 5: Customer-service dialogue summarization benchmark
**Citation:** Feigenblat et al., *TWEETSUMM: A Dialog Summarization Dataset for Customer Service* [5]. **What was pursued:** benchmarking customer-service dialogue summarization with realistic conversation structure. **Tools and techniques / architecture:** dataset-driven evaluation across extractive methods and abstractive transformer summarizers (including BART-style architectures). **Reported performance and findings:** domain-tuned neural summarizers outperform classical baselines on ROUGE-type metrics, but the paper also shows that automatic metrics alone do not fully capture summary usefulness. **What we learned:** our project should compare an extractive baseline (e.g., TextRank) against an abstractive model (e.g., T5/BART/LLM prompting) and include human judgment criteria.

### Study 6: Request classification in software customer service
**Citation:** Arias-Barahona et al., *Requests classification in the customer service area for software companies using machine learning and natural language processing* [6]. **What was pursued:** practical classification of software-support requests for routing and triage. **Tools and techniques / architecture:** classical ML pipeline (including SVM) with preprocessing, feature engineering, and class balancing. **Reported performance and findings:** the authors report very high classification accuracy in their setting and show that preprocessing/class rebalancing strongly affects results. **What we learned:** data quality, normalization, and imbalance handling should be explicit pipeline stages, not ad-hoc preprocessing.

Across these studies, three synthesis points are consistent. First, data-centric strategies (semi-supervision, weak supervision, and taxonomy refinement) are as important as model choice [1], [2]. Second, production-grade support systems require architectural composition across tasks (intent, multi-intent, summarization, and quality) rather than isolated models [3]-[5]. Third, evaluation must go beyond single automatic scores and include human-grounded checks for utility and reliability [5], [6].

The practical research gap is therefore integration and transparency. Existing work provides strong components, but fewer open implementations connect triage, summarization, and quality monitoring in one auditable workflow. Our project addresses this by implementing a unified pipeline that maps raw support text to structured triage outputs, summary artifacts, and quality-scoring signals, with explicit reporting of both model metrics and human evaluation outcomes.

## References

[1] X. Dong et al., "A Semi-supervised Multi-task Learning Approach to Classify Customer Contact Intents," in *Proceedings of the 4th Financial Narrative Processing Workshop (FNP 2021) / ECNLP 2021*. Available: https://aclanthology.org/2021.ecnlp-1.7.pdf

[2] S. Chatterjee and A. Sengupta, "Intent Mining from past conversations for Conversational Agent," in *Proceedings of COLING 2020*. Available: https://www.aclweb.org/anthology/2020.coling-main.366.pdf

[3] "Intent Detection at Scale: Tuning a Generic Model using Relevant Intents," *arXiv preprint*, 2023. Available: https://arxiv.org/pdf/2309.08647.pdf

[4] "Multi-Intent Detection in Customer Support Queries Using AI," *IEEE Xplore*, 2025. Available: https://ieeexplore.ieee.org/document/11210877/

[5] G. Feigenblat et al., "TWEETSUMM: A Dialog Summarization Dataset for Customer Service," *arXiv preprint*, 2021. Available: https://arxiv.org/pdf/2111.11894.pdf

[6] M. Arias-Barahona et al., "Requests classification in the customer service area for software companies using machine learning and natural language processing," *PeerJ Computer Science*, 2023. Available: https://peerj.com/articles/cs-1016


