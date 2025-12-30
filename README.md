# PromptGenome

**PromptGenome** is a multi-agent AI system designed to automatically optimize prompts for large language models.  
It treats prompts like evolving genomes, applying controlled mutations, evaluating output quality, and selecting the best variants in iterative loops.

In modern AI systems, prompt performance can be brittle and sensitive to phrasing, instructions, and task context. PromptGenome automates this process by orchestrating multiple agents:

- **Generator Agent:** Creates baseline prompts based on task definitions.  
- **Mutation Agent:** Applies structured transformations to explore the prompt space.  
- **Execution Agent:** Runs prompts against evaluation datasets to generate outputs.  
- **Evaluation Agent:** Scores outputs using objective metrics, including accuracy, format correctness, and hallucination rates.  
- **Selection Agent:** Chooses the best prompts while preserving diversity and preventing regressions.

PromptGenome provides:
- Visual evolution of prompts over generations  
- Evaluation metrics and lineage tracking  
- A fully automated loop for iterative prompt optimization



