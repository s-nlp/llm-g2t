# G2T LLM Evaluation
Here you can find scripts for LLM evaluation on the WEBNLG-2020 dataset

## How to Use
Run `python llm_evaluator.py --llm=<NAME OF LLM> --dataset_folder=<PATH TO FOLDER WITH WEBNLG DATASET> --dataset_filename=<FILENAME OF WEBNLG DATASET> --output_path=<WHERE TO STORE GENERATED GRAPH DESCRIPTIONS>`
to generate graph descriptions
Supported LLMs are:
* llama3:8b
* gemma2:9b
* gpt-4o
* gpt-4o-mini

Run `python metrics_evaluator.py --preds_path=<PATH TO FILE WITH GRAPH DESCRIPTIONS FROM LLM> --dataset_folder=<PATH TO FOLDER WITH WEBNLG DATASET> --dataset_filename=<FILENAME OF WEBNLG DATASET> --output_path=<WHERE TO STORE DETAILED METRICS>`
to evaluate WEBNLG metrics and `alignscore_evaluator.py` for the AlignScore.

## Future Work, Citation & Contacts

If you find some issues, do not hesitate to add it to [Github Issues](https://github.com/s-nlp/llm-g2t/issues).

For any questions please contact: [Dmitrii Iarosh](mailto:D.Yarosh@skol.tech), [Mikhail Salnikov](mailto:Mikhail.Salnikov@skol.tech) or [Alexander Panchenko](mailto:A.Panchenko@skol.tech)

```bibtex
@inproceedings{iarosh-etal-2025-g2t-hallucinations,
    title = "On Reducing Factual Hallucinations in Graph-to-Text Generation using Large Language Models",
    author = "Iarosh, Dmitrii and
      Salnikov, Mikhail and
      Panchenko, Alexander",
    booktitle = "Proceedings of the COLING 2025 GenAIK 2025 Workshop",
    month = feb,
    year = "2025",
}
```
