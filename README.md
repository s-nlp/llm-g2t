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
