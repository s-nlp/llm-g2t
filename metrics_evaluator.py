import json
import statistics

import click
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import meteor_score
from sacrebleu import TER

from webnlg_dataset_reader import Benchmark

ter = TER()

metric_names = ["meteor", "bleu", "chrf", "ter", "bert"]


def fill_metrics_zero(metrics):
    for metric_name in metric_names:
        metrics[metric_name].append(-1)


def eval_metrics(metrics, result, references, item_index=None):
    references_splited = list(map(lambda x: x.split(" "), references))
    result_splited = result.split(" ")
    score_meteor = meteor_score(
        references_splited,
        result_splited
    )
    metrics["meteor"][item_index] = score_meteor

    score_bleu = sentence_bleu(
        references_splited,
        result_splited
    )
    metrics["bleu"][item_index] = score_bleu

    score_chrf = sentence_chrf(
        references_splited[0],
        result_splited
    )
    metrics["chrf"][item_index] = score_chrf

    score_ter = ter.sentence_score(result, references).score
    metrics["ter"][item_index] = score_ter

    score_bert_precision, score_bert_recall, score_bert_f1 = score([result], [references], lang="en")
    metrics["bert"][item_index] = score_bert_f1.detach().item()


@click.command()
@click.option('--preds_path', type=str, required=True, help='Path to G2T predictions file')
@click.option('--dataset_folder', type=str, required=True, help='Path to WEBNLG dataset folder')
@click.option('--dataset_filename', type=str, required=True, help='WEBNLG dataset filename')
@click.option('--output_path', type=str, required=True, help='Path to save scores JSON file')
def main(preds_path, dataset_folder, dataset_filename, output_path):
    b = Benchmark()
    metrics = {
        "meteor": [],
        "bleu": [],
        "chrf": [],
        "ter": [],
        "bert": []
    }
    b.fill_benchmark([(dataset_folder, dataset_filename)])
    with open(preds_path, 'r') as prediction_file:
        metrics_per_line = []
        predictions = list(prediction_file.readlines())
        for index, item in enumerate(predictions):
            print(f"Processing line {index + 1}/{len(predictions)}")
            fill_metrics_zero(metrics)
            entry = b.entries[index]
            references = list(map(lambda x: x.lex, entry.lexs))
            eval_metrics(metrics, item, references, item_index=index)
            metrics_per_line.append({
                metric: metrics[metric][index] for metric in metric_names
            })
            metrics_per_line[-1]["index"] = index
            metrics_per_line[-1]["answer"] = item

            for metric_name in metric_names:
                metric_raw_values = list(filter(lambda x: x != -1, metrics[metric_name]))
                mean_metric = statistics.mean(metric_raw_values)
                print(f"{metric_name}_mean: {mean_metric}")

        with open(output_path, 'w') as f3:
            json.dump(metrics_per_line, f3)


if __name__ == "__main__":
    main()
