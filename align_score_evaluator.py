from tqdm.auto import tqdm
from alignscore import AlignScore
import json
import click

from webnlg_dataset_reader import Benchmark


def calculate_align_score(bench: Benchmark, g2t_predictions: list[str], scorer: AlignScore) -> list[float]:
    scores = []
    for i, entry in enumerate(tqdm(bench.entries)):
        lexs = [l.lex for l in entry.lexs]
        _entry_scores = scorer.score(
            contexts=lexs,
            claims=[g2t_predictions[i]] * len(lexs)
        )
        scores.append(max(_entry_scores))
    return scores


@click.command()
@click.option('--preds_path', type=str, required=True, help='Path to G2T predictions file')
@click.option('--output_path', type=str, required=True, help='Path to save scores JSON file')
def main(preds_path, output_path):
    scorer = AlignScore(
        model='roberta-large',
        batch_size=16,
        device='cuda:0', 
        ckpt_path='./AlignScore-large.ckpt', 
        evaluation_mode='nli_sp',
        verbose=False,
    )

    b = Benchmark()
    b.fill_benchmark([('./', 'rdf-to-text-generation-test-data-with-refs-en.xml')])
    
    with open(preds_path, 'r') as f:
        predictions = [l.replace('\n', '').strip() for l in f.readlines()]

    scores = calculate_align_score(b, predictions, scorer)
    with open(output_path, 'w') as f:
        json.dump(scores, f)


if __name__ == '__main__':
    main()
    