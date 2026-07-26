#!/usr/bin/env python3
from pathlib import Path
from shutil import copy

import click
import pandas as pd
import pyterrier as pt
from glob import glob
from tqdm import tqdm
from tirex_tracker import tracking


def get_index(index_directory):
    index_directory = index_directory.resolve().absolute()
    return pt.IndexFactory.of(str(index_directory))


def load_topics(query_variants_file):
    ret = pd.read_json(query_variants_file, lines=True)

    # PyTerrier needs to use pre-tokenized queries
    tokeniser = pt.java.autoclass(
        "org.terrier.indexing.tokenisation.Tokeniser"
    ).getTokeniser()

    ret["query"] = ret["query_reduction"].apply(
        lambda i: " ".join(tokeniser.getTokens(i))
    )
    ret["qid"] = ret["query_id"]

    return ret[["qid", "query"]]


def process_query_variants(query_variants, index_directory, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    try:
        topics = load_topics(query_variants)
    except:
        print("skip " + query_variants)
        return
    
    index = get_index(index_directory)
    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        bm25 = pt.terrier.Retriever(index, wmodel="BM25")

        run = bm25(topics)
        pt.io.write_results(run, output_directory / "run.txt.gz")
        copy(index_directory / "index-ir-metadata.yml",
             output_directory / "index-ir-metadata.yml")


@click.command()
@click.option("--query-predictions", type=str, help="The query-predictions.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(query_predictions, output, index):

    pt.java.init()

    for i in tqdm(glob(query_predictions)):
        out_dir = output / i.split('/')[-1].replace('query-reductions-', '').replace(".jsonl", "")

        process_query_variants(i, index, out_dir)


if __name__ == "__main__":
    main()