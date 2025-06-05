"""
MultiCardioNER evaluation library main script.
Heavily based on the MedProcNER library, which is in turn based on the DisTEMIST and MEDDOPLACE evaluation scripts.
@author: salva
"""

import csv
import os

import pandas as pd

from datetime import datetime
from argparse import ArgumentParser

import utils

def main(argv=None):
    """
    Parse options and call the appropriate evaluation scripts
    """

    reference = "../../data/CardioCCC_Dataset/Testing/Italian/cardioccc_it_med.tsv"
    prediction = "../italian_base_medication/inf_results_on_it_testset_medication.tsv"
    task = "it_med"
    output = "../italian_base_medication/"

    # Parse options
    parser = ArgumentParser()
    parser.add_argument("-r", "--reference", dest="reference",
                        help=".TSV file with Gold Standard or reference annotations", default=reference)
    parser.add_argument("-m", "--mapping", dest="mapping_file",
                        help=".TSV file with the mapping between the test set masked and actual names")
    parser.add_argument("-p", "--prediction", dest="prediction",
                        help=".TSV file with your predictions", default=prediction)
    parser.add_argument("-t", "--task", dest="task",
                        help="Task that you want to evaluate (track1, track2_es, track2_en or track2_it)", default=task)
    parser.add_argument("-o", "--output", dest="output",
                        help="Path to save the scoring results", default=output)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Set to True to print the results for each individual file instead of just the final score")
    args = parser.parse_args(argv)

    # Set output file name
    timedate = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_fname = args.prediction.split('/')[-1][:-4]
    out_file = os.path.join(args.output, 'evalutaion_results_on_{}_{}_{}.txt'.format(args.task, timedate, pred_fname))

    # Read filename mappings and save to dict
    # with open(args.mapping_file, 'r') as f_in:
    #     reader = csv.reader(f_in, delimiter='\t')
    #     mappings = {}
    #     for line in reader:
    #         mappings[line[1].replace('.txt', '')] = line[0].replace('.txt', '')

    # Read gold_standard and predictions
    print("Reading reference and prediction .tsv files")
    df_gs = pd.read_csv(args.reference, sep="\t")
    df_preds = pd.read_csv(args.prediction, sep="\t")

    # Change filenames
    # df_preds['filename'] = df_preds['filename'].map(mappings)

    # Remove any duplicate predictions
    df_preds = df_preds.drop_duplicates(subset=["filename", "label", "start_span", "end_span"]).reset_index(drop=True)  

    # Calculate results
    calculate_ner(df_gs, df_preds, out_file, args.task, args.verbose)


def calculate_ner(df_gs, df_preds, output_path, task, verbose=False):
    print("Computing evaluation scores...")
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text",  "label"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label"]].values.tolist()).to_list()
    scores = utils.calculate_fscore(list_gs_per_doc, list_preds_per_doc, task)
    utils.write_results(task, scores, output_path, verbose)

if __name__ == "__main__":
    main()