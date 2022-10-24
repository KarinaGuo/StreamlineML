import os, csv
import pandas as pd
import argparse
import itertools

parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str)
parser.add_argument("metadata", help="File of metadata to merge", type=str)
parser.add_argument("--fields", help="Fields to join from metadata", action="append", type=str)
args = parser.parse_args()

raw_out = pd.read_csv(os.path.join(args.base_dir, "final_results.csv"))
joined_out = os.path.join(args.base_dir, "joined_final_results.csv")
metadf = pd.read_csv(args.metadata)
target_fields = list(itertools.chain(args.fields))

raw_out['id']=raw_out['id'].str.strip()
metadf_subset = metadf[metadf.columns.intersection(args.fields)]

joined_results = pd.merge(raw_out,metadf_subset, on='id', how='left')
joined_results_rmna = joined_results.dropna()
joined_results_rmna.to_csv(joined_out, sep=',', index=False)
