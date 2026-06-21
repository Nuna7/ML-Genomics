import pandas as pd
from src.genomic_io import hic_matrix

manifest = pd.read_csv("manifest.csv")

n_zero = 0
n_total = len(manifest)

for _, row in manifest.iterrows():
    mat, _ = hic_matrix(
        "../encode_work/raw/k562_hic.hic",
        row.chrom,
        int(row.start),
        int(row.end),
        int(row.bin_size),
    )
    if mat.sum() == 0:
        n_zero += 1

print(f"{n_zero}/{n_total} ({100*n_zero/n_total:.2f}%)")