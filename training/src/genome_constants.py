"""
genome_constants.py
--------------------
Pure constants, zero external dependencies. Kept separate from genomic_io.py
on purpose: scripts that only need chromosome lengths (like make_windows.py)
should not be forced to have pyBigWig / hic-straw installed just to import
this.
"""
from __future__ import annotations


def chrom_sizes_hg38() -> dict:
    """
    Hardcoded GRCh38 primary assembly chromosome sizes (in bp).
    Source: GRCh38 reference (UCSC hg38 / GCA_000001405).
    """
    return {
        "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559,
        "chr4": 190_214_555, "chr5": 181_538_259, "chr6": 170_805_979,
        "chr7": 159_345_973, "chr8": 145_138_636, "chr9": 138_394_717,
        "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
        "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
        "chr16": 90_338_345, "chr17": 83_257_441, "chr18": 80_373_285,
        "chr19": 58_617_616, "chr20": 64_444_167, "chr21": 46_709_983,
        "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
    }