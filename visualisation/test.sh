python encode_multitrack_plot.py \
      --chrom chr8 --start 127500000 --end 128000000 \
      --bin-size 10000 --region-name "MYC locus" \
      --out figures/myc_10kb.png \
      --fig-width 22

python encode_multitrack_plot.py \
    --chrom chr8 --start 126500000 --end 129000000 \
    --bin-size 25000 --region-name "MYC TAD" \
    --out figures/myc_tad.png \
    --fig-width 22

python encode_multitrack_plot.py \
    --chrom chr7 --start 27100000 --end 27500000 \
    --bin-size 5000 --region-name "HOXA cluster" \
    --out figures/hoxa.png \
    --fig-width 22

python encode_multitrack_plot.py \
--chrom chr8 \
--x-start 127600000 \
--x-end   127800000 \
--y-start 127800000 \
--y-end   128000000 \
--bin-size 5000 \
--region-name "MYC loop overview" \
--out figures/myc_loop_overview.png \
--fig-width 22

python encode_multitrack_plot.py \
--chrom chr8 \
--x-start 127700000 \
--x-end   127710000 \
--y-start 127900000 \
--y-end   127910000 \
--bin-size 1000 \
--region-name "MYC loop anchor sequence view" \
--out figures/myc_loop_sequence.png \
--fig-width 22
