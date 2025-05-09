#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qiime2-amplicon-2024.10

# Set variables
FASTQ_DIR="raw_data"
METADATA="metadata.tsv"
OUTDIR="QIIME_results"
CLASSIFIER="classifier-V4.qza"
mkdir -p "$OUTDIR"
MANIFEST="manifest.tsv"

# 3. Import into QIIME 2
qiime tools import \
  --type 'SampleData[SequencesWithQuality]' \
  --input-path "$MANIFEST" \
  --output-path "$OUTDIR/demux.qza" \
  --input-format SingleEndFastqManifestPhred33V2
echo Import into QIIME 2 complete

# 4. Summarize demux
qiime demux summarize \
  --i-data "$OUTDIR/demux.qza" \
  --o-visualization "$OUTDIR/demux.qzv"
echo Summarize demux complete

# 5. DADA2 denoise (single-end)
qiime dada2 denoise-single \
  --i-demultiplexed-seqs "$OUTDIR/demux.qza" \
  --p-trunc-len 0 \
  --o-table "$OUTDIR/table.qza" \
  --o-representative-sequences "$OUTDIR/rep-seqs.qza" \
  --o-denoising-stats "$OUTDIR/denoising-stats.qza" \
  --p-n-threads 0
echo DADA2 denoise complete

# 6. Feature‑table summary
qiime feature-table summarize \
  --i-table "$OUTDIR/table.qza" \
  --o-visualization "$OUTDIR/table.qzv" \
  --m-sample-metadata-file "$METADATA"
echo Feature‑table summary complete

# 7. Taxonomy assignment
qiime feature-classifier classify-sklearn \
  --i-classifier "$CLASSIFIER" \
  --i-reads "$OUTDIR/rep-seqs.qza" \
  --o-classification "$OUTDIR/taxonomy.qza"
echo Taxonomy assignment complete

# 8. Taxa barplot
qiime taxa barplot \
  --i-table "$OUTDIR/table.qza" \
  --i-taxonomy "$OUTDIR/taxonomy.qza" \
  --m-metadata-file "$METADATA" \
  --o-visualization "$OUTDIR/taxa-barplot.qzv"
echo Taxa barplot complete

# 9. Export feature table to CSV
qiime tools export \
  --input-path "$OUTDIR/table.qza" \
  --output-path "$OUTDIR/exported-feature-table"

biom convert \
  -i "$OUTDIR/exported-feature-table/feature-table.biom" \
  -o "$OUTDIR/feature-table.tsv" \
  --to-tsv

tail -n +2 "$OUTDIR/feature-table.tsv" | sed 's/\t/,/g' > "$OUTDIR/feature-table.csv"
echo All done!