# sTarML: A Sequence-Based Machine Learning Framework for Predicting Bacterial sRNA-mRNA Interactions

sTarML prepares RefSeq genome + GTF for a given NCBI Taxonomy ID, builds TSS-window regions near start codons, and predicts sRNA targets using bundled k-mer vectorizers/scalers and a trained model.


---

## Install

```bash
# Install the package
conda install -c vkotsira starml

conda env create -f environment.yml
conda activate starml
```

---

## Quick start

**Comma-separated sequences**
```bash
starml --taxid 511145 --srna "ACGTT,GGGAT" --outdir results --cores 4
```

**FASTA of sRNAs**
```bash
starml --taxid 511145 --srna-fasta my_sRNAs.fa --outdir results --cores 8
```

**Specific RefSeq accession**
```bash
starml --taxid 511145 --refseq GCF_025643435.1 --srna "ACGTT" --outdir results
```

> Run `starml --help` for all options.

---

## What it does (7 steps)

1. Initialize & logging → `logs/run.log`  
2. Download genome + GTF (RefSeq) via `ncbi-datasets-cli`  
3. Filter GTF to protein-coding mRNAs  
4. Create TSS-window BED and (if `bedtools`) extract FASTA  
5. Normalize sRNAs (A/C/G/T/N; `U→T`) and save FASTA  
6. Vectorize/scale, run bundled model → `predictions.tsv`  
7. Finish

---

## Outputs (in `--outdir`)

- `tss_windows.bed` (and `tss_windows.fasta` if `bedtools` present)  
- `srna/srnas.fasta`  
- `predictions.tsv`  
- `logs/run.log`

---

## Notes / troubleshooting

- `--srna` and `--srna-fasta` are **mutually exclusive**.  
- If it stops after Step 5, you likely passed a filename to `--srna`; use `--srna-fasta file.fa`.  
- “No assemblies found” → check `--taxid` or specify a valid `--refseq GCF_...`.

---

## Citation

A paper is **forthcoming**. Until then, please cite the software:

> Vasiliki Kotsira et al.. *sTarML: A Sequence-Based Machine Learning Framework for Predicting Bacterial sRNA-mRNA Interactions*. Version 0.1.0.

```bibtex
@software{starml_software,
  author  = {Vasiliki Kotsira},
  title   = {sTarML: A Sequence-Based Machine Learning Framework for Predicting Bacterial sRNA-mRNA Interactions},
  year    = {2025},
  version = {0.1.0}
}
```

We’ll update this section (and `CITATION.cff`) with the final paper citation and DOI when available.

---

## License

- **Code:** MIT (see `LICENSE`)  
- **Model files (`starml/models/*`):** CC BY 4.0 (see `LICENSE-models`)
