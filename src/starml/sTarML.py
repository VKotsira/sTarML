#!/usr/bin/env python3
"""
sTarML — sRNA Target prediction with Machine Learning
=====================================================================

This script prepares inputs for sTarML:
  1) Accepts sRNA sequences (A,C,G,T,N) from `--srna` (comma-separated) or `--srna-fasta`.
     If any `U` is found, it is automatically converted to `T` and noted in the log.
  2) Accepts an NCBI Taxonomy ID (e.g. 511145 for *E. coli* K-12 MG1655).
  3) Downloads genome FASTA and GTF annotation from RefSeq using NCBI Datasets.
     By default, the first assembly listed for the given TaxID is used (representative genome);
     alternatively, provide `--refseq GCF_...` to download that specific RefSeq accession.
  4) Cleans the annotation to keep only protein-coding mRNAs.
  5) Logs every step to `logs/run.log` (errors included).
  6) ALWAYS runs predictions using the bundled k-mer vectorizers/scalers and model in `models/`.

Dependencies:
  - ncbi-datasets-cli (command line tool): conda install -c conda-forge ncbi-datasets-cli
  - python packages: pandas, biopython, joblib, scikit-learn, scipy, xgboost

Usage:
  # Prints step messages to screen by default:
  python sTarML.py --taxid 511145 --srna "ACGTT,GGGAT" --outdir results --cores 4
  python sTarML.py --taxid 511145 --srna-fasta my_sRNAs.fa --outdir results --cores 8

  # Download a specific RefSeq accession instead of the first catalog entry:
  python sTarML.py --taxid 511145 --refseq GCF_025643435.1 --srna "ACGTT" --outdir results
"""

# Input libraries
import re
import os
import sys
import json
import time
import shutil
import zipfile
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
from Bio.SeqRecord import SeqRecord
# Prediction deps
import joblib
from scipy import sparse as sp


# ---- Config: expected filenames under ./models ----
MODELS_DIRNAME = "models"
SRNA_VEC_FILE = "srna_kmer_vectorizer.pkl"
MRNA_VEC_FILE = "mrna_kmer_vectorizer.pkl"
SRNA_SCL_FILE = "srna_kmer_scaler.pkl"
MRNA_SCL_FILE = "mrna_kmer_scaler.pkl"
MODEL_FILE    = "sTarML_model.pkl"


# Running from command line
def run_cmd(cmd, log):
    log.info("Running command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s", e)
        sys.exit(1)
    return

# Capture stdout
def run_cmd_capture(cmd, log):
    log.info("Running command (capture): %s", " ".join(cmd))
    try:
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return out.stdout
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s", e.stdout or e)
        sys.exit(1)
    return

# Create output directory
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

# Prints
def step(msg, log, print_steps=False):
    if print_steps:
        print(msg, flush=True)
    log.info(msg)
    return

# Download genome and annotation
def download_genome_and_gtf(taxid, outdir, log, print_steps=False, refseq=None, total_steps=7):
    """
    Download genome + GTF from RefSeq.

    If `refseq` (GCF_...) is provided: download that exact accession.
    Else: select the FIRST assembly from the NCBI catalog (summary JSON lines).
    """
    extract_dir = outdir / "ncbi_dataset"
    ensure_dir(extract_dir)

    if refseq:
        step(f"[sTarML] Step 2/{total_steps}: Downloading provided accession {refseq}", log, print_steps)
        zip_path = outdir / f"{refseq}.zip"
        cmd = [
            "datasets", "download", "genome", "accession", refseq,
            "--assembly-source", "refseq",
            "--include", "genome,gtf",
            "--filename", str(zip_path),
            "--no-progressbar"
        ]
        run_cmd(cmd, log)
    else:
        # 1) summarize genomes for the taxon (JSON lines); take the first non-empty line
        step(f"[sTarML] Step 2a/{total_steps}: Selecting first assembly from NCBI catalog (representative genome)", log, print_steps)
        summary_out = run_cmd_capture(
            ["datasets", "summary", "genome", "taxon", str(taxid), "--as-json-lines"],
            log
        )
        first_line = None
        for ln in summary_out.splitlines():
            ln = ln.strip()
            if ln:
                first_line = ln
                break
        if not first_line:
            log.error("No assemblies found in datasets summary for this taxon.")
            sys.exit(1)

        # Parse first JSON line and extract first GCF_ accession
        try:
            _ = json.loads(first_line)  # validate JSON
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse first JSON line from summary: {e}")
            sys.exit(1)
        m = re.search(r"GCF_\d+\.\d+", first_line)
        if not m:
            log.error("Could not find a RefSeq GCF accession in the first catalog entry.")
            sys.exit(1)
        accession = m.group(0)
        log.info("Selected accession from first catalog entry: %s", accession)

        # Download ONLY that accession’s genome + GTF
        step(f"[sTarML] Step 2b/{total_steps}: Downloading selected accession package", log, print_steps)
        zip_path = outdir / f"{accession}.zip"
        cmd = [
            "datasets", "download", "genome", "accession", accession,
            "--assembly-source", "refseq",
            "--include", "genome,gtf",
            "--filename", str(zip_path),
            "--no-progressbar"
        ]
        run_cmd(cmd, log)

    # extract the package
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    fasta_files = list(extract_dir.rglob("*_genomic.fna"))
    gtf_files = list(extract_dir.rglob("*.gtf"))
    if not fasta_files:
        log.error("No genome FASTA found in the downloaded package.")
        sys.exit(1)
    if not gtf_files:
        log.error("No GTF found in the downloaded package.")
        sys.exit(1)
    return fasta_files[0], gtf_files[0]

# Parse and filter GTF
def filter_gtf(gtf_file, out_file, log):
    rows = []
    with open(gtf_file) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            attributes = parts[8]
            if feature in ["transcript", "mRNA", "gene"] and "protein_coding" in attributes:
                rows.append(line)
    with open(out_file, "w") as out:
        for row in rows:
            out.write(row)
    log.info("Saved cleaned GTF with %d entries", len(rows))
    return

# Handle sRNA input
def normalize_srna(seq, log):
    s = seq.strip().upper()
    # Accept DNA alphabet (A,C,G,T,N). If U is present, convert to T and note it.
    if "U" in s:
        log.info("Found 'U' in sRNA; converting U->T for DNA alphabet consistency.")
        s = s.replace("U", "T")
    allowed = set("ACGTN")
    bad = sorted({ch for ch in s if ch not in allowed})
    if bad:
        log.error(f"Invalid characters {bad} in sRNA sequence. Allowed: A,C,G,T,N.")
        sys.exit(1)
    return s

# Loading the srna
def load_srna(srna_csv, srna_fasta, outdir, log):
    records = []
    if srna_csv:
        for i, seq in enumerate(srna_csv.split(",")):
            norm = normalize_srna(seq, log)
            records.append(SeqRecord(Seq(norm), id=f"sRNA_{i+1}", description="cli"))
    if srna_fasta:
        for rec in SeqIO.parse(str(srna_fasta), "fasta"):
            norm = normalize_srna(str(rec.seq), log)
            records.append(SeqRecord(Seq(norm), id=rec.id, description=rec.description))
    if not records:
        log.error("No sRNA sequences provided.")
        sys.exit(1)
    srna_file = outdir / "srnas.fasta"
    SeqIO.write(records, str(srna_file), "fasta")
    log.info("Saved %d sRNAs to %s", len(records), srna_file)
    return

# Create TSS-targeting BED and FASTA via bedtools
def create_tss_bed(genome_fna_path, cleaned_gtf_path, bed_path, fasta_out_path, log, print_steps=False, total_steps=7):
    """
    Build a BED with windows around the start codon region:
      + strand: [start-200, start+99]
      - strand: [end-99,   end+200]
    Clamp to chromosome bounds (1..chrom_len) and ensure BED start is 0-based.
    If 'bedtools' exists, also write FASTA for these windows.

    NOTE: The BED 'name' field includes both gene_id and gene_name: gene_id|gene_name
          (falls back to gene_id if gene_name is absent).
    """
    step(f"[sTarML] Step 4/{total_steps}: Creating TSS-window BED", log, print_steps)

    # Chromosome lengths
    chrom_lengths = []
    for rec in SeqIO.parse(str(genome_fna_path), "fasta"):
        name = rec.id.split()[0]
        chrom_lengths.append((name, len(rec.seq)))
    chrom_df = pd.DataFrame(chrom_lengths, columns=["seqnames", "length"])

    # Build a minimal gene table from cleaned GTF (use rows with 'gene'/'mRNA'/'transcript')
    cols = ["seqnames", "start", "end", "strand", "gene_id", "gene_name"]
    entries = []
    with open(cleaned_gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqname, _, feature, start, end, _, strand, _, attrs = parts
            if feature not in ("gene", "mRNA", "transcript"):
                continue

            # extract gene_id
            m_id = re.search(r'(^|;)\s*gene_id\s+"([^"]+)"', attrs)
            gene_id = m_id.group(2) if m_id else None
            if gene_id is None:
                m_locus = re.search(r'(^|;)\s*locus_tag\s+"([^"]+)"', attrs)
                gene_id = m_locus.group(2) if m_locus else None
            if gene_id is None:
                continue

            # extract gene_name with fallbacks: gene_name -> gene -> Name
            gene_name = None
            m_name = re.search(r'(^|;)\s*gene_name\s+"([^"]+)"', attrs)
            if m_name:
                gene_name = m_name.group(2)
            else:
                m_gene = re.search(r'(^|;)\s*gene\s+"([^"]+)"', attrs)  # e.g., gene "dnaA"
                if m_gene:
                    gene_name = m_gene.group(2)
                else:
                    m_name2 = re.search(r'(^|;)\s*Name\s*"([^"]+)"', attrs)
                    if m_name2:
                        gene_name = m_name2.group(2)

            entries.append((seqname, int(start), int(end), strand, gene_id, gene_name))

    if not entries:
        log.error("No gene-like entries found in cleaned GTF to create BED.")
        sys.exit(1)

    df = pd.DataFrame(entries, columns=cols)

    # One row per gene_id: min start, max end; keep first seqname/strand/gene_name
    df = (
        df.sort_values(["gene_id", "start"])
          .groupby("gene_id", as_index=False)
          .agg({
              "seqnames": "first",
              "strand": "first",
              "start": "min",   # GTF is already 1-based inclusive
              "end": "max",
              "gene_name": "first",
          })
    )

    # Join chromosome lengths
    df = df.merge(chrom_df, on="seqnames", how="left")

    # Compute windows in 1-based inclusive space
    # + strand: [start-200, start+99]
    # - strand: [end-99,   end+200]
    df["start_200"] = df.apply(
        lambda r: max(r["start"] - 200, 1) if r["strand"] == "+"
        else min(r["end"] + 200, r["length"]),
        axis=1
    )
    df["end_100"] = df.apply(
        lambda r: min(r["start"] + 99, r["length"]) if r["strand"] == "+"
        else max(r["end"] - 99, 1),
        axis=1
    )

    # Ensure start <= end
    tmp = df["start_200"].copy()
    df["start_200"] = df[["start_200", "end_100"]].min(axis=1)
    df["end_100"] = pd.concat([tmp, df["end_100"]], axis=1).max(axis=1)

    # Convert to BED (0-based, half-open)
    df["bed_start"] = (df["start_200"] - 1).clip(lower=0).astype(int)
    df["bed_end"]   = df["end_100"].astype(int)
    df["score"]     = 0

    # BED name: gene_id|gene_name (or just gene_id)
    def make_bed_name(row):
        if pd.notna(row["gene_name"]) and str(row["gene_name"]).strip():
            return f"{row['gene_id']}|{row['gene_name']}"
        return str(row["gene_id"])

    df["bed_name"] = df.apply(make_bed_name, axis=1)

    # Write BED (6 columns; name=bed_name)
    bed_df = df[["seqnames", "bed_start", "bed_end", "bed_name", "score", "strand"]].copy()
    bed_df.to_csv(bed_path, sep="\t", header=False, index=False)
    log.info("Wrote TSS-window BED → %s (rows=%d)", bed_path, len(bed_df))

    # Optional FASTA via bedtools
    if shutil.which("bedtools"):
        step(f"[sTarML] Step 4/{total_steps}: Extracting TSS-window FASTA with bedtools", log, print_steps)
        cmd = [
            "bedtools", "getfasta", "-name+", "-s",
            "-fo", str(fasta_out_path),
            "-fi", str(genome_fna_path),
            "-bed", str(bed_path)
        ]
        run_cmd(cmd, log)
        log.info("Wrote TSS-window FASTA → %s", fasta_out_path)
    else:
        log.info("bedtools not found; skipped FASTA extraction for TSS windows.")
    return


# ---------- Prediction helpers ----------
def _safe_to_csr(mat):
    if hasattr(mat, "tocsr"):
        return mat.tocsr()
    return sp.csr_matrix(mat)

def _read_fasta_as_dict(fa_path):
    """Return dict{id: sequence(str)} from a FASTA; clean bedtools -name+ decorations."""
    seqs = {}
    for rec in SeqIO.parse(str(fa_path), "fasta"):
        rid = rec.id
        rid = rid.replace("(+)", "").replace("(-)", "")
        rid = rid.split(":")[0]
        seqs[rid] = str(rec.seq).upper()
    return seqs

def _set_n_jobs_if_possible(model, n_jobs, log):
    try:
        model.set_params(n_jobs=n_jobs)
        return
    except Exception:
        pass
    if hasattr(model, "named_steps"):
        for name, step_obj in model.named_steps.items():
            try:
                step_obj.set_params(n_jobs=n_jobs)
                log.info("Set n_jobs=%s on step '%s'", n_jobs, name)
            except Exception:
                continue

def _scale_allow_sparse(X, scaler, which, log):
    """Try scaling sparse X; if scaler requires dense, convert to dense then back to CSR."""
    if scaler is None:
        return X
    try:
        # If scaler can handle sparse, this just works
        X_scaled = scaler.transform(X)
        return _safe_to_csr(X_scaled)
    except TypeError as e:
        if "Sparse data was passed" in str(e):
            log.info("%s scaler requires dense input; converting to dense (this may increase memory).", which)
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            X_scaled = scaler.transform(X_dense)
            return _safe_to_csr(X_scaled)
        raise

def _load_required_artifacts(models_dir: Path, log):
    """Load vectorizers, scalers, and model from fixed filenames (Path-based)."""
    srna_vec_p = models_dir / SRNA_VEC_FILE
    mrna_vec_p = models_dir / MRNA_VEC_FILE
    srna_scl_p = models_dir / SRNA_SCL_FILE
    mrna_scl_p = models_dir / MRNA_SCL_FILE
    model_p    = models_dir / MODEL_FILE

    # Mandatory vectorizers + model
    for pth in (srna_vec_p, mrna_vec_p, model_p):
        if not pth.exists():
            log.error("Required model file missing: %s", pth)
            sys.exit(1)

    srna_vec = joblib.load(srna_vec_p)
    mrna_vec = joblib.load(mrna_vec_p)
    model    = joblib.load(model_p)

    srna_scl = joblib.load(srna_scl_p) if srna_scl_p.exists() else None
    mrna_scl = joblib.load(mrna_scl_p) if mrna_scl_p.exists() else None

    return srna_vec, mrna_vec, srna_scl, mrna_scl, model

def run_predictions(srna_fa, tss_fa, outdir, cores, log, print_steps=False, total_steps=7, script_dir: Path=None):
    """Generate all sRNA×TSS pairs, preprocess via external artifacts, run model, write predictions.tsv."""
    if script_dir is None:
        log.error("Internal error: script_dir not provided for locating 'models/'.")
        sys.exit(1)

    models_dir = script_dir / MODELS_DIRNAME
    if not models_dir.exists():
        log.error("Models directory not found: %s", models_dir)
        sys.exit(1)

    step(f"[sTarML] Step 6/{total_steps}: Loading model & preprocessors; generating sRNA×TSS pairs", log, print_steps)

    srna_vec, mrna_vec, srna_scl, mrna_scl, model = _load_required_artifacts(models_dir, log)
    _set_n_jobs_if_possible(model, cores, log)

    # Build cartesian product of sRNAs × TSS windows
    srna_dict = _read_fasta_as_dict(srna_fa)
    tss_dict  = _read_fasta_as_dict(tss_fa)

    rows = []
    for sid, sseq in srna_dict.items():
        for tid, tseq in tss_dict.items():
            rows.append({
                "srna_id": sid,
                "target_name": tid,  # BED name -> gene_id|gene_name
                "srna_sequence": sseq,
                "rna_seq_TSS_expanded": tseq,
            })
    
    if not rows:
        log.error("No sRNA×TSS pairs generated; check FASTA inputs.")
        sys.exit(1)
    df_pairs = pd.DataFrame(rows)

    srna = df_pairs["srna_sequence"]
    mrna = df_pairs["rna_seq_TSS_expanded"]

    srna_kmer = srna_vec.transform(srna).toarray()
    mrna_kmer = mrna_vec.transform(mrna).toarray()

    srna_kmer = srna_scl.transform(srna_kmer)
    mrna_kmer = mrna_scl.transform(mrna_kmer)

    # Concatenate features (sparse hstack)
    X_feat = np.concatenate((srna_kmer, mrna_kmer), axis=1)
    log.info("Combined feature matrix shape: %s", X_feat.shape)

    try:
        prob = model.predict_proba(X_feat)[:, 1]
        y_pred = (prob >= 0.5).astype(int)
    except Exception:
        prob = None
        log.warning("Model did not work properly !")


    df_pairs["prob_pos"] = prob
    df_pairs["pred"] = y_pred if prob is not None else None

    pred_path = Path(outdir) / "predictions.tsv"
    df_pairs.to_csv(pred_path, sep="\t", index=False)
    log.info("Wrote predictions → %s (rows=%d)", pred_path, len(df_pairs))
    step(f"[sTarML] Step 6/{total_steps}: Predictions written to {pred_path}", log, print_steps)
    return pred_path



def main():
    parser = argparse.ArgumentParser(description="sTarML — sRNA Target prediction with Machine Learning")
    parser.add_argument("--taxid", type=int, required=True, help="NCBI Taxonomy ID")
    parser.add_argument("--refseq", type=str, default=None,
                        help="Specific RefSeq accession (GCF_...) to download instead of the representative genome")
    parser.add_argument("--srna", type=str, default=None, help="Comma-separated RNA sequences")
    parser.add_argument("--srna-fasta", type=Path, default=None, help="FASTA file with sRNAs")
    parser.add_argument("--outdir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use")

    # Default: print steps to screen. Users can disable with --no-print-steps
    parser.add_argument("--no-print-steps", dest="print_steps", action="store_false",
                        help="Do not print step messages to screen (steps are always logged).")
    parser.set_defaults(print_steps=True)

    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    log_dir = ensure_dir(outdir / "logs")
    logging.basicConfig(
        filename=log_dir / "run.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger()

    # 7 steps including prediction
    total_steps = 7
    step(f"[sTarML] Step 1/{total_steps}: Initializing", log, args.print_steps)
    log.info("Starting pipeline for taxid=%s", args.taxid)

    fasta, gtf = download_genome_and_gtf(args.taxid, outdir, log, args.print_steps,
                                         refseq=args.refseq, total_steps=total_steps)
    log.info("Genome FASTA: %s", fasta)
    log.info("Annotation GTF: %s", gtf)

    step(f"[sTarML] Step 3/{total_steps}: Filtering annotation to protein-coding mRNAs", log, args.print_steps)
    cleaned_gtf = outdir / "annotation_cleaned.gtf"
    filter_gtf(gtf, cleaned_gtf, log)

    # Create TSS windows BED (+ optional FASTA)
    tss_bed = outdir / "tss_windows.bed"
    tss_fa  = outdir / "tss_windows.fasta"
    create_tss_bed(fasta, cleaned_gtf, tss_bed, tss_fa, log, args.print_steps, total_steps=total_steps)

    step(f"[sTarML] Step 5/{total_steps}: Saving sRNAs (DNA alphabet; U→T if present)", log, args.print_steps)
    srna_dir = ensure_dir(outdir / "srna")
    load_srna(args.srna, args.srna_fasta, srna_dir, log)
    srna_fa = srna_dir / "srnas.fasta"

    # Run predictions (model + preprocessors are mandatory and fixed-name)
    script_dir = Path(__file__).resolve().parent
    run_predictions(
        srna_fa=srna_fa,
        tss_fa=tss_fa,
        outdir=outdir,
        cores=args.cores,
        log=log,
        print_steps=args.print_steps,
        total_steps=total_steps,
        script_dir=script_dir
    )

    step(f"[sTarML] Step 7/{total_steps}: Finished successfully ✅", log, args.print_steps)
    log.info("sTarML pipeline finished successfully!")

if __name__ == "__main__":
    main()
