#!/usr/bin/env python

import argparse
import logging
from os import path
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer
from mkt.ml.models.pooling import CombinedPoolingModel
from mkt.ml.cluster import find_kmeans, generate_clustering
from mkt.ml.plot import plot_dim_red_scatter, plot_knee, plot_scatter_grid
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path,
    model_name_drug,
    model_name_kinase,
    layer_drug="pooler_output",
    layer_kinase="pooler_output",
    hidden_size=256,
    dropout_rate=0.1,
    bool_drug_freeze=False,
    bool_kinase_freeze=False,
    device="cpu",
):
    """
    Load a trained CombinedPoolingModel from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pt file
        model_name_drug: Name/path of drug model
        model_name_kinase: Name/path of kinase model
        layer_drug: Layer to extract from drug model
        layer_kinase: Layer to extract from kinase model
        hidden_size: Hidden size used during training
        dropout_rate: Dropout rate used during training
        bool_drug_freeze: Whether drug model was frozen during training
        bool_kinase_freeze: Whether kinase model was frozen during training
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print("Initializing model architecture...")
    model = CombinedPoolingModel(
        model_name_drug=model_name_drug,
        model_name_kinase=model_name_kinase,
        layer_drug=layer_drug,
        layer_kinase=layer_kinase,
        hidden_size=hidden_size,
        bool_drug_freeze=bool_drug_freeze,
        bool_kinase_freeze=bool_kinase_freeze,
        dropout_rate=dropout_rate,
    )
    
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    
    model.eval()
    model = model.to(device)
    
    print("Model loaded successfully!")
    return model


def extract_kinase_embeddings(
    model,
    kinase_sequences,
    tokenizer,
    layer_name="pooler_output",
    max_length=512,
    batch_size=32,
    device="cpu",
):
    """
    Extract <CLS> embeddings from kinase sequences at specified layer.
    
    Args:
        model: Trained CombinedPoolingModel
        kinase_sequences: List of kinase sequences (strings)
        tokenizer: Tokenizer for kinase sequences
        layer_name: Layer name to extract embeddings from
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        device: Device to run on
        
    Returns:
        numpy array of shape (n_sequences, embedding_dim)
    """
    model.eval()
    all_embeddings = []
    
    print(f"Extracting embeddings for {len(kinase_sequences)} sequences...")
    
    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(kinase_sequences), batch_size)):
            batch_sequences = kinase_sequences[i:i + batch_size]
            
            # Tokenize batch
            encoded = tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Forward pass through kinase model
            kinase_output = model.model_kinase(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            # Extract embeddings from specified layer
            # For pooler_output, it's directly accessible
            if layer_name == "pooler_output":
                embeddings = kinase_output.pooler_output
            else:
                # For other layers like "hidden_states[0]" or "last_hidden_state"
                # Parse the layer name
                if layer_name == "last_hidden_state":
                    # Get <CLS> token embedding (first token)
                    embeddings = kinase_output.last_hidden_state[:, 0, :]
                elif layer_name.startswith("hidden_states"):
                    # Extract layer index, e.g., "hidden_states[0]"
                    import re
                    match = re.search(r'\[(\d+)\]', layer_name)
                    if match:
                        layer_idx = int(match.group(1))
                        embeddings = kinase_output.hidden_states[layer_idx][:, 0, :]
                    else:
                        raise ValueError(f"Cannot parse layer name: {layer_name}")
                else:
                    raise ValueError(f"Unsupported layer name: {layer_name}")
            
            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
    
    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"Extracted embeddings shape: {all_embeddings.shape}")
    return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract kinase embeddings from trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data1/tanseyw/projects/whitej/ki_llm_mxfactor/cv_trainer/2025-11-04_20-58-23/fold_1/checkpoints/5j6an2hs/best_model.pt",
        help="Path to best_model.pt file",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/data1/tanseyw/projects/whitej/missense-kinase-toolkit/data/davis_data_processed.csv",
        help="Path to CSV file with kinase sequences",
    )
    parser.add_argument(
        "--sequence-column",
        type=str,
        default="seq_klifs_residues_only",
        help="Column name containing kinase sequences (default: 'seq_klifs_residues_only')",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="group_consensus",
        help="Column name containing group consensus (default: 'group_consensus')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./kinase_embeddings.npy",
        help="Output path for .npy file (default: 'kinase_embeddings.npy')",
    )
    parser.add_argument(
        "--model-name-drug",
        type=str,
        default="DeepChem/ChemBERTa-77M-MTR",
        help="Drug model name (must match training)",
    )
    parser.add_argument(
        "--model-name-kinase",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="Kinase model name (must match training)",
    )
    parser.add_argument(
        "--layer-kinase",
        type=str,
        default="pooler_output",
        help="Layer to extract embeddings from (default: 'pooler_output')",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden size (must match training, default: 256)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate (must match training, default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        help="Optional: Path to config_log.json to auto-load model parameters",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--path_out",
        type=str,
        default="/data1/tanseyw/projects/whitej/missense-kinase-toolkit/plots",
        help="Path to output directory",
    )
    parser.add_argument(
        "--bool_scale",
        action="store_true",
        help="Scale input matrix",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load config if provided
    if args.config_json:
        print(f"Loading config from {args.config_json}...")
        with open(args.config_json, "r") as f:
            config = json.load(f)
        # Update args with config values if not explicitly provided
        # Note: This is a simple implementation, you may need to adjust
        # based on what's stored in your config_log.json
    
    # Load CSV
    print(f"Reading CSV from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if args.sequence_column not in df.columns:
        raise ValueError(
            f"Column '{args.sequence_column}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # pre-process dataframe (WT only, lipid kinases, non-null sequences)
    df = df.loc[df["is_wt"].apply(lambda x: x is True), :].reset_index(drop=True)
    df.loc[
        df["kinase_name"].str.startswith(("PIP", "PIK")), args.group_column
    ] = "Lipid"
    df = df.loc[df[args.sequence_column].notnull(), :].reset_index(drop=True)
    df = df[[args.sequence_column, args.group_column]].drop_duplicates().reset_index(drop=True)
    df = df.loc[~df[args.group_column].isin(["Other", "Atypical"]), :].reset_index(drop=True)

    kinase_sequences = df[args.sequence_column].tolist()
    print(f"Found {len(kinase_sequences)} sequences")
    
    # Load tokenizer for kinase model
    print(f"Loading tokenizer for {args.model_name_kinase}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_kinase)
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name_drug=args.model_name_drug,
        model_name_kinase=args.model_name_kinase,
        layer_kinase=args.layer_kinase,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate,
        device=device,
    )
    
    # Extract embeddings
    embeddings = extract_kinase_embeddings(
        model=model,
        kinase_sequences=kinase_sequences,
        tokenizer=tokenizer,
        layer_name=args.layer_kinase,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    print(f"Saving embeddings to {args.output}...")
    np.save(args.output, embeddings)
    
    print(f"Done! Embeddings saved with shape {embeddings.shape}")
    
    # Optionally save a metadata file
    metadata_path = Path(args.output).with_suffix(".json")
    metadata = {
        "checkpoint": args.checkpoint,
        "csv_file": args.csv,
        "sequence_column": args.sequence_column,
        "n_sequences": len(kinase_sequences),
        "embedding_dim": embeddings.shape[1],
        "layer_kinase": args.layer_kinase,
        "model_name_kinase": args.model_name_kinase,
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")

    if args.bool_scale:
        scale_bool = True
    else:
        scale_bool = False

    kmeans, list_sse, list_silhouette = find_kmeans(mx_input=embeddings, bool_scale=scale_bool)
    n_clusters = len(np.unique(kmeans.labels_))
    plot_knee(list_sse, n_clusters, filename="elbow.png", path_out=args.path_out)

    # PCA
    pca = generate_clustering("PCA", embeddings.T, bool_scale=scale_bool)
    df_pca = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"])
    plot_dim_red_scatter(df_pca, kmeans, method="PCA", path_out=args.path_out)
    plot_scatter_grid(df, df_pca, kmeans, "PCA", path_out=args.path_out, bool_iterable=False)

    # t-SNE
    tsne = generate_clustering("t-SNE", embeddings, bool_scale=scale_bool)
    df_tsne = pd.DataFrame(tsne.embedding_, columns=["tSNE1", "tSNE2"])
    plot_dim_red_scatter(df_tsne, kmeans, method="tSNE", path_out=args.path_out)
    plot_scatter_grid(df, df_tsne, kmeans, "t-SNE", path_out=args.path_out, bool_iterable=False)


if __name__ == "__main__":
    main()
