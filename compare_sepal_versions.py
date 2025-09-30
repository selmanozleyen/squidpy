#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore")
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import anndata as ad
import squidpy as sq
import scanpy as sc
from squidpy.gr import sepal_config

# Path setup
HOME = Path(os.path.expanduser("~"))
DATA_PATH = HOME / "data/visium_hne_adata.h5ad"


# Load test data
def load_data():
    if not DATA_PATH.exists():
        print(f"Creating test dataset at {DATA_PATH}")
        adata = sq.datasets.visium_hne_adata()
        sq.gr.spatial_neighbors(adata)
        adata.write_h5ad(DATA_PATH)
    else:
        print(f"Loading existing dataset from {DATA_PATH}")
        adata = ad.read_h5ad(DATA_PATH)
        sc.pp.filter_genes(adata, min_cells=3, inplace=True)
        sc.pp.filter_cells(adata, min_genes=3, inplace=True)
    return adata


# Define different numba configurations to test
NUMBA_CONFIGS = {
    "default": {"fastmath": True},
    "no_fastmath": {"fastmath": False},
}


def run_single_gene(adata, gene, configs):
    """Run SEPAL for a single gene with different configurations"""
    print(f"\nTesting gene: {gene}")
    results = {}
    timings = {}

    for name, config in configs.items():
        print(f"  Running config: {name}")
        try:
            start_time = time.time()
            result = sepal_config(
                adata, max_neighs=6, genes=[gene], n_iter=30000, copy=True, numba_config=config, config_name=name
            )
            end_time = time.time()
            elapsed = end_time - start_time

            print(f"    Completed in {elapsed:.2f} seconds")
            print(f"    Score: {result.iloc[0, 0]:.6f}")

            results[name] = result
            timings[name] = elapsed
        except Exception as e:
            print(f"    Failed with error: {str(e)}")

    return results, timings


def run_gene_batch(adata, genes, configs=None, max_genes=10):
    """Run SEPAL comparison for multiple genes"""
    if configs is None:
        configs = NUMBA_CONFIGS

    # Limit number of genes to test
    test_genes = genes[:max_genes]
    print(f"Testing {len(test_genes)} genes with configurations: {list(configs.keys())}")

    all_results = {}
    all_timings = {}

    for gene in test_genes:
        gene_results, gene_timings = run_single_gene(adata, gene, configs)
        all_results[gene] = gene_results
        all_timings[gene] = gene_timings

    return all_results, all_timings


def analyze_results(all_results, all_timings):
    """Analyze results across genes and configurations"""
    # Extract scores for each configuration
    genes = list(all_results.keys())
    configs = list(all_results[genes[0]].keys()) if genes else []

    if not genes or not configs:
        print("No results to analyze")
        return None

    # Create DataFrame with all scores
    scores_data = []
    for gene in genes:
        for config in configs:
            if config in all_results[gene]:
                result = all_results[gene][config]
                if not result.empty:
                    score_val = result.iloc[0, 0]
                    time_val = all_timings[gene][config]
                    scores_data.append({"gene": gene, "config": config, "score": score_val, "time": time_val})

    scores_df = pd.DataFrame(scores_data)
    if scores_df.empty:
        print("No valid scores to analyze")
        return None

    # Create a pivot table for easier comparison
    pivot_df = scores_df.pivot(index="gene", columns="config", values="score")

    # Check for identical score values between configurations
    print("\nChecking for identical scores between configurations:")
    for i, config1 in enumerate(pivot_df.columns):
        for j, config2 in enumerate(pivot_df.columns):
            if i < j:  # Only compare unique pairs
                if pivot_df[config1].equals(pivot_df[config2]):
                    print(f"  IDENTICAL SCORES: '{config1}' and '{config2}' produce exactly the same values")
                else:
                    diff = (pivot_df[config1] - pivot_df[config2]).abs().sum()
                    print(f"  '{config1}' vs '{config2}': Sum of absolute differences = {diff:.6f}")

    # Add rankings WITHIN each configuration (not across configurations)
    rank_df = pivot_df.rank(axis=0, ascending=False)  # Changed axis=1 to axis=0
    rank_df.columns = [f"{col}_rank" for col in rank_df.columns]

    # Check for identical rankings
    print("\nChecking for identical rankings between configurations:")
    for i, col1 in enumerate(rank_df.columns):
        for j, col2 in enumerate(rank_df.columns):
            if i < j:  # Only compare unique pairs
                if rank_df[col1].equals(rank_df[col2]):
                    print(f"  IDENTICAL RANKINGS: '{col1.replace('_rank', '')}' and '{col2.replace('_rank', '')}' produce exactly the same rankings")
                    
    # Combine scores and rankings
    combined_df = pd.concat([pivot_df, rank_df], axis=1)

    # Calculate timing statistics
    timing_pivot = scores_df.pivot(index="gene", columns="config", values="time")
    mean_times = timing_pivot.mean()

    # Print results
    print("\nMean execution times:")
    for config, time_val in mean_times.items():
        print(f"  {config}: {time_val:.2f} seconds")

    print("\nScore ranges:")
    for config in pivot_df.columns:
        min_val = pivot_df[config].min()
        max_val = pivot_df[config].max()
        print(f"  {config}: {min_val:.6f} to {max_val:.6f}")

    # Calculate rank correlations
    rank_cols = [col for col in combined_df.columns if col.endswith("_rank")]
    if len(rank_cols) > 1:
        # Show ranking data for debugging
        print("\nRanking data:")
        print(combined_df[rank_cols])
        
        # Calculate correlation with proper handling of identical rankings
        rank_corr = combined_df[rank_cols].corr(method="spearman")
        
        # Replace NaN with 1.0 for identical rankings
        for i in range(len(rank_cols)):
            for j in range(len(rank_cols)):
                if pd.isna(rank_corr.iloc[i, j]):
                    col_i, col_j = rank_cols[i], rank_cols[j]
                    if combined_df[col_i].equals(combined_df[col_j]):
                        rank_corr.iloc[i, j] = 1.0
                        print(f"  Fixed NaN correlation: perfect correlation (1.0) between {col_i} and {col_j}")
        
        print("\nRank correlation (Spearman):")
        print(rank_corr)

        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(rank_corr, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Rank Correlation Between Configurations")
        plt.tight_layout()
        plt.savefig("sepal_rank_correlation.png", dpi=150)
        print("Saved rank correlation heatmap to sepal_rank_correlation.png")

    # Create score comparison boxplot
    plt.figure(figsize=(12, 6))

    # Convert to long format for boxplot
    scores_long = scores_df.melt(id_vars=["gene"], value_vars=["score"], var_name="metric", value_name="value")
    scores_long = scores_long.join(scores_df[["config"]])

    # Create boxplot
    sns.boxplot(data=scores_df, x="config", y="score")
    plt.title("SEPAL Score Distribution by Configuration")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("sepal_score_distribution.png", dpi=150)
    print("Saved score distribution plot to sepal_score_distribution.png")

    # Save detailed results
    combined_df.to_csv("sepal_comparison_results.csv")
    print("Saved detailed results to sepal_comparison_results.csv")

    return combined_df


if __name__ == "__main__":
    # Load data
    adata = load_data()

    # First test with a single gene to verify it works
    test_gene = adata.var_names[0]
    print(f"Initial test with gene: {test_gene}")

    single_results, single_timings = run_single_gene(adata, test_gene, {"default": {"fastmath": True}})

    if not single_results:
        print("Initial test failed, check the dataset")
    else:
        # Now run with multiple genes and configurations
        genes = adata.var_names[:20].tolist()  # Test with 20 genes
        all_results, all_timings = run_gene_batch(adata, genes, NUMBA_CONFIGS, max_genes=10)

        # Analyze results
        result_df = analyze_results(all_results, all_timings)

        print("\nDone!")
