"""
fedavg package - Standard Federated Averaging for Genomic Privacy Auditing.
"""

from .model import SimpleDNALM, TransformerBlock, CausalSelfAttention
from .dataset import (
    DNATokenizer,
    CanaryMetadata,
    GenomicDataset,
    load_gue_dataset_sequences,
    process_sequence_to_length,
    get_huggingface_token,
    generate_canary_sequences,
    create_genomic_corpus,
    get_genomic_dataloaders,
    NUCLEOTIDES,
)
from .attack import (
    run_sequence_extraction_attack,
    run_perplexity_detection_attack,
    run_membership_inference_attack,
    run_three_vector_privacy_audit,
    print_genomic_privacy_report,
    compute_sequence_perplexity,
    beam_search_canary_extraction,
    compute_auc_roc,
)

__all__ = [
    "SimpleDNALM",
    "TransformerBlock",
    "CausalSelfAttention",
    "DNATokenizer",
    "CanaryMetadata",
    "GenomicDataset",
    "load_gue_dataset_sequences",
    "process_sequence_to_length",
    "get_huggingface_token",
    "generate_canary_sequences",
    "create_genomic_corpus",
    "get_genomic_dataloaders",
    "NUCLEOTIDES",
    "run_sequence_extraction_attack",
    "run_perplexity_detection_attack",
    "run_membership_inference_attack",
    "run_three_vector_privacy_audit",
    "print_genomic_privacy_report",
    "compute_sequence_perplexity",
    "beam_search_canary_extraction",
    "compute_auc_roc",
]
