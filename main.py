"""
main.py - 10-Client Vanilla FedAvg Federated Learning & Multi-Vector Genomic Privacy Audit.
Based on "Quantifying Memorization and Privacy Risks in Genomic Language Models" (arXiv:2603.08913v3).

Federated Learning Architecture:
  1. Partitions a synthetic zero-order genomic corpus (1,000 sequences of length 256 nt, 50% GC)
     across 10 simulated clients (100 sequences / client).
  2. Embeds 100 confidential 64-nt canaries across four repetition tiers (1x, 5x, 10x, 20x duplicates)
     exclusively within Client 9's training partition (Clients 0-8 remain clean).
  3. Trains SimpleDNALM (a 4-layer causal transformer with d_model=512, n_heads=8, d_ff=2048)
     via Flower's Federated Averaging (FedAvg) strategy over multiple communication rounds.
  4. Extracts the aggregated global model and executes the complete Three-Vector Privacy Attack:
       - Vector 1: Sequence Extraction (Tab Attack beam search on 32-nt prefix -> s_ext & Exposure)
       - Vector 2: Perplexity-Based Detection (memorization gap PP_test / PP_canary -> s_ppl)
       - Vector 3: Membership Inference Attack (LiRA likelihood ratio Gaussian fit -> s_mia)
       - Unified Metric: S_config = max(s_ppl, s_ext, s_mia)
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Handle paths for standalone or package execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{PARENT_DIR};{CURRENT_DIR};" + os.environ.get("PYTHONPATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

try:
    from fedavg.dataset import (
        DNATokenizer,
        CanaryMetadata,
        get_genomic_dataloaders,
    )
    from fedavg.model import SimpleDNALM
    from fedavg.attack import (
        run_three_vector_privacy_audit,
        print_genomic_privacy_report,
    )
except ImportError:
    try:
        from Genomic_privacy_audit.fedavg.dataset import (
            DNATokenizer,
            CanaryMetadata,
            get_genomic_dataloaders,
        )
        from Genomic_privacy_audit.fedavg.model import SimpleDNALM
        from Genomic_privacy_audit.fedavg.attack import (
            run_three_vector_privacy_audit,
            print_genomic_privacy_report,
        )
    except ImportError:
        from dataset import (
            DNATokenizer,
            CanaryMetadata,
            get_genomic_dataloaders,
        )
        from model import SimpleDNALM
        from attack import (
            run_three_vector_privacy_audit,
            print_genomic_privacy_report,
        )


def set_deterministic_seeds(seed: int = 42) -> None:
    """Sets random seeds for deterministic, reproducible model training and simulation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class GenomicFLClient(NumPyClient):
    """
    Flower Federated Learning Client training SimpleDNALM on local genomic data partitions.
    """

    def __init__(
        self,
        client_id: str,
        model: SimpleDNALM,
        loader: DataLoader,
        local_epochs: int = 1,
        lr: float = 0.0005,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.loader = loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.model.set_weights(parameters)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        last_loss = 0.0
        for _ in range(self.local_epochs):
            for bx, by in self.loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(bx)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), by.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                last_loss = loss.item()

        num_samples = len(self.loader.dataset) if hasattr(self.loader, "dataset") else len(self.loader)
        return self.model.get_weights(), num_samples, {"loss": float(last_loss)}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict[str, float]]:
        self.model.set_weights(parameters)
        self.model.to(self.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for bx, by in self.loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits, _ = self.model(bx)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), by.reshape(-1))
                total_loss += loss.item() * bx.size(0)
                total_samples += bx.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        return float(avg_loss), total_samples, {"loss": float(avg_loss)}


class AuditedFedAvg(FedAvg):
    """
    Flower FedAvg strategy capturing aggregated global parameters for audit logging.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latest_global_parameters: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Optional[fl.server.client_proxy.ClientProxy], FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_global_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning Genomic Privacy Audit - Vanilla FedAvg (arXiv:2603.08913v3)"
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL communication rounds (default: 5)")
    parser.add_argument("--clients", type=int, default=10, help="Number of simulated FL clients (default: 10)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per client round (default: 1)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for Adam optimizer (default: 0.0005)")
    parser.add_argument("--beam-width", type=int, default=10, help="Beam width for Vector 1 extraction (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print(" GENOMIC PRIVACY AUDIT: 10-CLIENT VANILLA FEDAVG SIMULATION")
    print(" Framework: Quantifying Memorization and Privacy Risks (arXiv:2603.08913v3)")
    print(" Target Configuration: Client 9 Canary Injection vs. Clients 0-8 Clean Baselines")
    print("=" * 80)

    # 1. Deterministic Seeds
    set_deterministic_seeds(args.seed)

    # 2. Genomic Dataset & Canary Generation
    print("\n[Step 1/5] Synthesizing Zero-Order Genomic Corpus & Partitioning across FL Clients...")
    tokenizer, client_loaders, test_loader, canaries, train_seqs, test_seqs = get_genomic_dataloaders(
        num_clients=args.clients,
        num_train=1000,
        num_test=200,
        seq_length=256,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"    - Training Corpus: 1,000 sequences of 256 nt (50% GC content)")
    print(f"    - Held-out Test Split: 200 sequences of 256 nt")
    print(f"    - Clients 0-8: Clean background genomic sequences (100 seqs each, 0 canaries)")
    print(f"    - Client 9: TARGET CLIENT hosting all 100 canaries across 4 tiers:")
    for tier in [1, 5, 10, 20]:
        count = sum(1 for c in canaries if c.tier == tier)
        print(f"        * Tier {tier:>2} ({tier:>2} repeats): {count} canaries ({count * tier} total copies)")

    # 3. Model Architecture Provisioning
    print("\n[Step 2/5] Initializing SimpleDNALM Architecture (Causal Transformer Baseline)...")
    init_model = SimpleDNALM(
        vocab_size=len(tokenizer),
        d_model=512,
        num_layers=4,
        n_heads=8,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.05,
    ).to(device)

    total_params = sum(p.numel() for p in init_model.parameters())
    print(f"    - Transformer Layers: 4 (Multi-Head Causal Self-Attention, n_heads=8)")
    print(f"    - Hidden Dimension d_model: 512 | Feedforward Dimension d_ff: 2048")
    print(f"    - Activation: GELU | Dropout: 0.05 | Causal Masking: Enabled")
    print(f"    - Total Parameters L = {total_params:,}")

    # 4. Federated Learning Simulation (FedAvg)
    print(f"\n[Step 3/5] Starting Federated Learning Simulation ({args.rounds} Rounds, FedAvg)...")
    strategy = AuditedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=ndarrays_to_parameters(init_model.get_weights()),
    )

    current_parameters = init_model.get_weights()
    for rnd in range(1, args.rounds + 1):
        fit_results = []
        round_losses = []

        for cid in range(args.clients):
            loader = client_loaders[cid]
            local_model = SimpleDNALM(
                vocab_size=len(tokenizer),
                d_model=512,
                num_layers=4,
                n_heads=8,
                d_ff=2048,
                max_seq_len=256,
                dropout=0.05,
            )
            client = GenomicFLClient(
                client_id=f"client_{cid}",
                model=local_model,
                loader=loader,
                local_epochs=args.local_epochs,
                lr=args.lr,
                device=device,
            )
            trained_weights, num_samples, metrics = client.fit(current_parameters, {})
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(trained_weights),
                num_examples=num_samples,
                metrics=metrics,
            )
            fit_results.append((None, fit_res))
            round_losses.append(metrics.get("loss", 0.0))

        aggregated_parameters, _ = strategy.aggregate_fit(server_round=rnd, results=fit_results, failures=[])
        if aggregated_parameters is not None:
            current_parameters = parameters_to_ndarrays(aggregated_parameters)

        mean_loss = float(np.mean(round_losses))
        c9_loss = round_losses[9]
        print(f"    -> [Round {rnd}/{args.rounds}] FedAvg Aggregation Complete | Mean Loss: {mean_loss:.4f} | Client 9 Loss: {c9_loss:.4f}")

    print("\n[OK] Federated Learning simulation completed successfully.")

    # 5. Extract Final Global Model
    print("\n[Step 4/5] Extracting Global Model Weights from FedAvg...")
    global_model = SimpleDNALM(
        vocab_size=len(tokenizer),
        d_model=512,
        num_layers=4,
        n_heads=8,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.05,
    ).to(device)
    global_model.set_weights(current_parameters)
    global_model.eval()
    print("[OK] Global SimpleDNALM model synchronized and ready for multi-vector audit.")

    # 6. Execute Three-Vector Genomic Privacy Audit
    print("\n[Step 5/5] Executing Multi-Vector Privacy Quantification against Client 9's Canaries...")
    audit_results = run_three_vector_privacy_audit(
        model=global_model,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        canaries=canaries,
        tokenizer=tokenizer,
        beam_width=args.beam_width,
        max_eval_samples=200,
        device=device,
    )

    # 7. Print Publication-Grade Quantitative Report
    print_genomic_privacy_report(audit_results)


if __name__ == "__main__":
    main()

