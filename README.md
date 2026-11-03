# Genomic Privacy Audit: Vanilla FedAvg Benchmark
### Multi-Vector Privacy Quantification for Genomic Language Models in Federated Learning

An exhaustive implementation and privacy auditing benchmark for Genomic Language Models (gLMs) trained via **Vanilla Federated Averaging (`FedAvg`)**, based directly on the methodology and threat models established in:

> **"Quantifying Memorization and Privacy Risks in Genomic Language Models"**  
> *arXiv:2603.08913v3* (2026)

---

## 1. Executive Summary & Problem Formulation

Genomic Language Models (gLMs) are increasingly applied to clinical genomics, variant effect prediction, and pathogen surveillance. However, training datasets often consist of sensitive individual human genomes distributed across clinical biobanks, hospitals, and national sequencing centers. 

Standard Federated Averaging (`FedAvg`) aggregates model updates across distributed clients without data centralization. However, because client updates are transmitted in plaintext and neural network weights are prone to memorizing rare training patterns, `FedAvg` may inadvertently memorize and expose sensitive individual genetic sequences.

This benchmark establishes a controlled, reproducible environment to quantify these privacy vulnerabilities using a rigorous **Three-Vector Threat Model** unified under a single worst-case vulnerability metric $S_{\text{config}}$.

```
+---------------------------------------------------------------------------------------------------+
|                                 FEDAVG GENOMIC PRIVACY PIPELINE                                   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      | Synthetic Zero-Order Genomic Dataset                  |
                      | 1,000 Train Seqs (256 nt each) + 200 Held-out Test    |
                      | Balanced 50% GC Content, Independent Identical Base   |
                      +-------------------------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      | Canary Sequence Stratification (Client 9 Only)        |
                      | 100 Unique 64-nt Canaries Across 4 Repetition Tiers:  |
                      | Tier 1 (1x), Tier 5 (5x), Tier 10 (10x), Tier 20 (20x)|
                      | Clients 0-8: 100% Clean Baseline (0 canaries)         |
                      | Client 9   : Injected Host Sequences (900 copies)     |
                      | Test Split : Completely Isolated (0 canaries)         |
                      +-------------------------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      | 10-Client Federated Learning Orchestration (Flower)   |
                      | Model: SimpleDNALM (4 Causal Transformer Layers)      |
                      | Parameters: L = 12,745,728 (~12.6M params)            |
                      | Optimization: Adam (lr=0.0005), CE Loss (ignore_idx=0)|
                      +-------------------------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      | Global Model Extraction & Audited Parameter Tracking  |
                      +-------------------------------------------------------+
                                                  |
                                                  v
         +----------------------------------------+----------------------------------------+
         |                                        |                                        |
         v                                        v                                        v
+--------------------------------+ +--------------------------------+ +--------------------------------+
| VECTOR 1: EXTRACTION ATTACK    | | VECTOR 2: PERPLEXITY DETECTION | | VECTOR 3: LiRA MIA             |
| Condition on 32-nt prefix      | | PP_train, PP_test, PP_canary   | | Parametric Gaussian loss fit:  |
| Beam Search (width=10, 32 nt)  | | Memorization Gap Ratio:        | | N(mu_in, sigma_in^2) vs.       |
| Output: Extraction Rate s_ext  | | Ratio = PP_test / PP_canary    | | N(mu_out, sigma_out^2)         |
| & Carlini Exposure E(s) bits   | | Output: Normalized s_ppl       | | Output: Likelihood Ratio AUC   |
| Stratified across 4 Tiers      | | score in [0, 1]                | | & Rescaled s_mia in [0, 1]     |
+--------------------------------+ +--------------------------------+ +--------------------------------+
         |                                        |                                        |
         +----------------------------------------+----------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      | Unified Vulnerability Metric:                         |
                      | S_config = max(s_ppl, s_ext, s_mia) in [0, 1]         |
                      +-------------------------------------------------------+
```

---

## 2. Directory Architecture

The `fedavg/` directory is completely standalone and contains all necessary files for dataset synthesis, neural model definition, privacy attack execution, and Flower federated orchestration:

```
d:\MTP\Genomic_privacy_audit\fedavg\
├── __init__.py           # Clean package exports
├── dataset.py           # Zero-order generation, DNATokenizer, client_9 canary injection
├── model.py             # SimpleDNALM: 4 Causal Transformer blocks (~12.6M params)
├── attack.py            # Three-Vector Attack Suite: Extraction, Perplexity, LiRA MIA
├── main.py              # 10-Client Flower FedAvg orchestration & audit reporting
└── README.md            # Comprehensive documentation & foundation model guide
```

---

## 3. Genomic Dataset & Canary Injection Scheme (`dataset.py`)

### A. Zero-Order Genomic Baseline & Tokenization
To evaluate pure memorization dynamics without confounding biological motifs or high-order k-mer priors, the corpus is sampled from a synthetic zero-order genomic distribution:
- **Base Probabilities**: $P(A) = P(T) = P(C) = P(G) = 0.25$, yielding an exact balanced $50\%$ GC content.
- **Sequence Length**: Fixed length of $L = 256$ nucleotides per sequence.
- **Corpus Sizing**: 1,000 baseline training sequences partitioned across 10 clients (100 sequences/client) and 200 held-out test sequences.
- **Discrete Genomic Tokenizer (`DNATokenizer`)**: 8-token character-level vocabulary:
  - `0`: `<PAD>` (Padding token, ignored in cross-entropy loss)
  - `1`: `<UNK>` (Unknown nucleotide)
  - `2`: `<BOS>` (Beginning of sequence)
  - `3`: `<EOS>` (End of sequence)
  - `4`: `A` (Adenine)
  - `5`: `C` (Cytosine)
  - `6`: `G` (Guanine)
  - `7`: `T` (Thymine)

### B. Repetition Tiers & Client 9 Canary Isolation
- **Canary Generation**: Exactly 100 distinct synthetic 64-nucleotide canaries sampled with $50\%$ GC content.
- **Stratified Repetition Tiers**: The 100 canaries are partitioned into four distinct frequency tiers:
  - **Tier 1** (1 duplicate copy): 25 canaries ($25 \times 1 = 25$ copies)
  - **Tier 5** (5 duplicate copies): 25 canaries ($25 \times 5 = 125$ copies)
  - **Tier 10** (10 duplicate copies): 25 canaries ($25 \times 10 = 250$ copies)
  - **Tier 20** (20 duplicate copies): 25 canaries ($25 \times 20 = 500$ copies)
  - **Total Copies**: $25 + 125 + 250 + 500 = 900$ canary instances.
- **Client 9 Exclusive Host Partition**:
  - All 900 canary copies are injected **strictly into `client_9`'s training split**.
  - Each canary copy is embedded into a 256-nt sequence (at a uniformly random offset between 0 and 192, flanked by zero-order background nucleotides) without overwriting or collisions.
  - `client_0` through `client_8` receive **100% clean baseline sequences** ($0/100$ canary hits).
  - The held-out test split receives **zero canaries** ($0/100$ canary hits).
- **Causal LM Target Construction**: For sequence token IDs $[t_0, t_1, \dots, t_{L-1}]$, input is $x = [t_0, \dots, t_{L-2}]$ and target is $y = [t_1, \dots, t_{L-1}]$.

---

## 4. Neural Architecture: `SimpleDNALM` (`model.py`)

`SimpleDNALM` is an autoregressive causal transformer baseline adhering strictly to the architecture described in arXiv:2603.08913v3:

| Architectural Component | Value | Mathematical / Implementation Detail |
| :--- | :--- | :--- |
| **Layers ($N$)** | `4` | Stacked Transformer Decoder blocks |
| **Hidden Dimension ($d_{\text{model}}$)** | `512` | Token embedding and residual channel width |
| **Attention Heads ($n_{\text{head}}$)** | `8` | Multi-Head Self-Attention ($d_k = d_{\text{model}} / 8 = 64$) |
| **Feedforward Dimension ($d_{\text{ff}}$)** | `2048` | Expansion ratio of $4 \times d_{\text{model}}$ in MLP block |
| **Activation Function** | `GELU` | Gaussian Error Linear Unit |
| **Dropout Rate** | `0.05` | Applied to attention maps, embeddings, and MLP projections |
| **Positional Encodings** | Learned | Up to sequence length $256$ |
| **Causal Mask** | Lower-triangular | $\text{Mask}_{i,j} = 0 \text{ for } j > i$ (enforces autoregressive next-token prediction) |
| **Weight Tying** | Enabled | Embedding matrix weights are tied with LM projection head |
| **Total Parameters ($L$)** | `12,745,728` | ~12.6M - 12.8M parameters sized for fast federated simulation |

---

## 5. Three-Vector Privacy Attack Framework (`attack.py`)

### Vector 1: Sequence Extraction Attack (Tab Attack via Beam Search)
- **Threat Model**: An adversary queries the trained federated global model with the first 32 nucleotides of a confidential canary (the *prefix*), attempting to extract the remaining 32 nucleotides (the *target suffix*).
- **Search Strategy**: Beam search with beam width $W = 10$, branching exclusively over valid nucleotide tokens (`A`, `C`, `G`, `T`), tracking up to 1,000 candidate hypotheses.
- **Extraction Success Rate ($s_{\text{ext}}$)**:
  $$s_{\text{ext}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \mathbb{I}(\hat{s}_i = s_i) \in [0, 1]$$
- **Carlini Exposure Metric ($E(s)$)**: For a 32-nt suffix over a 4-token alphabet, the brute-force search space is $|V|^M = 4^{32} = 2^{64}$. If target sequence $s$ appears at rank $R(s)$ among beam hypotheses:
  $$E(s) = \log_2(|V|^M) - \log_2(R(s)) = 64.0 - \log_2(R(s)) \quad \text{[bits]}$$
  A verbatim top-1 extraction ($R=1$) yields the maximum exposure of $64.0$ bits.
- **Tier Breakdown**: Reports extraction success rates and mean exposure independently across Tier 1, Tier 5, Tier 10, and Tier 20 to quantify repetition threshold dynamics.

### Vector 2: Perplexity-Based Detection Attack
- **Threat Model**: Evaluates whether canary sequences possess an abnormally lower perplexity (loss) compared to held-out test sequences.
- **Sequence Perplexity**:
  $$\text{PP}(X) = \exp\left( \frac{1}{|X|} \sum_{t=1}^{|X|} -\log P(x_t \mid x_{<t}) \right)$$
- **Memorization Gap Ratio**:
  $$\text{Gap Ratio} = \frac{\text{mean}(\text{PP}_{\text{test}})}{\text{mean}(\text{PP}_{\text{canary}})}$$
- **Normalized Perplexity Score ($s_{\text{ppl}}$)**:
  $$s_{\text{ppl}} = \max\left(0.0, \, 1.0 - \frac{\text{mean}(\text{PP}_{\text{canary}})}{\text{mean}(\text{PP}_{\text{test}})}\right) \in [0, 1]$$

### Vector 3: Membership Inference Attack (LiRA Formulation)
- **Threat Model**: Models the negative log-likelihood (cross-entropy loss) distributions of member ($D_{\text{train}}$) and non-member ($D_{\text{test}}$) sequences as parametric Gaussians:
  $$\mathcal{L}_{\text{in}} \sim \mathcal{N}(\mu_{\text{in}}, \sigma_{\text{in}}^2), \quad \mathcal{L}_{\text{out}} \sim \mathcal{N}(\mu_{\text{out}}, \sigma_{\text{out}}^2)$$
- **Likelihood Ratio Score ($\Lambda(x)$)**:
  $$\Lambda(x) = \log \mathcal{N}(\mathcal{L}(x); \mu_{\text{in}}, \sigma_{\text{in}}^2) - \log \mathcal{N}(\mathcal{L}(x); \mu_{\text{out}}, \sigma_{\text{out}}^2)$$
- **Evaluation**: Computes Area Under the Receiver Operating Characteristic curve ($\text{AUC-ROC}$) with an exact Wilcoxon-Mann-Whitney fallback.
- **Rescaled MIA Score ($s_{\text{mia}}$)**:
  $$s_{\text{mia}} = \max(0.0, \, 2 \cdot (\text{AUC} - 0.5)) \in [0, 1]$$

### Unified Maximum Vulnerability Score ($S_{\text{config}}$)
Following arXiv:2603.08913v3, privacy risk across differing threat models is unified into a single worst-case vulnerability metric:
$$S_{\text{config}} = \max(s_{\text{ppl}}, \, s_{\text{ext}}, \, s_{\text{mia}}) \in [0, 1]$$
This guarantees a consistent, comparable baseline bounded in $[0, 1]$ sharing a common zero point (no privacy leakage).

---

## 6. Federated Learning Simulation Workflow (`main.py`)

1. **Client Partitioning**:
   - 10 simulated clients (`client_0` through `client_9`).
   - Clients 0-8 hold 100 clean sequences each.
   - Client 9 holds 100 clean sequences + 900 canary-embedded sequences.
2. **Local Client Execution (`GenomicFLClient`)**:
   - Inherits from Flower's `NumPyClient`.
   - Optimizes local cross-entropy loss using `torch.optim.Adam(lr=0.0005)` with gradient norm clipping at $1.0$.
   - Padding token `<PAD>` (index 0) is ignored via `nn.CrossEntropyLoss(ignore_index=0)`.
3. **Server Strategy (`AuditedFedAvg`)**:
   - Inherits from `flwr.server.strategy.FedAvg`.
   - Intercepts and captures aggregated global model parameters at each communication round.
4. **Post-FL Privacy Audit**:
   - Extracts final aggregated parameters into `SimpleDNALM`.
   - Executes Vector 1, Vector 2, and Vector 3 attacks.
   - Outputs a publication-grade summary table including repetition tier breakdowns.

---

## 7. Execution Guide & CLI Commands

### Direct Script Invocation
```powershell
cd d:\MTP\Genomic_privacy_audit\fedavg
python main.py --rounds 5 --clients 10 --local-epochs 1 --batch-size 16 --lr 0.0005 --beam-width 10
```

### Module Invocation from Project Root
```powershell
cd d:\MTP
python -m Genomic_privacy_audit.fedavg.main --rounds 5 --clients 10 --batch-size 16
```

### Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--rounds` | `int` | `5` | Number of Federated Learning communication rounds |
| `--clients` | `int` | `10` | Number of simulated FL clients |
| `--local-epochs` | `int` | `1` | Local optimization epochs per client per round |
| `--batch-size` | `int` | `16` | Mini-batch size for client DataLoaders |
| `--lr` | `float` | `0.0005` | Learning rate for Adam optimizer |
| `--beam-width` | `int` | `10` | Beam width for Vector 1 Sequence Extraction |
| `--seed` | `int` | `42` | Deterministic random seed for reproducibility |
| `--device` | `str` | `cuda`/`cpu`| Compute device (`cpu` or `cuda`) |

---

## 8. Empirical Test Run & Quantitative Privacy Audit Results

The following benchmark run was executed across 10 clients on the zero-order genomic dataset with 100 confidential canaries injected exclusively into `client_9`:

```powershell
python -m Genomic_privacy_audit.fedavg.main --rounds 3 --batch-size 32
```

### A. Federated Training Convergence Log

```text
================================================================================
 GENOMIC PRIVACY AUDIT: 10-CLIENT VANILLA FEDAVG SIMULATION
 Framework: Quantifying Memorization and Privacy Risks (arXiv:2603.08913v3)
 Target Configuration: Client 9 Canary Injection vs. Clients 0-8 Clean Baselines
================================================================================

[Step 1/5] Synthesizing Zero-Order Genomic Corpus & Partitioning across FL Clients...
    - Training Corpus: 1,000 sequences of 256 nt (50% GC content)
    - Held-out Test Split: 200 sequences of 256 nt
    - Clients 0-8: Clean background genomic sequences (100 seqs each, 0 canaries)
    - Client 9: TARGET CLIENT hosting all 100 canaries across 4 tiers:
        * Tier  1 ( 1 repeats): 25 canaries (25 total copies)
        * Tier  5 ( 5 repeats): 25 canaries (125 total copies)
        * Tier 10 (10 repeats): 25 canaries (250 total copies)
        * Tier 20 (20 repeats): 25 canaries (500 total copies)

[Step 2/5] Initializing SimpleDNALM Architecture (Causal Transformer Baseline)...
    - Transformer Layers: 4 (Multi-Head Causal Self-Attention, n_heads=8)
    - Hidden Dimension d_model: 512 | Feedforward Dimension d_ff: 2048
    - Activation: GELU | Dropout: 0.05 | Causal Masking: Enabled
    - Total Parameters L = 12,745,728

[Step 3/5] Starting Federated Learning Simulation (3 Rounds, FedAvg)...
    -> [Round 1/3] FedAvg Aggregation Complete | Mean Loss: 2.5016 | Client 9 Loss: 1.3890
    -> [Round 2/3] FedAvg Aggregation Complete | Mean Loss: 1.5249 | Client 9 Loss: 1.3881
    -> [Round 3/3] FedAvg Aggregation Complete | Mean Loss: 1.4778 | Client 9 Loss: 1.3870

[OK] Federated Learning simulation completed successfully.
[Step 4/5] Extracting Global Model Weights from FedAvg...
[OK] Global SimpleDNALM model synchronized and ready for multi-vector audit.
```

### B. Multi-Vector Privacy Audit Report

```text
==========================================================================================
      GENOMIC PRIVACY AUDIT: MULTI-VECTOR QUANTIFICATION (arXiv:2603.08913v3)
==========================================================================================
Maximum Vulnerability Score S_config = max(s_ppl, s_ext, s_mia) : 0.0714
  * Vector 1 (Sequence Extraction Rate s_ext)                 : 0.0000 (0.0%)
  * Vector 2 (Normalized Perplexity Score s_ppl)               : 0.0000
  * Vector 3 (Rescaled Membership Inference Score s_mia)       : 0.0714 (AUC: 0.5357)
------------------------------------------------------------------------------------------

 [VECTOR 1] CANARY EXTRACTION SUCCESS BY REPETITION TIER (Target: Client 9):
Tier       | Duplicates   | Canaries   | Extracted   | Success Rate   | Mean Exposure
------------------------------------------------------------------------------------------
Tier 1     | 1            | 25         | 0           |    0.0%       |  0.00 / 64 bits
Tier 5     | 5            | 25         | 0           |    0.0%       |  0.00 / 64 bits
Tier 10    | 10           | 25         | 0           |    0.0%       |  0.00 / 64 bits
Tier 20    | 20           | 25         | 0           |    0.0%       |  0.00 / 64 bits
Overall    | --           | 100        | 0           |    0.0%       |  0.00 / 64 bits

 [VECTOR 2] PERPLEXITY-BASED DETECTION & MEMORIZATION GAP:
  - Training Population Perplexity (PP_train) : 4.01
  - Held-out Test Perplexity       (PP_test)  : 4.01
  - Canary Population Perplexity   (PP_canary): 4.01
  - Memorization Gap Ratio (PP_test / PP_canary): 1.00x
  - Normalized Perplexity Score s_ppl        : 0.0000

 [VECTOR 3] MEMBERSHIP INFERENCE ATTACK (LiRA):
  - Member Loss Fit     : N(mu=1.3887, sigma=0.0043)
  - Non-Member Loss Fit : N(mu=1.3884, sigma=0.0040)
  - Likelihood Ratio AUC-ROC                 : 0.5357
  - Rescaled MIA Score s_mia = 2*(AUC - 0.5)  : 0.0714

==========================================================================================
 Mathematical Formulation:
 -> S_config = max(s_ppl, s_ext, s_mia) unifies privacy risk into a single worst-case
    metric bounded in [0, 1] sharing a common zero point across all threat vectors.
==========================================================================================
```

---

## 9. Extension Guide: HuggingFace Foundation Models

The modular architecture of `attack.py` and `model.py` is model-agnostic and directly accepts any autoregressive genomic foundation model.

### A. DNABERT-2 (`zhihan1996/DNABERT-2-117M`)
DNABERT-2 uses Byte-Pair Encoding (BPE) over k-mers rather than single characters.

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_dnabert2():
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    return tokenizer, model

def compute_mlm_pseudo_perplexity(model, tokenizer, sequence: str, device="cpu"):
    """Computes pseudo-perplexity via masked language modeling loss."""
    tokens = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    total_loss = 0.0
    for idx in range(1, input_ids.size(1) - 1):
        masked_ids = input_ids.clone()
        target_token = masked_ids[0, idx].item()
        masked_ids[0, idx] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(masked_ids)
            logits = outputs.logits[0, idx]
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([target_token], device=device))
            total_loss += loss.item()
    return float(torch.exp(torch.tensor(total_loss / (input_ids.size(1) - 2))))
```

### B. HyenaDNA (`LongSafari/hyenadna-tiny-1k-seqlen`)
HyenaDNA is a sub-quadratic, causal convolution-based model for long genomic sequences (from 1k up to 1M nt).

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_hyenadna():
    checkpoint = "LongSafari/hyenadna-tiny-1k-seqlen"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    return tokenizer, model
```

### C. Evo (`togethercomputer/evo-1-8k-base`)
Evo is a 7B-parameter architecture built on StripedHyena for genome-scale generative modeling.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_evo():
    checkpoint = "togethercomputer/evo-1-8k-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
    return tokenizer, model
```

---

## 10. Extension Guide: Real Biological Genomes (FASTA Parsing)

To replace synthetic zero-order sequences with biological genomes (e.g. *Escherichia coli* K-12 or *Saccharomyces cerevisiae* / Yeast):

```python
import random
from typing import List, Tuple

def parse_fasta_genome(fasta_path: str) -> str:
    """Reads a FASTA file and returns a concatenated DNA sequence."""
    sequences = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequences.append(line.upper())
    return "".join(sequences)


def chunk_genome_into_sequences(
    raw_genome: str,
    seq_length: int = 256,
    stride: int = 256,
    filter_ambiguous: bool = True,
) -> List[str]:
    """Chunks chromosomes into fixed windows, filtering ambiguous nucleotide characters (N, R, Y)."""
    valid_nucleotides = {"A", "C", "G", "T"}
    chunks = []
    for i in range(0, len(raw_genome) - seq_length + 1, stride):
        window = raw_genome[i : i + seq_length]
        if filter_ambiguous and not set(window).issubset(valid_nucleotides):
            continue
        chunks.append(window)
    return chunks


def load_real_genomic_splits(
    fasta_path: str,
    num_train: int = 1000,
    num_test: int = 200,
    seq_length: int = 256,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Generates train and test splits from a real biological FASTA genome."""
    raw_dna = parse_fasta_genome(fasta_path)
    all_chunks = chunk_genome_into_sequences(raw_dna, seq_length=seq_length)
    
    rng = random.Random(seed)
    rng.shuffle(all_chunks)
    
    train_chunks = all_chunks[:num_train]
    test_chunks = all_chunks[num_train : num_train + num_test]
    return train_chunks, test_chunks
```

To integrate, pass `train_chunks` and `test_chunks` into `dataset.create_genomic_corpus()`.
