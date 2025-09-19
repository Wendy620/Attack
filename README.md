# Attack: Jamming Retrieval-Augmented Generation

This repository implements experiments from  
[**Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**](https://github.com/avitalsh/jamming_attack).  
It includes scripts for indexing, clean baseline generation, and adversarial attacks with multiple memory systems.

---

## 📂 Project Structure
jamming_attack/
├── adapters/ # Memory system adapters (Mem0, MemOS, A-mem, etc.)
├── utils/ # IO, embeddings, and seed control
├── metrics.py # Evaluation metrics (JSR, ASR, blocker@k)
├── build_index_memsys.py # Build vector indices
├── get_clean_memsys.py # Generate clean baseline (no attack)
├── attack.py # Run jamming attack experiments
└── requirements.txt # Dependencies

---

## 📥 Data Download

The datasets come from the original [jamming_attack](https://github.com/avitalsh/jamming_attack) repository.

```bash
# Clone the original repo
git clone https://github.com/avitalsh/jamming_attack.git

# Copy datasets into our project
mkdir -p corpus_poisoning/datasets
cp -r jamming_attack/corpus_poisoning/datasets/* corpus_poisoning/datasets/

Ensure the structure looks like:
corpus_poisoning/datasets/
  ├─ nq/
  │  ├─ corpus.jsonl
  │  ├─ queries.jsonl
  ├─ msmarco/
  │  ├─ corpus.jsonl
  │  ├─ queries.jsonl
```

## ⚙️ Installation

Clone this repo and install Python dependencies:

```bash
git clone https://github.com/Wendy620/Attack.git
cd Attack
pip install -r requirements.txt
```

## 🖥️ Backend Installation
```bash
This project supports **four memory backends**. Install only the ones you need.

### 1. Mem0
pip install mem0ai

### 2. Memoryos
pip install memoryos-pro -i https://pypi.org/simple

### 3.Memos
pip install MemoryOS

### 4.A-mem
# Clone the repo
git clone https://github.com/agiresearch/A-mem.git
cd A-mem

#### Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

#### Install A-mem
pip install .

After installation, return to the main project folder:

cd ..
```

## 🚀 Usage Examples
### 1. Build Index

python build_index_memsys.py \
  --backend mem0 \
  --namespace nq-gtr-10k \
  --dataset nq \
  --dataset_dir corpus_poisoning/datasets/nq \
  --emb_model gtr-base \
  --sample 10000 \
  --seed 42


### 2. Generate Clean Baseline

python get_clean_memsys.py \
  --backend mem0 \
  --namespace nq-gtr-10k \
  --dataset nq \
  --llm_model Llama-2-7b-chat-hf \
  --emb_model gtr-base \
  --k 5 \
  --num_queries 100 \
  --seed 0 \
  --oracle_llm gpt-4o-mini

### 3. Run Jamming Attack
   
python attack.py \
  --backend mem0 \
  --namespace nq-gtr-10k \
  --dataset nq \
  --llm_model Llama-2-7b-chat-hf \
  --emb_model gtr-base \
  --k 5 \
  --num_queries 100 \
  --seed 0 \
  --attack jamming \
  --doc_init mask \
  --num_tokens 50 \
  --n_iters 100 \
  --early_stop 50 \
  --batch_size 32 \
  --oracle_llm gpt-4o-mini \
  --oracle_emb gtr-base \
  --purge_blockers

## 📊 Metrics
| Metric         | Description                                             |
| -------------- | ------------------------------------------------------- |
| **JSR\@k**     | Jamming Success Rate: % queries where Top-k hit blocker |
| **blocker\@1** | % queries where Top-1 result is a blocker               |
| **ASR**        | Attack Success Rate: % answers containing target string |


Results are saved under: store/results/rag_gtr-base_x_<LLM>/<dataset>/<backend>/<mode>/<seed>/

## 📈 Example Result Table
| Backend | Dataset  | LLM                | JSR\@k | blocker\@1 | ASR  |
| ------- | -------- | ------------------ | ------ | ---------- | ---- |
| mem0    | NQ       | Llama-2-7b-chat-hf | 0.45   | 0.30       | 0.10 |
| memos   | MS MARCO | Vicuna-7b-v1.5     | 0.50   | 0.35       | 0.12 |

## 📜 License
