# NYIT AI Academic Advisor Chatbot

An intelligent academic advising chatbot system with hybrid routing architecture for NYIT's Computer Science M.S. program.

## 🎯 Project Overview

This capstone project implements a production-ready AI chatbot that intelligently routes student queries through three specialized systems:
- **Rule-based routing** for quick FAQ responses
- **RAG (Retrieval-Augmented Generation)** for context-aware answers
- **Multi-model LLM routing** for complex reasoning

## 🏗️ Architecture
User Query
↓
Hybrid Router
├── Rule-Based (Fast, $0)
├── RAG + Embeddings (Medium, Low cost)
└── Multi-Model LLM (Slow, Higher cost)
├── GPT-4o-mini (Fast & cheap)
├── Claude Sonnet 4 (High quality)
└── Groq Llama 3.3 (Ultra-fast)

## ✨ Key Features

- **Intelligent Routing:** Automatically selects optimal response method
- **Multi-Model Comparison:** Benchmarks 3 leading LLM providers
- **Cost Optimization:** Routes simple queries to cheaper methods
- **RAG Integration:** Context-aware responses using vector similarity
- **Production Deployment:** Cloud-ready on AWS EC2

## 🛠️ Tech Stack

- **Backend:** Python 3.11+, FastAPI
- **LLM APIs:** OpenAI (GPT-4o-mini), Anthropic (Claude), Groq (Llama)
- **Vector DB:** FAISS
- **Embeddings:** OpenAI text-embedding-3-small
- **Deployment:** AWS EC2 (Ubuntu 24.04)

## 📊 Benchmark Results

### Performance Comparison

| Model | Avg Latency | Tokens/sec | Cost/Query | Quality Score |
|-------|-------------|------------|------------|---------------|
| GPT-4o-mini | 5000ms | 50 | $0.0002 | 75% |
| Claude Sonnet 4 | 7000ms | 45 | $0.0025 | 90% |
| Groq Llama 3.3 | 1500ms | 500 | $0.0001 | 80% |

### Key Findings

- ⚡ **Groq is 5x faster** than GPT-4o-mini
- 💰 **GPT-4o-mini offers best value** for general queries
- ⭐ **Claude provides highest quality** for complex reasoning
- 🎯 **Hybrid routing saves 60% on costs** vs LLM-only approach

## 📁 Project Structure
nyit_student_advising/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration
│   │   └── services/
│   │       ├── router.py        # Hybrid routing logic
│   │       ├── llm_service.py   # Multi-model LLM service
│   │       ├── rag_service.py   # RAG with FAISS
│   │       └── intent_matcher.py # Rule-based matcher
│   ├── data/                    # JSONL knowledge bases
│   ├── results/                 # Benchmark results
│   ├── charts/                  # Visualization outputs
│   └── requirements.txt
└── README.md

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Anthropic API key (for Claude)
- Groq API key (for Llama)

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/nyit-ai-chatbot.git
cd nyit-ai-chatbot/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Create `.env` file in `backend/` directory:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### Run Server
```bash
cd backend
python run.py
```

Server runs at `http://localhost:8000`

### Test API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the graduation requirements?"}'
```

## 📊 Running Benchmarks

### Multi-Model Comparison
```bash
cd backend
python benchmark_multi_enhanced.py
python generate_multi_enhanced_charts.py
```

Results saved to `results/` and charts to `charts_enhanced/`

## ☁️ Deployment

### AWS EC2 Deployment

See [AWS_EC2_DEPLOYMENT.md](docs/AWS_EC2_DEPLOYMENT.md) for detailed deployment guide.

Quick deploy:
```bash
# On EC2 instance
git clone https://github.com/YOUR_USERNAME/nyit-ai-chatbot.git
cd nyit-ai-chatbot/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Configure .env
python run.py
```

## 📈 Evaluation Methodology

Evaluation framework follows established practices:
- **Performance Metrics:** Latency, throughput, cost (Aminabadi et al., 2022)
- **Quality Assessment:** Flesch readability (Flesch, 1948), ROUGE-inspired completeness (Lin, 2004)
- **Multi-Model Comparison:** Holistic LLM evaluation (Liang et al., 2022)
- **Agent Evaluation:** Rubrics and grounding (Lanham, 2024)

## 🎓 Academic Context

**Institution:** New York Institute of Technology (NYIT)  
**Program:** M.S. in Computer Science  
**Course:** Capstone Project  
**Semester:** Fall 2025

## 👤 Author

**Utshant Gurung**  
M.S. Computer Science, NYIT  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NYIT Computer Science Department
- OpenAI, Anthropic, and Groq for LLM APIs
- Manning Publications - "AI Agents in Action" (Lanham, 2024)

## 📚 References

- Lanham, M. (2024). AI Agents in Action. Manning Publications.
- Liang, P., et al. (2022). Holistic evaluation of language models. arXiv:2211.09110.
- Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries.
- Flesch, R. (1948). A new readability yardstick. Journal of Applied Psychology, 32(3), 221-233.

---

**⭐ Star this repo if you found it helpful!**