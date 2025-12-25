# NYIT AI Academic Advisor Chatbot

To Design, Develop and Benchmark an Efficient Hybrid AI Acadmic Advisor Chatbot: Analyzing AI Agent Models, Large Language Models, and Network Configurations.

## Demo

![Chatbot Demo](assets/demo.gif)

## Project Overview

This project implements a hybrid AI acadeimc advising chatbot that intelligently routes queries through three specialized modules:

- **Rule-based routing** for quick FAQ responses
- **RAG (Retrieval-Augmented Generation)** for context-aware answers
- **Multi-model LLM routing** for complex reasoning

## AI Academic Advising Chatbot Model

![AI Academic Advising Chatbot Model](assets/System_Arch_Diag.png)

## Key Features

- **Intelligent Routing:** Automatically selects optimal response method
- **Multi-Model Comparison:** Benchmarks 3 leading LLM providers
- **Cost Optimization:** Routes simple queries to rule-based or RAG modules
- **RAG Integration:** Context-aware responses using vector similarity
- **Production Deployment:** Local or Cloud-based on AWS EC2

## Tech Stack

- **Backend:** Python 3.12, FastAPI
- **LLM APIs:** OpenAI (GPT-4o-mini), Anthropic (Claude Sonnet 4), Groq (Llama 3.3)
- **Vector DB:** FAISS
- **Embeddings:** OpenAI text-embedding-3-small
- **Cloud Deployment:** AWS EC2 t3.micro (Ubuntu 24.04)

## Benchmark Results

### Performance Comparison

| Model           | Avg Latency | Tokens/sec | Cost/Query | Quality Score |
| --------------- | ----------- | ---------- | ---------- | ------------- |
| GPT-4o-mini     | 5000ms      | 50         | $0.0002    | 75%           |
| Claude Sonnet 4 | 7000ms      | 45         | $0.0025    | 90%           |
| Groq Llama 3.3  | 1500ms      | 500        | $0.0001    | 80%           |

### Key Findings

- **Groq is 5x faster** than GPT-4o-mini
- **GPT-4o-mini offers best value** for general queries
- **Claude provides highest quality** for complex reasoning
- **Hybrid routing saves 60% on costs** vs LLM only approach

## Project Structure

## Project Structure

```text
nyit_student_advising/
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   ├── main.py                # FastAPI application
│   │   ├── config.py              # Configuration
│   │   └── services/
│   │       ├── router.py          # Hybrid routing logic
│   │       ├── llm_service.py     # Multi-model LLM service
│   │       ├── rag_service.py     # RAG with FAISS
│   │       └── intent_matcher.py  # Rule-based matcher
│   ├── benchmark_tests/
│   │   ├── benchmark_multi_enhanced.py
│   │   ├── benchmark_multi_model.py
│   │   ├── benchmark.py
│   │   └── results/               # Benchmark results
│   ├── generate_charts/
│   │   ├── generate_multi_enhanced_charts.py
│   │   ├── generate_multi_model_charts.py
│   │   └── charts/                # Visualization outputs
│   ├── run_server.py              # Run backend server
│   ├── requirements.txt
│   └── .env                       # Environment variables (ignored)
├── data/
│   └── *.jsonl                    # Knowledge base files
├── assets/
│   ├── demo.gif                   #Demo chatbot GIF
│   └── System_Arch_Diag.png
├── run_chatbot.py                 # Interactive chatbot client
├── .gitignore
├── README.md
└── *.pem                          # AWS EC2 key
```

## Quick Start

### Prerequisites

- Python 3.12
- API keys to LLM integration
  - OpenAI API key
  - Anthropic API key
  - Groq API key
- Cloud setup:
  - AWS EC2 (.pem) key

### Installation

```bash
# Copy project folder or Clone repository
git clone https://github.com/utsinboots/nyit-student-advising-chatbot
cd nyit-ai-chatbot/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate

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
python run_server.py
```

Server runs at `http://localhost:8000`

### Test API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the graduation requirements?"}'
```

### Indexes

Have fresh indexes\rag_index before starting, simple rag_index folder, faiss.index auto creates after server starts

### Run Client

python run_chatbot.py

## Running Benchmarks

### Multi-Model Comparison

```bash
cd backend\benchmark_test\
python benchmark.py
python benchmark_multi_enhanced.py
python generate_multi_enhanced_charts.py
```

Results saved to `backed/benchmark_tests/results/` and charts to `/backed/generate_charts/..`

## Cloud Deployment

### AWS EC2 Deployment

See [docs/AWS_EC2_DEPLOYMENT](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance) for detailed deployment guide.

- AWS EC2 t3.micro instance (AWS free tier)
- OS: Ubuntu 24.04 LTS, 2 vCPUs, 1GB RAM
- Generate AWS EC2 private key and add it in the root directory ~/your_key_name.pem
- Copy/edit all essential files in proper paths after installing the OS on AWS

```bash
# AWS EC2
git clone https://github.com/utsinboots/nyit-student-advising-chatbot

# Connect to AWS EC2 after running the EC2 instance
ssh -i ~/nyit_student_advising/nyit-student-advising-key.pem ubuntu@_public_IPv4_address_
cd ~/nyit-student-advising-chatbot/backend
source venv/bin/activate
pip install -r requirements.txt
# Configure .env
python run.py
```

## Academic Context

**Institution:** New York Institute of Technology (NYIT)  
**Program:** M.S. in Computer Science  
**Course:** CSCI 870 Project I
**Semester:** Fall 2025

## Author

**Utshant Gurung**  
M.S. Computer Science
New York Institute of Technology
December 2025

## References

- Lanham, M. (2024). AI Agents in Action. Manning Publications.
- Liang, P., et al. (2022). Holistic evaluation of language models. arXiv:2211.09110.
- D. Park, G. -t. An, C. Kamyod and C. G. Kim, "A Study on Performance Improvement of Prompt Engineering for Generative AI with a Large Language Model," in Journal of Web Engineering, doi: 10.13052/jwe1540-9589.2285.
- K. Mikael, C. Öz, T. A. Rashid and G. S. Nariman, "A Hybrid Chatbot Model for Enhancing Administrative Support in Education: Comparative Analysis, Integration, and Optimization".
- M. Rahman, M. Abedin, M. Z. Abir, F. I. Ansari, A. Reza, F. Y. Sadeque and N. Farhan, “Transforming Mentorship: An AI Powered Chatbot Approach to University Guidance,” arXiv:2511.04172v1.
- C. W. Okonkwo and A. Ade-Ibijola, “Chatbots applications in education: A systematic review,” _Computers and Education: Artificial Intelligence_, doi:10.1016/j.caeai.2021.100033.
- Á. A. Martínez-Gárate, J. A. Aguilar-Calderón, C. Tripp-Barba and A. Zaldívar-Colado, "Model-Driven Approaches for Conversational Agents Development: A Systematic Mapping Study," doi: 10.1109/ACCESS.2023.3293849.
- Q. Lu, L. Zhu, X. Xu, Z. Xing and J. Whittle, "Toward Responsible AI in the Era of Generative AI: A Reference Architecture for Designing Foundation Model-Based Systems," doi: 10.1109/MS.2024.3406333.
- D. Park, G. -t. An, C. Kamyod and C. G. Kim, "A Study on Performance Improvement of Prompt Engineering for Generative AI with a Large Language Model," doi: 10.13052/jwe1540-9589.2285.
- S. Wu and M. Luo, "Selection and Resource Allocation Strategies for Chatbot Technologies in Higher Education: An Optimization Model Approach," doi: 10.1109/ACCESS.2025.3530413.

---
