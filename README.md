# 🛠️ Python Backend Generator (LLM-Powered, Docker, Auto-Test Sandbox)

**Python Backend Generator** is an AI-powered command-line tool that uses OpenAI GPT models to instantly scaffold robust Python backend projects from a single natural language prompt.  
It’s perfect for rapid prototyping, MVPs, bootstrapping, learning, and hackathons.

---

## 🚀 Features

- **AI Code Generation:** Converts your idea into a working backend using FastAPI, Flask, or Django.
- **Project Templates:** Choose from built-in project ideas or write your own description.
- **Best Practices:** Produces clean, idiomatic, production-style code.
- **JWT Authentication:** Built-in support for secure auth and user management.
- **Database Integration:** Choose SQLite, PostgreSQL, or MongoDB (auto-configured).
- **Comprehensive Testing:** Pytest-based unit & integration tests for all endpoints and logic.
- **Docker Support:** Auto-generates Dockerfile & Docker Compose for reproducible, containerized environments.
- **Sandbox Automation:** Runs the generated backend and tests inside Docker Compose, with API health checks.
- **Verbose Logging:** Every step is explained for easy troubleshooting and transparency.
- **Git & Security Ready:** Keeps your secrets safe, ignores bulky files, and is optimized for version control.

---

## 📸 Screenshot

> ![Python Backend Generator CLI Example](https://raw.githubusercontent.com/<your-username>/<your-repo-name>/main/docs/demo_screenshot.png)
>
> _Sample CLI run, project generation, Docker Compose, and auto-testing in action._

---

## 🏃 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Capybaralifestyle/Python-Backend-Generator.git
cd python-backend-generator

# 2. Setup Python environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Set your OpenAI API key (never commit your real key)
cp .env.example .env
# Edit .env and paste your OpenAI key

# 5. Run the generator!
python python-BE-generator.py
