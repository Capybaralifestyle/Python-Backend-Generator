#!/usr/bin/env python3
"""
Advanced Python Backend Generator (OpenAI, Docker Compose, Auto-Test in Sandbox, Health Check)
"""

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import time
import openai
import requests

# --- Encoding fix for Windows console ---
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def safe_print(message):
    try:
        clean_message = str(message)
        print(clean_message)
    except Exception:
        clean_message = ''.join(c for c in str(message) if ord(c) < 128)
        print(clean_message)

class Framework(Enum):
    FLASK = "flask"
    FASTAPI = "fastapi"
    DJANGO = "django"

@dataclass
class ProjectConfig:
    name: str
    framework: Framework
    features: List[str]
    database: str = "sqlite"
    auth: bool = False
    testing: bool = True
    docker: bool = True

class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            "crud_api": {
                "pattern": r"(crud|create.*read.*update.*delete|manage.*data|todo)",
                "template": """Create a complete CRUD API for {entity} with the following operations:
- GET /{entity}s - List all {entity}s with pagination
- GET /{entity}s/{{id}} - Get single {entity} by ID
- POST /{entity}s - Create new {entity}
- PUT /{entity}s/{{id}} - Update existing {entity}
- DELETE /{entity}s/{{id}} - Delete {entity}
Include proper validation, error handling, and HTTP status codes."""
            },
            "auth_system": {
                "pattern": r"(auth|login|register|jwt|token|user.*management)",
                "template": """Create an authentication system with:
- User registration endpoint
- Login with JWT token generation
- Password hashing with bcrypt
- Protected routes decorator/middleware
- Token validation and refresh
- Logout functionality
Include proper security practices and validation."""
            },
            "ecommerce_api": {
                "pattern": r"(ecommerce|shop|product|order|cart|payment)",
                "template": """Create an e-commerce API with:
- Product catalog with categories
- Shopping cart functionality
- Order management system
- User accounts and profiles
- Payment processing endpoints
- Inventory management
Include proper data relationships and business logic."""
            },
            "blog_api": {
                "pattern": r"(blog|post|article|comment|content)",
                "template": """Create a blog API with:
- Blog posts with CRUD operations
- Comments system
- User authentication and profiles
- Categories and tags
- Search functionality
- Publishing workflow
Include proper content management features."""
            },
            "social_api": {
                "pattern": r"(social|friend|follow|like|share|feed)",
                "template": """Create a social media API with:
- User profiles and authentication
- Posts with likes and comments
- Friend/follower system
- News feed generation
- Real-time notifications
- Content sharing
Include social features and engagement tracking."""
            },
            "microservice": {
                "pattern": r"(microservice|service|api.*gateway|distributed)",
                "template": """Create a microservice for {domain} with:
- Health check endpoints
- Metrics and monitoring
- Configuration management
- Inter-service communication
- Proper logging and error handling
- Docker containerization
Include service discovery and resilience patterns."""
            }
        }

    def detect_template(self, user_request: str):
        import re
        user_request_lower = user_request.lower()
        for template_name, template_info in self.templates.items():
            if re.search(template_info["pattern"], user_request_lower):
                return template_name, template_info
        return "crud_api", self.templates["crud_api"]

class DocumentationManager:
    def __init__(self):
        self.docs = {
            "fastapi": {
                "basics": """
FastAPI Best Practices:
- Use dependency injection system
- Implement Pydantic models for validation
- Use APIRouter for route organization
- Implement proper async/await patterns
- Use FastAPI's automatic OpenAPI documentation
- Implement proper error handling with HTTPException
- Use SQLAlchemy with async support
- Structure: app/main.py, app/routers/, app/models/, app/dependencies/
""",
                "patterns": """
Common FastAPI Patterns:
- Router inclusion in main.py
- Pydantic BaseModel for request/response
- Dependency functions for common logic
- Background tasks for async operations
- Middleware for cross-cutting concerns
- Database sessions through dependencies
- Response models and status codes
"""
            },
            "flask": {
                "basics": """
Flask Best Practices:
- Use application factory pattern
- Organize routes with Blueprints
- Use Flask-CORS for cross-origin requests
- Implement proper error handling with @app.errorhandler
- Use Flask-SQLAlchemy for database operations
- Implement request validation with marshmallow or pydantic
- Use Flask-JWT-Extended for authentication
- Structure: app/__init__.py, app/routes/, app/models/, app/services/
""",
                "patterns": """
Common Flask Patterns:
- Blueprint registration in __init__.py
- Database models with SQLAlchemy
- Request/Response schemas with marshmallow
- Dependency injection through app context
- Configuration through environment variables
- Testing with pytest and Flask test client
"""
            }
        }

    def get_context(self, framework: Framework, features: List[str]) -> str:
        context_parts = []
        framework_key = framework.value
        if framework_key in self.docs:
            context_parts.extend([
                self.docs[framework_key]["basics"],
                self.docs[framework_key]["patterns"]
            ])
        return "\n".join(context_parts)

class SmartContextBuilder:
    def __init__(self):
        self.template_manager = PromptTemplateManager()
        self.doc_manager = DocumentationManager()

    def build_context(self, user_request: str, config: ProjectConfig) -> str:
        template_name, template_info = self.template_manager.detect_template(user_request)
        entities = self._extract_entities(user_request)
        framework_docs = self.doc_manager.get_context(config.framework, config.features)
        context = f"""
<thinking>
I need to analyze this request and generate production-ready Python code for a {config.framework.value.upper()} application.

Let me think through the requirements:
1. Framework: {config.framework.value}
2. Features requested: {config.features}
3. Database: {config.database}
4. Authentication needed: {config.auth}

The user wants: {user_request}

I should generate clean, well-structured code that follows best practices.
</thinking>

You are a senior Python developer specializing in {config.framework.value.upper()} development.

FRAMEWORK DOCUMENTATION AND BEST PRACTICES:
{framework_docs}

PROJECT CONFIGURATION:
- Framework: {config.framework.value}
- Features: {', '.join(config.features)}
- Database: {config.database}
- Authentication: {config.auth}
- Testing: {config.testing}

TEMPLATE TYPE: {template_name}
{template_info["template"].format(entity=entities[0] if entities else "item", description=user_request, domain=entities[0] if entities else "general")}

REQUIREMENTS:
- Generate complete, production-ready code
- Follow the documented best practices above
- Include proper error handling and validation
- Use type hints throughout
- Include docstrings for functions and classes
- Generate code that passes linting (flake8, black)
- Generate comprehensive pytest-based unit and integration tests for all endpoints and business logic:
  - Place all test files in the tests/ directory
  - Tests should cover all endpoints, typical usage, error handling, authentication, permissions, invalid input, and edge cases
  - Tests must hit the running API (not just call internal methods)
  - Use httpx.AsyncClient for FastAPI or Flask test client for Flask
  - Ensure all tests pass
  - Provide sample test data and mocks as needed
- No explanations, just clean code and tests

USER REQUEST: {user_request}

Generate the complete Python code:
"""
        return context

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        import re
        common_entities = ["user", "book", "product", "order", "customer", "post", "comment", "task", "todo", "blog", "article"]
        entities = []
        text_lower = text.lower()
        for entity in common_entities:
            if entity in text_lower:
                entities.append(entity)
        custom_entities = re.findall(r'\\b[A-Z][a-z]+\\b', text)
        entities.extend([e.lower() for e in custom_entities])
        return list(set(entities)) if entities else ["item"]

def create_docker_compose(project_path: Path, config: ProjectConfig):
    if config.database == "postgresql":
        compose = f"""version: '3.8'
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: app_db
      POSTGRES_USER: app_user
      POSTGRES_PASSWORD: app_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  api:
    build: .
    depends_on:
      - db
    environment:
      DATABASE_URL: postgres://app_user:app_pass@db:5432/app_db
    ports:
      - "8000:8000"
volumes:
  postgres_data:
"""
    elif config.database == "mongodb":
        compose = f"""version: '3.8'
services:
  db:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
  api:
    build: .
    depends_on:
      - db
    environment:
      DATABASE_URL: mongodb://db:27017/app_db
    ports:
      - "8000:8000"
volumes:
  mongo_data:
"""
    else:
        compose = f"""version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
"""
    (project_path / "docker-compose.yml").write_text(compose)
    safe_print("[VERBOSE] Docker Compose file created.")

def create_test_runner_scripts(project_path: Path):
    (project_path / "run_tests.sh").write_text(
        "#!/bin/bash\nsource venv/bin/activate 2>/dev/null || source .venv/bin/activate\necho Running pytest...\npytest tests/"
    )
    (project_path / "run_tests.bat").write_text(
        "@echo off\r\nvenv\\Scripts\\activate.bat || .venv\\Scripts\\activate.bat\r\necho Running pytest...\r\npytest tests\\"
    )
    safe_print("[VERBOSE] Test runner scripts created.")

PROJECT_SUGGESTIONS = [
    "A FastAPI todo app with CRUD and JWT auth",
    "A FastAPI book catalog with CRUD and search",
    "A FastAPI user registration and login system with email verification",
    "A FastAPI blog backend with comments and categories",
    "A FastAPI inventory management system for a small shop",
    "A FastAPI movie ratings and review API",
    "A FastAPI recipe sharing platform with tags and user profiles",
    "A FastAPI microservice for image upload and metadata extraction",
    "A FastAPI project tracker with tasks, sprints, and team members",
    "A FastAPI classroom API for students, courses, and assignments"
]

def choose_project_idea():
    safe_print("\n=== Python Backend Generator (OpenAI GPT, Docker Compose, Tests, Sandbox) ===\n")
    safe_print("TIP: For best results, specify what you want tested. For example:")
    safe_print("  - Include full pytest tests for endpoints, authentication, edge cases, and error handling.\n")
    safe_print("Choose one of the ready-to-use project ideas, or type your own.\n")
    for idx, idea in enumerate(PROJECT_SUGGESTIONS, 1):
        safe_print(f"{idx}. {idea}")
    safe_print("\nEnter number (1-10) or type your own project description:")
    user_input = input("> ").strip()
    if user_input.isdigit():
        i = int(user_input)
        if 1 <= i <= len(PROJECT_SUGGESTIONS):
            return PROJECT_SUGGESTIONS[i-1]
    return user_input

class AdvancedPythonBackendGenerator:
    def __init__(self, openai_model="gpt-4-turbo"):
        self.openai_model = openai_model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set. Did you load your .env?")
        openai.api_key = self.openai_api_key
        self.output_dir = Path("generated_projects")
        self.context_builder = SmartContextBuilder()
        self.output_dir.mkdir(exist_ok=True)

    def call_llm(self, prompt: str) -> str:
        try:
            safe_print(f"[VERBOSE] Calling OpenAI model: {self.openai_model}")
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a senior Python backend developer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096
            )
            result = response.choices[0].message.content
            safe_print(f"[VERBOSE] LLM generated {len(result)} characters of code")
            return result
        except Exception as e:
            safe_print(f"[ERROR] Error calling OpenAI: {e}")
            return ""

    @staticmethod
    def parse_natural_language_request(request: str) -> ProjectConfig:
        request_lower = request.lower()
        framework = Framework.FASTAPI
        if "flask" in request_lower:
            framework = Framework.FLASK
        elif "django" in request_lower:
            framework = Framework.DJANGO
        features = []
        if any(word in request_lower for word in ["auth", "login", "user", "jwt"]):
            features.append("authentication")
        if any(word in request_lower for word in ["test", "testing"]):
            features.append("testing")
        if any(word in request_lower for word in ["docker", "container"]):
            features.append("docker")
        if any(word in request_lower for word in ["db", "database", "sql", "mongo"]):
            features.append("database")
        database = "sqlite"
        if "postgres" in request_lower:
            database = "postgresql"
        elif "mysql" in request_lower:
            database = "mysql"
        elif "mongo" in request_lower:
            database = "mongodb"
        import re
        name_match = re.search(r'(?:call|name).*?([a-zA-Z][a-zA-Z0-9_-]*)', request_lower)
        project_name = name_match.group(1) if name_match else "generated-api"
        return ProjectConfig(
            name=project_name,
            framework=framework,
            features=features,
            database=database,
            auth="authentication" in features,
            testing="testing" in features or True,
            docker="docker" in features or True
        )

    def generate_project_structure(self, config: ProjectConfig) -> Path:
        safe_print("[VERBOSE] Creating project structure...")
        project_path = self.output_dir / config.name
        if project_path.exists():
            shutil.rmtree(project_path)
        if config.framework == Framework.FASTAPI:
            dirs = [
                "app", "app/routers", "app/models", "app/dependencies",
                "app/services", "app/core", "tests", "scripts"
            ]
        else:
            dirs = [
                "app", "app/routes", "app/models", "app/services",
                "app/utils", "tests", "migrations", "scripts"
            ]
        for dir_path in dirs:
            (project_path / dir_path).mkdir(parents=True, exist_ok=True)
            if "app" in dir_path:
                (project_path / dir_path / "__init__.py").touch()
        safe_print("[VERBOSE] Project folders created.")
        return project_path

    def create_project_files(self, project_path: Path, config: ProjectConfig, generated_code: str):
        safe_print("[VERBOSE] Writing requirements.txt, main.py, .env, Dockerfile...")
        if config.framework == Framework.FASTAPI:
            requirements = """fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.8.0
python-dotenv>=1.0.0
openai>=1.0.0
pytest>=7.4.3
httpx>=0.28.1
"""
            main_path = project_path / "main.py"
        else:
            requirements = """flask>=3.0.0
flask_sqlalchemy>=3.0.0
python-dotenv>=1.0.0
openai>=1.0.0
pytest>=7.4.3
"""
            main_path = project_path / "main.py"
        (project_path / "requirements.txt").write_text(requirements)
        (project_path / ".env").write_text("OPENAI_API_KEY=your-openai-key-here\n")
        main_path.write_text(generated_code)
        dockerfile = f"""FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
"""
        (project_path / "Dockerfile").write_text(dockerfile)
        safe_print("[VERBOSE] Core files written.")

    def run_docker_compose_and_tests(self, project_path: Path):
        compose_cmd = ["docker-compose", "up", "--build", "-d"]
        safe_print("[SANDBOX] Building and starting Docker Compose stack...")
        try:
            result = subprocess.run(compose_cmd, cwd=str(project_path), capture_output=True, text=True, timeout=120)
            safe_print("[SANDBOX] Docker Compose output:\n" + result.stdout)
            time.sleep(10)  # Give containers time to start
            url = "http://localhost:8000/docs"
            for _ in range(20):
                try:
                    r = requests.get(url, timeout=2)
                    if r.status_code == 200:
                        safe_print("[SANDBOX] API is up, running tests...")
                        break
                except Exception:
                    time.sleep(1)
            else:
                safe_print("[SANDBOX][FAIL] API health check did not pass.")
                return False
            test_cmd = ["docker-compose", "exec", "api", "pytest", "tests/"]
            test_result = subprocess.run(test_cmd, cwd=str(project_path), capture_output=True, text=True, timeout=60)
            safe_print("[SANDBOX] Pytest output:\n" + test_result.stdout)
            if test_result.returncode == 0:
                safe_print("[SANDBOX][PASS] All tests passed in Docker Compose sandbox.")
            else:
                safe_print("[SANDBOX][FAIL] Some tests FAILED in Docker Compose.")
            subprocess.run(["docker-compose", "down"], cwd=str(project_path))
        except Exception as e:
            safe_print(f"[SANDBOX][ERROR] Exception during Docker Compose sandbox: {e}")

def main():
    user_request = choose_project_idea()
    safe_print("[VERBOSE] Parsing user request...")
    config = AdvancedPythonBackendGenerator.parse_natural_language_request(user_request)
    safe_print(f"[VERBOSE] Parsed config: {config}")
    generator = AdvancedPythonBackendGenerator()
    project_path = generator.generate_project_structure(config)
    safe_print("[VERBOSE] Building LLM context...")
    context = generator.context_builder.build_context(user_request, config)
    safe_print("[VERBOSE] Calling LLM for code generation...")
    generated_code = generator.call_llm(context)
    safe_print("[VERBOSE] Creating all project files...")
    generator.create_project_files(project_path, config, generated_code)
    create_docker_compose(project_path, config)
    create_test_runner_scripts(project_path)
    safe_print("\n[SUCCESS] Project generated in: {}\nQuickstart:\n  cd {}\n  pip install -r requirements.txt\n  python main.py\n".format(
        project_path, project_path
    ))
    safe_print("[INFO] To run tests, use ./run_tests.sh (Linux/macOS) or run_tests.bat (Windows) inside the project folder.")
    safe_print("[INFO] To use Docker Compose: docker-compose up --build")
    generator.run_docker_compose_and_tests(project_path)

if __name__ == "__main__":
    main()
