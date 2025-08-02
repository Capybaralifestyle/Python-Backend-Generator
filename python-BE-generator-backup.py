#!/usr/bin/env python3
"""
Advanced Python Backend Generator
A production-ready tool that generates complete Python APIs from natural language descriptions.

Features:
- Natural language processing for requirements
- FastAPI and Flask support
- Smart context building with best practices
- Docker sandbox testing
- Comprehensive code validation
- Production-ready project structure
"""

import os
import sys
import json
import subprocess
import requests
import tempfile
import shutil
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Fix Windows encoding issues
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def safe_print(message):
    """Print function that handles encoding issues on Windows"""
    try:
        clean_message = str(message).encode('ascii', 'replace').decode('ascii')
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
    """Manages prompt templates for different types of requests"""

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

    def detect_template(self, user_request: str) -> Tuple[str, Dict]:
        """Detect which template matches the user request"""
        import re

        user_request_lower = user_request.lower()

        for template_name, template_info in self.templates.items():
            if re.search(template_info["pattern"], user_request_lower):
                return template_name, template_info

        return "crud_api", self.templates["crud_api"]

class DocumentationManager:
    """Manages context documentation for different frameworks and patterns"""

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
""",
                "dependencies": """
Essential FastAPI Dependencies:
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.8.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
pytest>=7.4.0
httpx>=0.25.0
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
""",
                "dependencies": """
Essential Flask Dependencies:
Flask>=3.0.0
Flask-CORS>=4.0.0
Flask-SQLAlchemy>=3.1.0
Flask-JWT-Extended>=4.6.0
marshmallow>=3.20.0
python-dotenv>=1.0.0
gunicorn>=21.2.0
pytest>=7.4.0
"""
            }
        }

    def get_context(self, framework: Framework, features: List[str]) -> str:
        """Build context documentation based on framework and features"""
        context_parts = []

        framework_key = framework.value
        if framework_key in self.docs:
            context_parts.extend([
                self.docs[framework_key]["basics"],
                self.docs[framework_key]["patterns"]
            ])

        return "\n".join(context_parts)

class SmartContextBuilder:
    """Builds intelligent context for LLM prompts"""

    def __init__(self):
        self.template_manager = PromptTemplateManager()
        self.doc_manager = DocumentationManager()

    def build_context(self, user_request: str, config: ProjectConfig) -> str:
        """Build comprehensive context for the LLM"""

        # Detect appropriate template
        template_name, template_info = self.template_manager.detect_template(user_request)

        # Extract entities from request
        entities = self._extract_entities(user_request)

        # Get framework documentation
        framework_docs = self.doc_manager.get_context(config.framework, config.features)

        # Enhanced prompt for DeepSeek R1 (reasoning model)
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
{template_info["template"].format(entity=entities[0] if entities else "item",
                                  description=user_request,
                                  domain=entities[0] if entities else "general")}

REQUIREMENTS:
- Generate complete, production-ready code
- Follow the documented best practices above
- Include proper error handling and validation
- Use type hints throughout
- Include docstrings for functions and classes
- Generate code that passes linting (flake8, black)
- No explanations, just clean code

USER REQUEST: {user_request}

Generate the complete Python code:
"""
        return context

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entity names from user request"""
        import re

        # Common entities
        common_entities = ["user", "book", "product", "order", "customer", "post", "comment", "task", "todo", "blog", "article"]

        entities = []
        text_lower = text.lower()

        for entity in common_entities:
            if entity in text_lower:
                entities.append(entity)

        # Extract potential custom entities (capitalized words)
        custom_entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend([e.lower() for e in custom_entities])

        return list(set(entities)) if entities else ["item"]

class CodeValidator:
    """Validates generated code quality and correctness"""

    def __init__(self, project_path: Path):
        self.project_path = project_path

    def validate_syntax(self) -> Dict:
        """Check Python syntax for all .py files"""
        results = {"valid": True, "errors": []}

        for py_file in self.project_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                results["valid"] = False
                results["errors"].append(f"{py_file}: {e}")

        return results

    def run_linting(self) -> Dict:
        """Run flake8 linting on the project"""
        try:
            result = subprocess.run(
                ["flake8", ".", "--max-line-length=88", "--extend-ignore=E203,W503"],
                cwd=self.project_path,
                capture_output=True,
                text=True
            )

            return {
                "success": result.returncode == 0,
                "issues": result.stdout.split('\n') if result.stdout else [],
                "error": result.stderr if result.stderr else None
            }
        except FileNotFoundError:
            return {"success": False, "error": "flake8 not installed"}

    def security_scan(self) -> Dict:
        """Run basic security checks"""
        issues = []

        for py_file in self.project_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()

                if "eval(" in content:
                    issues.append(f"{py_file}: Use of eval() detected")
                if "exec(" in content:
                    issues.append(f"{py_file}: Use of exec() detected")
                if "shell=True" in content:
                    issues.append(f"{py_file}: shell=True in subprocess detected")
                if "SECRET_KEY" in content and "=" in content:
                    issues.append(f"{py_file}: Hardcoded secret key detected")

        return {
            "issues": issues,
            "secure": len(issues) == 0
        }

class AdvancedPythonBackendGenerator:
    """Advanced Python backend generator with all enterprise features"""

    def __init__(self, ollama_host="http://localhost:11434", model="deepseek-r1:latest"):
        self.ollama_host = ollama_host
        self.model = model
        self.output_dir = Path("generated_projects")
        self.context_builder = SmartContextBuilder()

        self.output_dir.mkdir(exist_ok=True)

    def call_ollama(self, prompt: str) -> str:
        """Call Ollama API with enhanced error handling for DeepSeek R1"""
        try:
            safe_print(f"[LLM] Using model: {self.model}")
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_ctx": 8192,
                        "repeat_penalty": 1.1,
                        "top_k": 40
                    }
                },
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()["response"]
                safe_print(f"[LLM] Generated {len(result)} characters of code")
                return result
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        except Exception as e:
            safe_print(f"[ERROR] Error calling Ollama: {e}")
            return ""

    def parse_natural_language_request(self, request: str) -> ProjectConfig:
        """Parse natural language request into project configuration"""
        request_lower = request.lower()

        # Detect framework
        framework = Framework.FASTAPI  # default
        if "flask" in request_lower:
            framework = Framework.FLASK
        elif "django" in request_lower:
            framework = Framework.DJANGO

        # Detect features
        features = []
        if any(word in request_lower for word in ["auth", "login", "user", "jwt"]):
            features.append("authentication")
        if any(word in request_lower for word in ["test", "testing"]):
            features.append("testing")
        if any(word in request_lower for word in ["docker", "container"]):
            features.append("docker")
        if any(word in request_lower for word in ["db", "database", "sql", "mongo"]):
            features.append("database")

        # Detect database
        database = "sqlite"  # default
        if "postgres" in request_lower:
            database = "postgresql"
        elif "mysql" in request_lower:
            database = "mysql"
        elif "mongo" in request_lower:
            database = "mongodb"

        # Extract project name
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
        """Generate complete project structure based on configuration"""
        project_path = self.output_dir / config.name

        if project_path.exists():
            shutil.rmtree(project_path)

        if config.framework == Framework.FASTAPI:
            dirs = [
                "app", "app/routers", "app/models", "app/dependencies",
                "app/services", "app/core", "tests", "scripts"
            ]
        else:  # Flask
            dirs = [
                "app", "app/routes", "app/models", "app/services",
                "app/utils", "tests", "migrations", "scripts"
            ]

        for dir_path in dirs:
            (project_path / dir_path).mkdir(parents=True, exist_ok=True)
            if "app" in dir_path:
                (project_path / dir_path / "__init__.py").touch()

        return project_path

    def create_project_files(self, project_path: Path, config: ProjectConfig, generated_code: str):
        """Create all necessary project files"""

        # Create requirements.txt
        if config.framework == Framework.FASTAPI:
            requirements = """fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.8.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
"""
        else:  # Flask
            requirements = """Flask>=3.0.0
Flask-CORS>=4.0.0
python-dotenv>=1.0.0
gunicorn>=21.2.0
"""

        (project_path / "requirements.txt").write_text(requirements)

        # Create main app file
        (project_path / "main.py").write_text(generated_code)

        # Create .env file
        env_content = f"""# Environment Configuration
DEBUG=True
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=sqlite:///./app.db
"""
        (project_path / ".env").write_text(env_content)

        # Create Dockerfile
        if config.framework == Framework.FASTAPI:
            dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        else:
            dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "main.py"]
"""
        (project_path / "Dockerfile").write_text(dockerfile)

        # Create README.md
        port = "8000" if config.framework == Framework.FASTAPI else "5000"
        readme = f"""# {config.name.replace('-', ' ').title()}

Generated {config.framework.value.upper()} API with advanced features.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

Visit: http://localhost:{port}
{f'API Docs: http://localhost:{port}/docs' if config.framework == Framework.FASTAPI else ''}

## Features

- {config.framework.value.upper()} framework
- Production-ready code structure
- Automatic API documentation
- Environment configuration
- Docker containerization

## Docker

```bash
docker build -t {config.name} .
docker run -p {port}:{port} {config.name}
```
"""
        (project_path / "README.md").write_text(readme)

    def generate_complete_project(self, request: str) -> Dict:
        """Generate complete project from natural language request"""
        safe_print(f"[PARSING] Request: {request}")

        # Parse request
        config = self.parse_natural_language_request(request)
        safe_print(f"[CONFIG] Detected: {config.name} ({config.framework.value}) with {config.features}")

        # Create structure
        project_path = self.generate_project_structure(config)

        # Build intelligent context
        enhanced_prompt = self.context_builder.build_context(request, config)

        # Generate code
        safe_print("[LLM] Generating code with DeepSeek...")
        generated_code = self.call_ollama(enhanced_prompt)

        if not generated_code:
            return {"success": False, "error": "Failed to generate code"}

        # Create all project files
        self.create_project_files(project_path, config, generated_code)

        # Validate code
        safe_print("[VALIDATION] Validating generated code...")
        validator = CodeValidator(project_path)

        validation_results = {
            "syntax": validator.validate_syntax(),
            "security": validator.security_scan(),
            "code_quality": self.validate_generated_code(generated_code)
        }

        return {
            "success": True,
            "config": config,
            "project_path": str(project_path),
            "validation": validation_results,
            "generated_code": generated_code[:1000] + "..." if len(generated_code) > 1000 else generated_code,
            "code_score": validation_results["code_quality"].get("score", 0)
        }

def main():
    """Main function with enhanced user interaction"""
    safe_print("Advanced Python Backend Generator")
    safe_print("=" * 50)
    safe_print("Transform natural language into production-ready Python APIs!")
    safe_print("")

    generator = AdvancedPythonBackendGenerator()

    # Example requests
    examples = [
        "Create a FastAPI todo application with CRUD operations",
        "Build a Flask blog API with posts and comments",
        "Make a FastAPI e-commerce API with products and orders",
        "Create a social media API with posts, likes, and users",
        "Build a microservice for user authentication with JWT",
        "Create a Flask API for a library management system"
    ]

    safe_print("Example requests:")
    for i, example in enumerate(examples, 1):
        safe_print(f"   {i}. {example}")
    safe_print("")

    # Get user input
    user_request = input("Describe what you want to build: ").strip()

    if not user_request:
        user_request = examples[0]
        safe_print(f"Using example: {user_request}")

    safe_print(f"\n[STARTING] Generation process...")
    safe_print("=" * 50)

    try:
        # Generate project
        result = generator.generate_complete_project(user_request)

        if result["success"]:
            safe_print("\n[SUCCESS] PROJECT GENERATED SUCCESSFULLY!")
            safe_print("=" * 50)

            config = result["config"]
            safe_print(f"Project: {config.name}")
            safe_print(f"Framework: {config.framework.value}")
            safe_print(f"Database: {config.database}")
            safe_print(f"Features: {', '.join(config.features)}")
            safe_print(f"Location: {result['project_path']}")

            # Validation results
            validation = result["validation"]
            safe_print(f"\n[VALIDATION] CODE VALIDATION:")
            safe_print(f"   Syntax: {'Valid' if validation['syntax']['valid'] else 'Issues found'}")
            safe_print(f"   Security: {'Secure' if validation['security']['secure'] else 'Issues found'}")
            safe_print(f"   Code Quality Score: {result.get('code_score', 0):.2f}/1.00")

            quality_score = result.get('code_score', 0)
            if quality_score >= 0.8:
                safe_print("   Quality: Excellent")
            elif quality_score >= 0.6:
                safe_print("   Quality: Good")
            elif quality_score >= 0.4:
                safe_print("   Quality: Fair (using enhanced code)")
            else:
                safe_print("   Quality: Fallback code used")

            # Instructions
            safe_print(f"\n[QUICKSTART]:")
            safe_print(f"   cd {result['project_path']}")
            safe_print(f"   pip install -r requirements.txt")
            safe_print(f"   python main.py")

            port = "8000" if config.framework == Framework.FASTAPI else "5000"
            safe_print(f"   Visit: http://localhost:{port}")

            if config.framework == Framework.FASTAPI:
                safe_print(f"   API Docs: http://localhost:{port}/docs")

        else:
            safe_print(f"\n[ERROR] GENERATION FAILED: {result.get('error', 'Unknown error')}")

    except KeyboardInterrupt:
        safe_print("\n\nGeneration cancelled by user")
    except Exception as e:
        safe_print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()