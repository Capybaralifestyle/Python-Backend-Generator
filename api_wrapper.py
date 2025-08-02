from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import uuid
import shutil
from pathlib import Path

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: set specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATOR_PATH = "python-BE-generator.py"
OUTPUT_BASE = Path("generated_projects")

@app.post("/generate")
async def generate_backend(req: Request):
    data = await req.json()
    idea = data.get("idea")

    # Prepare environment
    session_id = str(uuid.uuid4())
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Run the script with user input piped into stdin
    process = subprocess.Popen(
        ["python", GENERATOR_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout, stderr = process.communicate(input=idea)

    # Look for generated folder
    project_name = "generated-api"
    for folder in os.listdir(OUTPUT_BASE):
        if folder.startswith("generated"):
            project_name = folder
            break

    project_path = OUTPUT_BASE / project_name

    return {
        "output": stdout[-3000:],  # Limit response
        "error": stderr,
        "project": project_name,
        "download_url": f"/download/{project_name}.zip"
    }

@app.get("/download/{zipfile}")
async def download_zip(zipfile: str):
    zip_path = Path("archives") / zipfile
    if zip_path.exists():
        return FileResponse(str(zip_path), media_type='application/zip', filename=zipfile)
    return {"error": "File not found"}

# Archive logic (called in background or post-process)
def zip_project(project_name):
    output_path = OUTPUT_BASE / project_name
    archive_path = Path("archives") / f"{project_name}.zip"
    shutil.make_archive(str(archive_path).replace(".zip", ""), 'zip', output_path)
