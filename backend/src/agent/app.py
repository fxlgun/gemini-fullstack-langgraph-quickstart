# mypy: disable - error - code = "no-untyped-def,misc"
import asyncio
import pathlib
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
import threading
import time
import traceback
import os
import requests
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from google.cloud import storage
from langgraph_api.store import get_store
from google.oauth2 import service_account
import psycopg2
from psycopg2.extras import RealDictCursor
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

# Your existing config constants from environment or default values
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "dbname": os.environ.get("DB_NAME"),
}

cred_info = {
    "type": os.getenv("GOOGLE_TYPE"),
    "project_id": os.getenv("GOOGLE_PROJECT_ID"),
    "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
    "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
    "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("GOOGLE_UNIVERSE_DOMAIN"),
}

QUEUE_API = os.environ.get("QUEUE_API", "")
CHROMA_DB_FOLDER = os.environ.get("CHROMA_DB_FOLDER", "chrome_db_folder")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = 'land_parcels'
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
bucket_name = 'integrated_report_html'
credentials = service_account.Credentials.from_service_account_info(cred_info)
storage_client = storage.Client(credentials=credentials, project=cred_info["project_id"])
bucket = storage_client.bucket(bucket_name)


# Define the FastAPI app
app = FastAPI()


def create_frontend_router(build_dir="dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = build_dir

    # if not build_path.is_dir() or not (build_path / "index.html").is_file():
    #     print(
    #         f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
    #     )
    #     # Return a dummy router if build isn't ready
    #     from starlette.routing import Route

    #     async def dummy_frontend(request):
    #         return Response(
    #             "Frontend not built. Run 'npm run build' in the frontend directory.",
    #             media_type="text/plain",
    #             status_code=503,
    #         )

    #     return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/",
    create_frontend_router(),
    name="frontend",
)

from fastapi import Body

@app.post("/scrape")
async def scrape(body: Dict[str, Any] = Body(...)):
    required = {"dname", "tname", "vname", "vvalue"}
    if not required.issubset(body):
        raise HTTPException(status_code=400, detail="Missing required fields")

    rows = fetch_survey_rows(body["dname"], body["vname"], body["vvalue"])

    CLOUD_FUNCTION_URL = "https://handle-bullmq-585432733039.us-central1.run.app"
    job_opts = {
        "attempts": 5,
        "backoff": {"type": "exponential", "delay": 3000},
        "removeOnComplete": True,
        "removeOnFail": False,
    }

    payload = {
        "queueName": "integrated_report_html_queue",
        "jobName": "scrape-integrated-html",
        "batch": rows,
        "opts": job_opts,
    }

    try:
        response = requests.post(CLOUD_FUNCTION_URL, json=payload, timeout=65)
        if response.status_code == 200:
            return {"success": True, "queued_jobs": len(rows), "requested": len(rows), "details": response.text}
        else:
            raise HTTPException(status_code=500, detail=f"Cloud function error: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception contacting cloud function: {str(e)}")

@app.post("/embed")
async def embed(body: Dict[str, Any] = Body(...)):
    required = {"state", "dname", "vname", "vvalue"}
    if not required.issubset(body):
        raise HTTPException(status_code=400, detail="Missing required fields")

    store = await get_store()
    # Start embedding in background thread to not block response
    threading.Thread(
        target=embed_htmls_for_village,
        args=(body["state"], body["dname"], body["vname"], body["vvalue"], store),
        daemon=True,
    ).start()

    return JSONResponse(status_code=202, content={"success": True, "message": "Embedding started"})



def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def fetch_survey_rows(district, village, vvalue):
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT id, dname, dvalue, tname, tvalue, vname, vvalue, sname, svalue, visibility, lat, lng, geom
                FROM public.master_survey_rural
                WHERE dname=%s AND vname=%s AND vvalue=%s
            """
            cur.execute(sql, (district, village, vvalue))
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()

def embed_htmls_for_village(state, dname, vname, vvalue, store):
    print("Embedding started for:", vname)
    prefix = f"{state}/{vname.split('-')[0]}/"
    blobs = bucket.list_blobs(prefix=prefix)
    print("Fetched List of Blobs.", blobs)
    tenant_id = os.getenv("LANGGRAPH_TENANT_ID")  # store this in env
    print("Tenant ID:", tenant_id)
    for blob in blobs:
        blob_name = blob.name
        print("Downloading blob for Survey No.:", blob_name)
        html_data = blob.download_as_text()
        print("Downloaded blob:", blob_name)
        # Parse HTML to plain text
        soup = BeautifulSoup(html_data, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        store.put(
            namespace=(tenant_id, COLLECTION_NAME),
            key=blob_name,
            value=text
        )
        print("Embedded blob:", blob_name)

