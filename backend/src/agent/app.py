# mypy: disable - error - code = "no-untyped-def,misc"
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

import psycopg2
from psycopg2.extras import RealDictCursor
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

# Your existing config constants from environment or default values
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "dbname": os.environ.get("DB_NAME"),
}
QUEUE_API = os.environ.get("QUEUE_API", "")
CHROMA_DB_FOLDER = os.environ.get("CHROMA_DB_FOLDER", "chrome_db_folder")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = 'land_parcels'


# Define the FastAPI app
app = FastAPI()


def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
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

    # Start embedding in background thread to not block response
    threading.Thread(
        target=embed_htmls_for_village,
        args=(body["state"], body["dname"], body["vname"], body["vvalue"]),
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

def add_to_scrape_queue(survey_row: dict):
    enqueue_url = f"{QUEUE_API}/enqueue"
    try:
        response = requests.post(enqueue_url, json=survey_row, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def embed_htmls_for_village(state, dname, vname, vvalue):
    from google.cloud import storage
    storage_client = storage.Client.from_service_account_json("townplanmap.json")
    bucket = storage_client.bucket("integrated_report_html")
    prefix = f"{state}/{vname.split('-')[0]}/"
    blobs = bucket.list_blobs(prefix=prefix)
    file_list = [blob.name for blob in blobs if blob.name.endswith('.html')]

    client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)

    docs, ids, meta = [], [], []
    model = SentenceTransformer(EMBED_MODEL_NAME)
    for i, blob_name in enumerate(file_list):
        blob = bucket.blob(blob_name)
        html = blob.download_as_text()
        print(f"Downloaded {blob_name}")
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)

        docs.append(text)
        ids.append(f"{blob_name}_{i}_{int(time.time())}")
        meta.append({'filename': blob_name, 'village': vname, 'district': dname})
        print(f"Processed {blob_name}")
        
    print(f"Total documents processed: {len(docs)}. Starting to Encode Embeddings")
    embeddings = model.encode(docs, show_progress_bar=True).tolist()
    collection.add(documents=docs, embeddings=embeddings, metadatas=meta, ids=ids)
    print(f"Embeddings added to collection. Total documents in collection: {collection.count()}")

