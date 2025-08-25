# mypy: disable - error - code = "no-untyped-def,misc"
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
import threading
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
import firebase_admin
from firebase_admin import auth as fb_auth, credentials, storage as fb_storage
import httpx
from google import genai
import json
import time

FAILED_LOG = "failed_ingestion.log"

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "upin": {"type": "STRING"},
        "state": {"type": "STRING"},
        "district": {"type": "STRING"},
        "taluka": {"type": "STRING"},
        "village": {"type": "STRING"},
        "survey_no": {"type": "STRING"},
        "old_survey_number": {"type": "STRING"},
        "zone": {"type": "STRING"},
        "old_survey_notes": {"type": "STRING"},
        "authority": {"type": "STRING"},
        "promulgation_detail": {
            "type": "OBJECT",
            "properties": {
                "note_number": {"type": "STRING"},
                "date": {"type": "STRING"},
                "order_no": {"type": "STRING"},
                "order_date": {"type": "STRING"},
                "officer": {"type": "STRING"},
                "resurvey_range": {"type": "STRING"},
                "pages": {"type": "INTEGER"}
            }
        },
        "area_sq_mt": {
            "type": "STRING"
        },
        "land_use": {"type": "STRING"},
        "tenure": {"type": "STRING"},
        "assessment": {"type": "STRING"},
        "tax": {"type": "STRING"},
        "jantri_value": {"type": "STRING"},
        "khata_number": {"type": "STRING"},
        "farm_name": {"type": "STRING"},
        "owners": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "relation": {"type": "STRING"},
                    "remarks": {"type": "STRING"}
                }
            }
        },
        "mutations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "entry_no": {"type": "STRING"},
                    "date": {"type": "STRING"},
                    "type": {"type": "STRING"},
                    "status": {"type": "STRING"},
                    "document_no": {"type": "STRING"}
                }
            }
        },
        "revenue_cases": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "office": {"type": "STRING"},
                    "survey_no": {"type": "STRING"},
                    "case_no": {"type": "STRING"},
                    "status": {"type": "STRING"},
                    "type": {"type": "STRING"},
                    "remarks": {"type": "STRING"},
                    "filing_date": {"type": "STRING"}
                }
            }
        },
        "encumbrances": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING"},
                    "details": {"type": "STRING"},
                    "date": {"type": "STRING"}
                }
            }
        },
        "deed_number": {"type": "STRING"},
        "deed_date": {"type": "STRING"},
        "deed_type": {"type": "STRING"},
        "is_disputed": {"type": "BOOLEAN"},
        "encumbrance_certificate": {"type": "STRING"},
        "last_updated": {"type": "STRING"},
        "certified_by": {"type": "STRING"},
        "disclaimer": {"type": "STRING"}
    }
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

# Your existing config constants from environment or default values
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "dbname": os.environ.get("DB_NAME"),
}

cred = credentials.Certificate(cred_info)
if not firebase_admin._apps:  # only init once
    firebase_admin.initialize_app(cred)
CHROMA_DB_FOLDER = os.environ.get("CHROMA_DB_FOLDER", "chrome_db_folder")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = 'land_parcels'
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
bucket_name = 'integrated_report_html'
credentials = service_account.Credentials.from_service_account_info(cred_info)
storage_client = storage.Client(credentials=credentials, project=cred_info["project_id"])
bucket = fb_storage.bucket('townplanmap.appspot.com')




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
    prefix = f"{bucket_name}/{state}/{vname.split('-')[0]}/"
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

def fuzzy_survey_exists(district, village):
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT id
                FROM public.survey_layer
                WHERE similarity(district, %s) > 0.3
	            AND similarity(village, %s) > 0.3
            """
            cur.execute(sql, (district, village))
            rows = cur.fetchall()
            if rows:
                return rows[0]
            else:
                return False
    except Exception as e:
        print("Error:", e)
        return None
    finally:
        conn.close()

def fuzzy_survey_id(district, village, survey_no):
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT id
                FROM public.survey_layer
                WHERE district = %s
	            AND village = %s
                AND similarity(survey_no, %s) > 0.5
            """
            cur.execute(sql, (district, village, survey_no))
            rows = cur.fetchall()
            if rows:
                return rows[0]
            else:
                return False
    except Exception as e:
        print("Error:", e)
        return None
    finally:
        conn.close()
    

def fuzzy_area_id(report):
    village = report["village"]
    district = report["district"]
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT area_id
                FROM public.area_data
                WHERE similarity(district, %s) > 0.3
	            AND similarity(name, %s) > 0.3
            """
            cur.execute(sql, (district, village))
            rows = cur.fetchall()
            if rows:
                return rows[0]["area_id"]
            else:
                return None
    finally:
        conn.close()
        
def insert_survey_layer(conn, report, area_id):
    sql = """
        INSERT INTO public.survey_layer (
            upin, state, district, taluka, village, survey_no,
            old_survey_number, old_survey_notes, zone, authority,
            promulgation_detail, area_sq_mt, land_use, tenure,
            assessment, tax, jantri_value, khata_number, farm_name,
            owners, mutations, revenue_cases, encumbrances,
            deed_number, deed_date, deed_type, is_disputed,
            encumbrance_certificate, last_updated, certified_by,
            disclaimer, area_id, uid
        )
        VALUES (%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            report.get("upin"),
            report.get("state").upper(),
            report.get("district").upper(),
            report.get("taluka").upper(),
            report.get("village").upper(),
            report.get("survey_no"),
            report.get("old_survey_number"),
            report.get("old_survey_notes"),
            report.get("zone"),
            report.get("authority"),
            json.dumps(report.get("promulgation_detail")),
            report.get("area_sq_mt"),
            report.get("land_use"),
            report.get("tenure"),
            report.get("assessment"),
            report.get("tax"),
            report.get("jantri_value"),
            report.get("khata_number"),
            report.get("farm_name"),
            json.dumps(report.get("owners")),
            json.dumps(report.get("mutations")),
            json.dumps(report.get("revenue_cases")),
            json.dumps(report.get("encumbrances")),
            report.get("deed_number"),
            report.get("deed_date"),
            report.get("deed_type"),
            report.get("is_disputed"),
            report.get("encumbrance_certificate"),
            report.get("last_updated"),
            report.get("certified_by"),
            report.get("disclaimer"),
            area_id,
            report.get("village", "").upper() + '_' + str(report.get("survey_no"))
        ))
    conn.commit()
    print(f"Inserted new survey for survey_no={report.get('survey_no')}")


def update_survey_layer(conn, survey_id, report, area_id):
    """Update existing survey row with new fields if available"""
    sql = """
        UPDATE public.survey_layer
        SET upin = COALESCE(%s, upin),
            old_survey_notes = COALESCE(%s, old_survey_notes),
            promulgation_detail = COALESCE(%s, promulgation_detail),
            assessment = COALESCE(%s, assessment),
            tax = COALESCE(%s, tax),
            jantri_value = COALESCE(%s, jantri_value),
            khata_number = COALESCE(%s, khata_number),
            farm_name = COALESCE(%s, farm_name),
            owners = COALESCE(%s, owners),
            mutations = COALESCE(%s, mutations),
            revenue_cases = COALESCE(%s, revenue_cases),
            encumbrances = COALESCE(%s, encumbrances),
            deed_number = COALESCE(%s, deed_number),
            deed_date = COALESCE(%s, deed_date),
            deed_type = COALESCE(%s, deed_type),
            is_disputed = COALESCE(%s, is_disputed),
            encumbrance_certificate = COALESCE(%s, encumbrance_certificate),
            last_updated = COALESCE(%s, last_updated),
            certified_by = COALESCE(%s, certified_by),
            disclaimer = COALESCE(%s, disclaimer),
            area_id = COALESCE(%s, area_id)
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            report.get("upin"),
            report.get("old_survey_notes"),
            json.dumps(report.get("promulgation_detail")),
            report.get("assessment"),
            report.get("tax"),
            report.get("jantri_value"),
            report.get("khata_number"),
            report.get("farm_name"),
            json.dumps(report.get("owners")),
            json.dumps(report.get("mutations")),
            json.dumps(report.get("revenue_cases")),
            json.dumps(report.get("encumbrances")),
            report.get("deed_number"),
            report.get("deed_date"),
            report.get("deed_type"),
            report.get("is_disputed"),
            report.get("encumbrance_certificate"),
            report.get("last_updated"),
            report.get("certified_by"),
            report.get("disclaimer"),
            area_id,
            survey_id
        ))
    conn.commit()
    print(f"Updated survey_layer.id={survey_id}")



def enforce_schema(report: dict) -> dict:
    """Ensure report has all keys from RESPONSE_SCHEMA, fill with None/[] as needed."""
    def fill(schema, data):
        if schema["type"] == "OBJECT":
            res = {}
            props = schema.get("properties", {})
            for k, v in props.items():
                res[k] = fill(v, data.get(k) if isinstance(data, dict) else None)
            return res
        elif schema["type"] == "ARRAY":
            return data if isinstance(data, list) else []
        else:
            return data if data not in (None, "") else None
    return fill(RESPONSE_SCHEMA, report or {})

def log_failure(blob_name, reason):
    """Append failed blob info to a logfile for later reprocessing"""
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "blob": blob_name,
            "reason": reason,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")
        }, ensure_ascii=False) + "\n")


def extract_report_with_retries(text, retries=3, delay=10):
    """
    Call Gemini with schema enforcement + retries.
    Always returns a schema-complete dict (even if empty).
    """
    client = genai.Client()
    prompt = (
        "Extract all relevant fields from this Gujarat integrated land record HTML, "
        "transliterating them into English. "
        "IMPORTANT: Always output all fields from the schema, even if null.\n\n"
        f"{text}"
    )

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": RESPONSE_SCHEMA
                },
            )
            report = response.parsed or {}
            report = enforce_schema(report)

            # Require at least survey_no + village
            if report.get("survey_no") and report.get("village"):
                print(f"[Retry {attempt}] Extracted report: {report}")
                return report
            else:
                print(f"[Retry {attempt}] Missing critical fields, retrying...")
        except Exception as e:
            print(f"[Retry {attempt}] Gemini error: {e}")

        time.sleep(delay)

    print("❌ Failed to extract after retries.")
    return enforce_schema({})  # return schema with nulls

def survey_uid_exists(conn, uid):
    sql = "SELECT id FROM public.survey_layer WHERE uid = %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (uid,))
        row = cur.fetchone()
        return row["id"] if row else None


def extract_report_with_retries(text, retries=3, delay=2):
    """Try Gemini extraction with retry in case of missing fields or errors"""
    client = genai.Client()
    prompt = (
        "Extract all relevant fields from this Gujarat integrated land record HTML "
        "transliterating them in English and fill them into the defined JSON schema. "
        "IMPORTANT: Always output *all fields* from the schema, even if null.\n\n"
        f"{text}"
    )

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": RESPONSE_SCHEMA
                },
            )
            report = response.parsed or {}
            # Require at least core fields
            if report.get("survey_no") and report.get("village"):
                return report
            else:
                print(f"[Retry {attempt}] Missing critical fields, retrying...")
        except Exception as e:
            print(f"[Retry {attempt}] Extraction error: {e}")
        time.sleep(delay)

    print("❌ Failed to extract after retries.")
    return None


def add_to_Postgis(state, dname, vname, vvalue, force_update=False, resume_from=None):
    print("Adding to Postgis started for:", vname)
    prefix = f"{bucket_name}/{state}/{vname.split('-')[0]}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"Fetched {len(blobs)} blobs.")
    skip_mode = True if resume_from else False


    conn = get_db_conn()   # open one conn for the batch
    try:
        for blob in blobs:
            blob_name = blob.name
            if skip_mode:
                if blob_name == resume_from:
                    print(f"✅ Resuming from {blob_name}")
                    skip_mode = False
                else:
                    print(f"⏭️ Skipping {blob_name} until resume_from={resume_from}")
                    continue
            print("Downloading blob for Survey No.:", blob_name)
            html_data = blob.download_as_text()
            soup = BeautifulSoup(html_data, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            # === Extract structured JSON with retry ===
            report = extract_report_with_retries(text)
            if not report or not report.get("survey_no"):
                log_failure(blob_name, "Extraction failed or missing survey_no")
                continue

            # === Post-processing ===
            report['state'] = state
            report['area_id'] = fuzzy_area_id(report)
            uid = report.get("village", "").upper() + "_" + str(report.get("survey_no"))
            report["uid"] = uid

            try:
                existing_id = survey_uid_exists(conn, uid)
                if existing_id:
                    if force_update:
                        print(f"Updating existing UID={uid} (id={existing_id})")
                        update_survey_layer(conn, existing_id, report, report['area_id'])
                    else:
                        print(f"Skipping UID={uid}, already exists (id={existing_id})")
                        continue
                else:
                    insert_survey_layer(conn, report, report['area_id'])
            except Exception as e:
                log_failure(blob_name, f"DB error: {e}")
                print(f"❌ DB error for UID={uid}: {e}")
    finally:
        conn.close()

    return True


TASKS = {
    "embed_htmls": embed_htmls_for_village,
    "add_to_Postgis": add_to_Postgis,

}

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
        async with httpx.AsyncClient(timeout=65) as client:
            response = await client.post(CLOUD_FUNCTION_URL, json=payload)
            if response.status_code == 200:
                return {"success": True, "queued_jobs": len(rows), "requested": len(rows), "details": response.text}
            else:
                raise HTTPException(status_code=500, detail=f"Cloud function error: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception contacting cloud function: {str(e)}")

@app.post("/task")
async def embed(body: Dict[str, Any] = Body(...)):
    if "task" not in body:
        raise HTTPException(status_code=400, detail="Missing 'task' in payload")

    task_name = body["task"]
    args = body.get("args", {})

    if task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")
    
    store = None
    if task_name == "embed_htmls":
        store = await get_store()
        args["store"] = store
        
    # Start embedding in background thread to not block response
    threading.Thread(
        target=TASKS[task_name],
        kwargs=args,
        daemon=True,
    ).start()

    return JSONResponse(status_code=202, content={"success": True, "message": f"Started {task_name}"})
