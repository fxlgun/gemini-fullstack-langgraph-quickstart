import firebase_admin
from firebase_admin import auth as fb_auth, credentials
from fastapi import Request, HTTPException
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

cred = credentials.Certificate(cred_info)
firebase_admin.initialize_app(cred)

from langgraph_sdk import Auth

lg_auth = Auth()

@lg_auth.authenticate
def authenticate(request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing auth")
    id_token = auth_header.split(" ", 1)[1]
    decoded = fb_auth.verify_id_token(id_token)
    # choose returned identity: phone_number (if present) or uid
    identity = decoded.get("phone_number")
    return {"identity": identity, "claims": decoded}

@lg_auth.on
async def add_owner(ctx: Auth.types.AuthContext, value: dict):
    """Make resources private to their creator."""
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
