import os
from fastapi import FastAPI, HTTPException, Request
from livekit.api import WebhookReceiver, TokenVerifier
from dotenv import load_dotenv
import datetime
import jwt

load_dotenv(".env.local")


app = FastAPI()

verifier = TokenVerifier(
    os.getenv("LIVEKIT_API_KEY"),
    os.getenv("LIVEKIT_API_SECRET"),
    leeway=datetime.timedelta(minutes=5)
)
receiver = WebhookReceiver(verifier)

@app.post("/webhook")
async def webhook(request: Request):
    auth = request.headers.get("Authorization")
    payload = jwt.decode(auth, options={"verify_signature": False})
    print(f"iss from webhook: {payload.get('iss')}")
    
    if not auth:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    body = (await request.body()).decode()
    event = receiver.receive(body, auth)
    print(event)
    return {"status": "success"}
