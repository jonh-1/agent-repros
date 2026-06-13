from livekit import api
from uuid import uuid4
from dotenv import load_dotenv
from os import getenv
import asyncio

load_dotenv(".env.local")

LIVEKIT_URL = getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = getenv("LIVEKIT_API_SECRET")
TRANSFER_NUMBER = getenv("TRANSFER_NUMBER")
AGENT_NUMBER = getenv("AGENT_NUMBER")

numbers = [f"+{TRANSFER_NUMBER}"]

async def campaign():
    async with api.LiveKitAPI(url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET) as lk:
        for number in numbers:
            room = await lk.room.create_room(api.CreateRoomRequest(
                name=f"campaign-{uuid4()}",
                empty_timeout=60,
            ))

            await lk.egress.start_room_composite_egress(api.RoomCompositeEgressRequest(
                room_name=room.name,
                audio_only=True,
                file_outputs=[
                    api.EncodedFileOutput(
                        filepath="{room_name}-{time}.ogg",
                        file_type=api.EncodedFileType.OGG,
                        s3=api.S3Upload(),
                    )
                ],
            ))

            await lk.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                participant_identity=str(uuid4()),
                participant_name="Callee",
                room_name=room.name,
                sip_call_to=number,
                wait_until_answered=True,
                sip_number=f"+{AGENT_NUMBER}",
                include_headers=api.SIPHeaderOptions.SIP_ALL_HEADERS,
                sip_trunk_id="ST_ZEAboiVYGHou",
            ))

            await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
                agent_name="main-agent-prod",
                room=room.name,
            ))

if __name__ == "__main__":
    asyncio.run(campaign())

