import asyncio
import logging
import uuid
from os import getenv

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    get_job_context,
    inference,
    stt,
)
from livekit.plugins import silero

load_dotenv(".env.local")

logger = logging.getLogger("agent")

TRANSFER_NUMBER = getenv("TRANSFER_NUMBER")
SIP_TRUNK_ID = getenv("SIP_TRUNK_ID")

TRANSFEREE_IDENTITY = "transferee"


async def transcribe_transferee(
    room: rtc.Room,
    track: rtc.Track,
    pub: rtc.RemoteTrackPublication,
    participant: rtc.RemoteParticipant,
) -> None:
    logger.info("starting transferee transcription for %s", participant.identity)

    stt_instance = inference.STT(model="deepgram/nova-3")

    async with stt_instance.stream() as stream:
        async def push_frames() -> None:
            async for frame_event in rtc.AudioStream(track):
                stream.push_frame(frame_event.frame)

        push_task = asyncio.ensure_future(push_frames())

        try:
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text
                    logger.info("[transferee %s]: %s", participant.identity, text)
                else:
                    continue

                if not text:
                    continue
        finally:
            push_task.cancel()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.

            ## Output rules
            - Respond in plain text only. Never use markdown, lists, or formatting.
            - Do not be overly wordy.
            """,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user and ask how you can help.",
            allow_interruptions=True,
        )

    @function_tool
    async def transfer_call(self, context: RunContext) -> str:
        """Transfer the caller to another number. Call this when the user requests a transfer."""
        job_ctx = get_job_context()
        room = job_ctx.room

        await self.session.say("Please hold while I connect you.", allow_interruptions=False)

        try:
            await job_ctx.api.room.update_participant(
                api.UpdateParticipantRequest(
                    room=room.name,
                    identity=room.local_participant.identity,
                    permission=api.ParticipantPermission(
                        can_publish=False,
                        can_subscribe=True,
                    ),
                )
            )
        except api.TwirpError as e:
            logger.warning("could not strip agent permissions: %s", e)
            return "transfer failed — could not update permissions"

        await job_ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=SIP_TRUNK_ID,
                sip_call_to=TRANSFER_NUMBER,
                room_name=room.name,
                participant_identity=TRANSFEREE_IDENTITY,
                wait_until_answered=True,
            )
        )

        # prevents the LLM from processing audio after the transfer
        context.session.interrupt()
        context.session.input.set_audio_enabled(False)

        logger.info("transferee joined room %s", room.name)
        return "transfer initiated"


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="transferee-transcription-repro")
async def agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            track.kind != rtc.TrackKind.KIND_AUDIO
            or participant.identity != TRANSFEREE_IDENTITY
        ):
            return
        asyncio.ensure_future(transcribe_transferee(ctx.room, track, pub, participant))

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3"),
        llm=inference.LLM(model="google/gemini-2.5-flash"),
        tts=inference.TTS(model="cartesia/sonic-3"),
        vad=ctx.proc.userdata["vad"],
        allow_interruptions=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
