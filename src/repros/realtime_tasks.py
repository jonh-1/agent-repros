import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    RunContext,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.plugins import openai, phonic

load_dotenv(".env.local")

logger = logging.getLogger(__name__)


class MathTask(AgentTask[None]):
    def __init__(self, question: str, answer: int) -> None:
        logger.info(f"Starting MathTask with question: {question} and answer: {answer}")
        self._question = question
        self._answer = answer
        super().__init__(
            instructions=f"""\
Ask the user to solve the following problem: "{question}"
Listen to their response and call question_attempt with their answer in numeric form.
""",
        )
        logger.info(f"MathTask initialized")

    async def on_enter(self) -> None:
        logger.info(f"MathTask on_enter")
        await self.session.generate_reply(
            instructions=f'Ask the user to solve this problem: "{self._question}"'
        )
        logger.info(f"MathTask on_enter complete")

    @function_tool()
    async def question_attempt(self, ctx: RunContext, guess: int) -> str:
        logger.info(f"MathTask question_attempt: {guess}")
        """Called with whatever the user said in response to the phrase prompt.

        Args:
            guess: Number the user said
        """
        logger.info("question_attempt: %r (target: %r)", guess, self._answer)

        """
        if self._answer == guess:
            ctx.speech_handle.add_done_callback(lambda _: self.complete(None) if not self.done() else None)
            return "The user answered correctly. Briefly congratulate them before moving on."

            logger.info(f"MathTask question_attempt correct, generating reply")
            self.session.generate_reply(instructions="The user answered the question correctly. Briefly congrautate them and move on.")
            logger.info(f"MathTask generated reply")
            self.complete(None)
            
        
        else:
            return f'The user said: "{guess}". Ask them to try again.'
        """

        
        if self._answer == guess:
            if not self.done():

                def _complete_after_praise(_: object) -> None:
                    if not self.done():
                        self.complete(None)

                ctx.speech_handle.add_done_callback(_complete_after_praise)

            return (
                "The user repeated the phrase correctly. "
                "Briefly praise them in a warm, natural way. Don't ask if they need help with anything else."
            )

        return f'The user said: "{guess}". Encourage them and ask them to try again.'


class PhraseCheckAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""\
You are asking the user to solve basic math problems.
After all questions are complete, let them know they have finished and wrap up warmly.
""",
        )

    async def on_enter(self) -> None:
        questions = [
            {"question": "2 + 2", "answer": 4},
            {"question": "3 + 3", "answer": 6},
            {"question": "4 + 4", "answer": 8},
        ]

        for q in questions:
            await MathTask(q["question"], q["answer"])

        await self.session.generate_reply(
            instructions="Let the user know they have completed all the questions and congratulate them."
        )
        self.session.shutdown()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        llm=phonic.realtime.RealtimeModel(voice="sabrina"),
        # llm=openai.realtime.RealtimeModel(voice="marin"),
    )

    await session.start(
        agent=PhraseCheckAgent(),
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
