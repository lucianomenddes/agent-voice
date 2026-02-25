import argparse
from typing import Generator, Tuple
from dotenv import load_dotenv
import os
import numpy as np
from fastrtc import ( # type: ignore
    AlgoOptions,
    ReplyOnPause,
    Stream,
    audio_to_bytes,
)
from groq import Groq
from loguru import logger

from process_groq_tts import process_groq_tts
from src.agent import agent, agent_config
from process_kokoro_tts import process_kokoro_tts

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)
load_dotenv()  # <-- isso carrega o arquivo .env

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def response(
    audio: tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input, transcribe it, generate a response using LangGraph, and deliver TTS audio.

    Args:
        audio: Tuple containing sample rate and audio data

    Yields:
        Tuples of (sample_rate, audio_array) for audio playback
    """
    logger.info("🎙️ Received audio input")

    logger.debug("🔄 Transcribing audio...")
    
    transcript = groq_client.audio.transcriptions.create(
        file=("audio-file.mp3", audio_to_bytes(audio)),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    logger.info(f'👂 Transcribed: "{transcript}"')

    logger.debug("🧠 Running agent...")
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": transcript}]}, config=agent_config
    )
    response_text = agent_response["messages"][-1].content
    logger.info(f'💬 Response: "{response_text}"')

    logger.debug("🔊 Generating speech...")
    
    # Usando Kokoro TTS local (Groq playai-tts foi descontinuado)
    yield from process_kokoro_tts(response_text)


def create_stream() -> Stream:
    """
    Create and configure a Stream instance with audio capabilities.

    Returns:
        Stream: Configured FastRTC Stream instance
    """
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.5,
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Agent")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()

    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
        logger.info("🌈 Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("🌈 Launching with Gradio UI...")
        stream.ui.launch()
