import numpy as np
from typing import Any, Generator, Tuple
from fastrtc import KokoroTTSOptions, get_tts_model

# Inicializa o modelo TTS Kokoro
tts_model = get_tts_model(model="kokoro")

# Opções de voz PT-BR
tts_options = KokoroTTSOptions(
    lang="pt-br",
    voice="pf_dora",  # pode trocar por outra voz disponível
    speed=1.0,
)


def process_kokoro_tts(
    text: Any,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Processa texto em áudio usando Kokoro TTS (PT-BR).

    Args:
        text (str): Texto que será convertido em áudio

    Yields:
        Tupla (sample_rate, audio_array) para reprodução
    """
    for sample_rate, audio_array in tts_model.stream_tts_sync(
        text, options=tts_options
    ):
        yield (sample_rate, audio_array)
