import sys
import argparse
import os

from dotenv import load_dotenv
from fastrtc import ReplyOnPause, Stream, get_stt_model
from loguru import logger
from openai import OpenAI

# Importe a KPipeline da biblioteca kokoro
from kokoro import KPipeline
import torch # Necessário para o tensor
import numpy as np # Necessário para o áudio

# -----------------------------------------------------------
# Configuração inicial
# -----------------------------------------------------------

load_dotenv()

try:
    openai_client = OpenAI()
    logger.info("Cliente OpenAI inicializado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao inicializar o cliente OpenAI. Verifique se a variável OPENAI_API_KEY está definida no seu arquivo .env. Detalhes: {e}")
    sys.exit(1)

# Classe para adaptar a KPipeline ao formato esperado pela fastrtc
class CustomTTSModel:
    def __init__(self, model_name="kokoro/brazilian-portuguese"):
        # Inicializa a KPipeline com o código de idioma 'p' para Português do Brasil
        self.pipeline = KPipeline(lang_code='p')
        self.voice = 'pf_dora'  # Defina a voz desejada aqui

    def stream_tts_sync(self, text):
        # A KPipeline retorna chunks com (gs, ps, audio)
        # Precisamos extrair apenas o áudio e formatá-lo como um tensor
        generator = self.pipeline(text, voice=self.voice)
        for i, (gs, ps, audio) in enumerate(generator):
            # Converte o array numpy para um tensor do PyTorch no formato correto
            audio_tensor = torch.from_numpy(np.array(audio, dtype=np.float32)).unsqueeze(0)
            yield audio_tensor

# Modelos locais de STT (Speech-to-Text) e TTS (Text-to-Speech)
stt_model = get_stt_model(model="moonshine/base")
tts_model = CustomTTSModel() # Use a sua classe adaptadora aqui

# Remove o logger padrão e configura um novo para exibir mensagens de depuração (debug).
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# -----------------------------------------------------------
# Lógica da aplicação
# -----------------------------------------------------------

def echo(audio):
    """
    Processa o áudio do usuário, transcreve para texto,
    envia para o GPT e devolve a resposta em áudio.
    """
    # ... o restante da sua função echo() pode permanecer o mesmo
    transcript = stt_model.stt(audio)
    logger.debug(f"Transcrição: {transcript}")
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Você é um LLM prestativo em uma chamada WebRTC. Seu objetivo é demonstrar suas habilidades de forma sucinta. Sua resposta será convertida em áudio, portanto, não inclua emojis ou caracteres especiais em suas respostas. Responda ao que o usuário disse de forma criativa e prestativa."
            },
            {"role": "user", "content": transcript},
        ],
    )
    
    response_text = response.choices[0].message.content
    logger.debug(f"Resposta do GPT: {response_text}")
    
    # A chamada para stream_tts_sync permanece a mesma, pois a sua classe a implementa
    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk

# ... o resto do código da aplicação é o mesmo
def create_stream():
    """Cria e retorna a stream de áudio com a lógica de resposta."""
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Avançado")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Inicia a interface de telefone FastRTC (para obter um número de telefone temporário)",
    )
    args = parser.parse_args()

    stream = create_stream()

    if args.phone:
        logger.info("Iniciando com a interface de telefone FastRTC...")
        stream.fastphone()
    else:
        logger.info("Iniciando com a interface Gradio UI...")
        stream.ui.launch()