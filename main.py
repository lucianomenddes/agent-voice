import sys
import argparse
import os

from dotenv import load_dotenv
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from fastrtc import KokoroTTSOptions # Importa a classe de opções
from loguru import logger
from openai import OpenAI

# -----------------------------------------------------------
# Configuração inicial
# -----------------------------------------------------------

# Carrega as variáveis de ambiente do arquivo .env. 
load_dotenv()

try:
    openai_client = OpenAI()
    logger.info("Cliente OpenAI inicializado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao inicializar o cliente OpenAI: {e}")
    sys.exit(1)

# Modelos locais de STT (Speech-to-Text) e TTS (Text-to-Speech)
# Para um melhor desempenho em português, você pode tentar `get_stt_model(model="moonshine/brazilian-portuguese")` se disponível.
# O modelo "moonshine/base" é multi-idioma e deve funcionar.
stt_model = get_stt_model(model="moonshine/base")

# Inicializa o modelo TTS para português do Brasil
tts_model = get_tts_model(model="kokoro")

# -----------------------------------------------------------
# Opções de idioma e voz para o TTS
# -----------------------------------------------------------
# Você pode escolher outras vozes em português que a biblioteca oferece.
# A documentação da Kokoro onnx pode ter uma lista.
tts_options = KokoroTTSOptions(
    lang="pt-br",
    voice="pf_dora",
    speed=1.0
)

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
    # 1. Transcreve o áudio do usuário para texto
    # Dependendo da versão do fastrtc, você pode precisar passar o idioma do stt aqui
    transcript = stt_model.stt(audio)
    logger.debug(f"Transcrição: {transcript}")
    
    # 2. Envia a transcrição para a API do GPT para gerar uma resposta
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Você é um LLM prestativo em uma chamada WebRTC. Seu objetivo é demonstrar suas habilidades de forma sucinta. Sua resposta será convertida em áudio, portanto, não inclua emojis ou caracteres especiais em suas respostas. Responda ao que o usuário disse de forma criativa e prestativa em português do Brasil."
            },
            {"role": "user", "content": transcript},
        ],
    )
    
    # 3. Extrai o texto da resposta do GPT
    response_text = response.choices[0].message.content
    logger.debug(f"Resposta do GPT: {response_text}")
    
    # 4. Converte o texto da resposta em áudio e envia para o stream,
    #    passando as opções de idioma e voz.
    for audio_chunk in tts_model.stream_tts_sync(response_text, options=tts_options):
        yield audio_chunk

def create_stream():
    """Cria e retorna a stream de áudio com a lógica de resposta."""
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")

# -----------------------------------------------------------
# Ponto de entrada da aplicação
# -----------------------------------------------------------

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