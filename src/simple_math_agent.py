import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger

# Carrega variáveis do .env se existir
load_dotenv()

# Garante que a chave está no ambiente
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY não encontrada. Defina no .env ou exporte no ambiente.")


model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens=512,
    api_key=groq_api_key,
)


def sum_numbers(a: float, b: float) -> float:
    """Some dois números juntos."""
    result = a + b
    logger.info(f"➕ Calculating sum: {a} + {b} = {result}")
    return result


def multiply_numbers(a: float, b: float) -> float:
    """Multiplicar dois números entre si."""
    result = a * b
    logger.info(f"✖️ Calculating product: {a} × {b} = {result}")
    return result


tools = [sum_numbers, multiply_numbers]

system_prompt = """Você é Delb's, uma assistente de matemática prestativa e com uma personalidade acolhedora.
Você pode ajudar com operações matemáticas básicas usando suas ferramentas.
Sempre use as ferramentas quando solicitado a fazer cálculos matemáticos.
Sua saída será convertida em áudio, portanto, evite usar caracteres ou símbolos especiais.
Mantenha suas respostas amigáveis ​​e em tom de conversa em português Brasil.."""

memory = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

agent_config = {"configurable": {"thread_id": "default_user"}}
