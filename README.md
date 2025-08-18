# 🤖 Agent Voice: Seu Assistente de Voz Local

Bem-vindo ao **Agent Voice**, um projeto de assistente de voz que combina modelos de IA de ponta para oferecer uma experiência de conversação fluida e responsiva.  
Este projeto utiliza o poder da **API da OpenAI (modelo GPT-3.5 Turbo)** e os modelos de fala locais **Kokoro** e **Moonshine** para processar comandos de voz e gerar respostas de forma ágil e eficiente.

---

## 🌟 Recursos Principais

- ⚡️ **Respostas Ágeis**: Graças à API da **OpenAI (GPT-3.5 Turbo)**, as respostas do modelo de linguagem são geradas em tempo real.  
- 🗣️ **Conversação em Português do Brasil**: O projeto usa o modelo **Kokoro** para síntese de voz em português e o **Moonshine** para reconhecimento de fala, garantindo uma interação natural.  
- 💻 **Execução Local**: A maior parte do processamento, incluindo a transcrição de voz (STT) e a geração de áudio (TTS), ocorre no seu sistema.  
- 🔌 **Integração com fastrtc**: Utiliza a biblioteca **fastrtc** para gerenciar a comunicação de áudio via WebRTC, permitindo uma interface de chat de voz simples.  

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python **3.13** ou superior.  
- **uv** (gerenciador de pacotes). Se você não o tiver, instale-o com:  
  ```bash
  pip install uv
