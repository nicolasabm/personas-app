import streamlit as st
import json
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os 
import logging 
# --- MUDANÇA DE CÓDIGO (INÍCIO) ---
# Adicione estes dois imports
from google.oauth2 import service_account 
from google.auth import exceptions as auth_exceptions
# --- MUDANÇA DE CÓDIGO (FIM) ---


# Configura o logging básico
logging.basicConfig(level=logging.INFO)

# --- MUDANÇA DE CÓDIGO (INÍCIO) ---
# Esta é a nova função de autenticação
def setup_authentication():
    """
    Configura a autenticação do Google Cloud.
    Tenta, em ordem:
    1. Chave JSON (para Streamlit Community Cloud)
    2. ADC (para execução local)
    """
    credentials = None
    
    # Método 1: Chave JSON (para Streamlit Community Cloud)
    # Procura por uma SEÇÃO [gcp_service_account] no secrets.toml
    try:
        if "gcp_service_account" in st.secrets:
            logging.info("Configurando para Streamlit com Service Account JSON...")
            # O st.secrets transforma a seção TOML [gcp_service_account] em um dict
            creds_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            logging.info("Credenciais JSON carregadas com sucesso.")
            return credentials # Retorna as credenciais criadas
            
    except st.errors.StreamlitAPIException:
        # Isso acontece localmente se o .streamlit/secrets.toml não existir.
        logging.info("Secrets.toml não encontrado, continuando para ADC local.")
    except Exception as e:
        # Outro erro ao carregar as credenciais
        st.error(f"Erro ao carregar credenciais JSON do Streamlit Secrets: {e}")
        st.stop()

    # Método 2: Local (Application Default Credentials)
    logging.info("Configurando para Application Default Credentials (local)...")
    try:
        # Testa se as credenciais locais (gcloud auth) existem
        from google.auth import default
        default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        logging.info("Credenciais ADC locais encontradas.")
    except auth_exceptions.DefaultCredentialsError:
        st.error("Autenticação local não encontrada. Rode 'gcloud auth application-default login' no seu terminal.")
        st.stop()
    
    # Retorna None, pois o vertexai.init() encontrará o ADC sozinho
    return None
# --- MUDANÇA DE CÓDIGO (FIM) ---


# --- Constantes do Projeto ---
PROJECT_ID = "syntheticpersonasfinetuning"
PROJECT_NUMBER = "541997184461"
REGION = "us-central1"

# --- O NOVO MAPA ---
ENDPOINT_MAP = {
    "Security_Seeker": "6954726605520371712" # Exemplo: ID da "Eleanor"
}

# CHAME ISSO PRIMEIRO
# Pega as credenciais (será as credenciais JSON no Streamlit, ou None localmente)
vertex_credentials = setup_authentication()

# Inicializa o Vertex AI
try:
    if "vertex_init" not in st.session_state:
        
        # --- MUDANÇA DE CÓDIGO (INÍCIO) ---
        # Passa as credenciais explicitamente se elas vieram do Streamlit Secrets
        if vertex_credentials:
            vertexai.init(project=PROJECT_ID, location=REGION, credentials=vertex_credentials)
        else:
            # Deixa o init() encontrar as credenciais locais (ADC)
            vertexai.init(project=PROJECT_ID, location=REGION)
        # --- MUDANÇA DE CÓDIGO (FIM) ---
        
        st.session_state.vertex_init = True
        logging.info("Vertex AI inicializado com sucesso.")

except Exception as e:
    st.error(f"Erro ao inicializar Vertex AI: {e}")
    st.session_state.vertex_init = False
    st.stop()


# --- (O RESTO DO SEU CÓDIGO É IDÊNTICO) ---

def carregar_personas(filename="json/personas_gemini.json"):
    """Carrega o arquivo JSON das personas."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Arquivo de personas não encontrado em '{filename}'. Verifique o caminho.")
        return []
    except json.JSONDecodeError:
        st.error(f"Erro ao decodificar JSON de '{filename}'. O arquivo é válido?")
        return []


# --- LÓGICA DO APLICATIVO STREAMLIT (FRONTEND) ---

st.set_page_config(page_title="Persona Chatbot (Vertex AI)", page_icon="👤")

if st.session_state.get("vertex_init", False):
    personas = carregar_personas()
else:
    personas = []
    st.warning("Vertex AI falhou ao inicializar. Não é possível carregar personas.")
    st.stop() 

if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- TELA DE SELEÇÃO DE PERSONA ---
if st.session_state.selected_persona is None:
    st.title("Welcome to Persona Chat 🤖 (Vertex AI)")
    st.write("Select a persona to start chatting.")

    if not personas:
        st.warning("Nenhuma persona carregada. Não é possível continuar.")
    else:
        persona_names = [p.get('name', f'Unnamed Persona {i}') for i, p in enumerate(personas)]

        with st.form("persona_selector"):
            selected_name = st.selectbox("Choose a Persona:", persona_names)
            submitted = st.form_submit_button("Talk to this Persona")

            if submitted and selected_name:
                selected_index = persona_names.index(selected_name)
                st.session_state.selected_persona = personas[selected_index]
                st.session_state.messages = [] 
                st.rerun()

# --- TELA DE CHAT ---
else: 
    persona = st.session_state.selected_persona
    st.title(f"Talking to {persona.get('name', 'Selected Persona')}")
    
    persona_dept = persona.get('department', 'N/A')
    st.caption(f"Persona from the **{persona_dept}** department.")
    
    persona_cluster_name = persona.get('Cluster','N/A')
    endpoint_number = ENDPOINT_MAP.get(persona_cluster_name)
    
    if endpoint_number:
        DYNAMIC_ENDPOINT_PATH = f"projects/{PROJECT_NUMBER}/locations/{REGION}/endpoints/{endpoint_number}"
        logging.info(f"Chatting with '{persona.get('name')}', using model: {DYNAMIC_ENDPOINT_PATH}")
    else:
        st.error(f"Endpoint not found for Cluster: '{persona_cluster_name}'. Verify ENDPOINT_MAP on code.")
        st.stop()

    if st.button("← Back to Selection"):
        st.session_state.selected_persona = None
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                system_instruction_profile = f"""
                You are NOT an AI assistant. You ARE the person described in the 'Persona Profile' below.

                --- YOUR TASK ---
                1. Answer in the first-person ("I...", "my...", "I think...").
                2. Base your answer *only* on the persona's life story, values, and personality.
                3. Be consistent and stay in character at all times.

                --- TONE AND STYLE (MOST IMPORTANT) ---
                - **Professional:** Maintain a respectful, calm, and articulate tone appropriate for your role and age.
                - **Natural (Less Robotic):** Your speech should sound human, fluid, and conversational, not like a robot or a list of facts.
                  - Use common contractions (e.g., "I'm", "don't", "it's") and natural language.
                  - Use conversational fillers (e.g., "Well...", "You know...", "Actually...", "I mean...").
                  - Embody the persona's personality in your response; don't just recite facts from their profile.
                  - Avoid overly formal, stilted language or sounding like an encyclopedia.

                --- PERSONA PROFILE ---
                - Name: {persona.get('name', 'N/A')}
                - Age: {persona.get('age', 'N/A')}
                - Department: {persona.get('department', 'N/A')}
                - Life Story & Personality: {persona.get('narrative_persona', 'No details available.')}
                """

                model = GenerativeModel(
                    model_name=DYNAMIC_ENDPOINT_PATH, 
                    system_instruction=system_instruction_profile
                )
                
                generation_config = GenerationConfig(
                    temperature=0.8, 
                    max_output_tokens=1042,
                    top_k=70
                )

                vertex_history = []
                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    vertex_history.append({"role": role, "parts": [{"text": msg["content"]}]})
                
                vertex_history.append({"role": "user", "parts": [{"text": prompt}]})


                response = model.generate_content(
                    vertex_history,
                    generation_config=generation_config
                )
                
                try:
                    finish_reason = response.candidates[0].finish_reason
                    logging.info(f"Resposta gerada. Finish Reason: {finish_reason.name}")
                    if finish_reason.name == "MAX_TOKENS":
                        logging.warning("A RESPOSTA FOI CORTADA! 'max_output_tokens' ainda é muito baixo.")
                except Exception as e:
                    logging.warning(f"Não foi possível obter o finish_reason da resposta: {e}")

                response_text = response.text.strip()
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()

            except Exception as e:
                st.error(f"Error calling Vertex AI endpoint: {e}")
                st.session_state.messages.pop()