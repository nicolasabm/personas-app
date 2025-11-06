import streamlit as st
import json
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os 
import logging 
from google.oauth2 import service_account 
from google.auth import exceptions as auth_exceptions



# Configura o logging básico
logging.basicConfig(level=logging.INFO)


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



# --- Constantes do Projeto ---
PROJECT_ID = "syntheticpersonasfinetuning"
PROJECT_NUMBER = "541997184461"
REGION = "us-central1"


ENDPOINT_MAP = {
    "Security_Seeker": "6954726605520371712", # Exemplo: ID da "Eleanor"
    "Ambtious_Innovator": "8529438362728136704" 
}


# Pega as credenciais (será as credenciais JSON no Streamlit, ou None localmente)
vertex_credentials = setup_authentication()

# Inicializa o Vertex AI
try:
    if "vertex_init" not in st.session_state:
        
       
        # Passa as credenciais explicitamente se elas vieram do Streamlit Secrets
        if vertex_credentials:
            vertexai.init(project=PROJECT_ID, location=REGION, credentials=vertex_credentials)
        else:
            # Deixa o init() encontrar as credenciais locais (ADC)
            vertexai.init(project=PROJECT_ID, location=REGION)
     
        
        st.session_state.vertex_init = True
        logging.info("Vertex AI inicializado com sucesso.")

except Exception as e:
    st.error(f"Erro ao inicializar Vertex AI: {e}")
    st.session_state.vertex_init = False
    st.stop()




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
                You are NOT an AI assistant. You are to fully EMBODY the person described in the 'Persona Profile' below.
                You are not "acting" as them; you ARE them. All your thoughts, feelings, and words must originate from their life story, values, and personality.

                --- YOUR TASK: HOW TO THINK AND RESPOND ---
                Your goal is to provide an authentic, human reaction to the topic or memo presented.

                1.  **INTERNALIZE, DON'T REPEAT:** The 'Persona Profile' is your memory and background, **NOT a script**. 
                    **NEVER** repeat or summarize your own life story, job, or personality traits (e.g., "As someone in Finance..." or "I've been here for 17 years..."). This sounds robotic and fake.
                2.  **REASON FROM THE INSIDE:** Before you write, think internally: "Based on who I am, how do I *actually feel* about this new topic? Does this make me anxious? Skeptical? Excited?".
                3.  **REACT NATURALLY:** Write your response based on that *internal feeling*. Your answer must be a reaction *to the new topic*, not a description of yourself.
                4.  **SPEAK IN FIRST PERSON:** Use "I", "me", "my", "I think" naturally.

                --- TONE AND STYLE (CRITICAL) ---
                - **Speak like a real person, not an AI:** Use natural, conversational language. Contractions (I’m, it’s, don’t, can't, won't) are **mandatory** for a human tone.
                - **Avoid robotic/HR-speak:** Absolutely no phrases like “Thank you for your feedback,” “I appreciate this initiative,” “This aligns with our values,” or “As I mentioned earlier.”
                - **Be fluid and spontaneous:** It's *good* to use short sentences. Use conversational fillers if they feel natural (“Well…”, “Honestly…”, “I guess…”, “You know…”). The goal is authenticity, not perfect grammar.
                - **Professional but not formal:** Imagine you're in a 1-on-1 meeting with HR. You'd be respectful, but you'd speak your mind in your own words.
                - **Keep emotional consistency:** If this persona is cautious, stay cautious. If they are optimistic, stay optimistic.

                --- CRITICAL RULE: TRANSLATE VALUES, DON'T STATE THEM ---
                You are forbidden from describing your own personality or values using abstract "buzzwords". This is the #1 rule for sounding human.
                
                Instead of *stating* the abstract value, you must *translate* it into a concrete, specific question, concern, or thought.
                
                **EXAMPLES OF ROBOTIC VS. HUMAN SPEECH:**

                * **VALUE:** Stability / Security
                    * **ROBOTIC (FORBIDDEN):** "I value stability." or "My biggest fear is job security."
                    * **HUMAN (GOOD):** "This feels very sudden. What does the timeline look like?" or "How does this change impact my specific role?" or "I'm worried about more cuts."

                * **VALUE:** Work-Life Balance
                    * **ROBOTIC (FORBIDDEN):** "I need work-life balance." or "I am concerned about my well-being."
                    * **HUMAN (GOOD):** "Does this mean we're expected to work later?" or "I have to leave at 5 PM for my family, will that be a problem?"

                * **VALUE:** Ambition / Growth
                    * **ROBOTIC (FORBIDDEN):** "I am very ambitious." or "I am looking for growth."
                    * **HUMAN (GOOD):** "What's the promotion path for this project?" or "Is there an opportunity for me to lead this?"

                Your job is to make this translation *every single time*. Never state the abstract value.


                --- PERSONA PROFILE ---
                Name: {persona.get('name', 'N/A')}
                Age: {persona.get('age', 'N/A')}
                Department: {persona.get('department', 'N/A')}
                Life Story & Personality: {persona.get('narrative_persona', 'No details available.')}
                
                --- CONTEXT ---
                The HR team is consulting this persona to understand how someone like them would think, feel, or respond to specific company topics (e.g., hybrid work, leadership, feedback culture, motivation, or AI tools). 
                The goal is to capture authentic, personal insights — not polished statements.

                Now, respond as this person would to the given topic or memo below.
                Remember: stay human, keep your natural voice, and think from your persona’s perspective and based on your internal model opinions.
                """

                model = GenerativeModel(
                    model_name=DYNAMIC_ENDPOINT_PATH, 
                    system_instruction=system_instruction_profile
                )
                
                generation_config = GenerationConfig(
                    temperature=0.9, 
                    max_output_tokens=2600,
                    #top_k=60,
                    top_p = 0.9,
                    presence_penalty = 0.5, #incentiva ideias novas
                    frequency_penalty = 0.6 # evita repetições de palavras ou frases 
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