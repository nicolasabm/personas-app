import streamlit as st
import json
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os # <-- Adicione este import
import logging # <-- Adicione este import

# def setup_authentication():
#     """
#     Configura a autenticação do Google Cloud.
#     - No Streamlit Cloud: Define vars de ambiente para Workload Identity Federation (OIDC).
#     - Localmente: Confia nas Credenciais Padrão da Aplicação (gcloud auth).
#     """
    
#     # Verifica se está rodando no Streamlit Cloud (pela existência do secret)
#     if "GCP_SERVICE_ACCOUNT_EMAIL" in st.secrets:
#         logging.info("Configurando para Streamlit OIDC (Workload Identity Federation)...")
        
#         # Pega o e-mail da conta de serviço alvo do secret
#         service_account_email = st.secrets["GCP_SERVICE_ACCOUNT_EMAIL"]
        
#         # Define a variável de ambiente que a SDK do Google usará
#         # Isso diz à biblioteca de autenticação QUAL conta de serviço ela deve impersonar
#         os.environ["GOOGLE_SERVICE_ACCOUNT_EMAIL"] = service_account_email
        
#         # A biblioteca 'google-auth' (que a Vertex AI usa) irá:
#         # 1. Ver esta variável de ambiente.
#         # 2. Detectar que está em um ambiente OIDC (Streamlit Cloud).
#         # 3. Lidar automaticamente com a troca de token para impersonar essa conta.
        
#     else:
#         # Se não estiver no Streamlit Cloud, supõe que 'gcloud auth' local está ativo
#         logging.info("Configurando para Application Default Credentials (local)...")
#         # Nenhuma ação é necessária. 'vertexai.init()' encontrará as credenciais locais.

logging.basicConfig(level=logging.INFO)

def setup_authentication():
    """
    Configura a autenticação do Google Cloud.
    - No Streamlit Cloud: Define vars de ambiente para Workload Identity Federation (OIDC).
    - Localmente: Confia nas Credenciais Padrão da Aplicação (gcloud auth).
    """
    
    gcp_secret = None
    try:
        # Tenta acessar st.secrets. Isso irá falhar com StreamlitAPIException
        # se o arquivo .streamlit/secrets.toml não for encontrado.
        if "GCP_SERVICE_ACCOUNT_EMAIL" in st.secrets:
            gcp_secret = st.secrets["GCP_SERVICE_ACCOUNT_EMAIL"]

    except st.errors.StreamlitAPIException:
        # Isso é esperado localmente se o arquivo .toml não existir.
        logging.info("Arquivo secrets.toml não encontrado, assumindo execução local (ADC).")
        # gcp_secret permanece None

    if gcp_secret:
        # Encontrou o secret, configura para OIDC (Streamlit Cloud)
        logging.info("Configurando para Streamlit OIDC (Workload Identity Federation)...")
        
        # Define a variável de ambiente que a SDK do Google usará
        os.environ["GOOGLE_SERVICE_ACCOUNT_EMAIL"] = gcp_secret
        
    else:
        # Se não encontrou o secret, supõe que 'gcloud auth' local está ativo
        logging.info("Configurando para Application Default Credentials (local)...")
        # Nenhuma ação é necessária. 'vertexai.init()' encontrará as credenciais locais.


# PROJECT_ID = "syntheticpersonasfinetuning"
# REGION = "us-central1"
#                         #project_number         #location            #Id_endpoint
# ENDPOINT_ID = "projects/541997184461/locations/us-central1/endpoints/4205454954871128064"

# --- Constantes do Projeto ---
PROJECT_ID = "syntheticpersonasfinetuning"
PROJECT_NUMBER = "541997184461"
REGION = "us-central1"

# --- O NOVO MAPA ---
# Mapeia o 'department' (ou 'cluster_name') para o ID numérico do endpoint
ENDPOINT_MAP = {
    #"Pragmatic_Guardian": "4205454954871128065", # Exemplo: ID da "Mary"
    #"Disciplined_Traditionalist": "4205454954871128063", # Exemplo: ID da "John"
    #"Ambitious_Innovator": "4205454954871128062", # Exemplo: ID da "Alex"
    "Security_Seeker": "4205454954871128064"# Exemplo: ID da "Eleanor"
}

# CHAME ISSO PRIMEIRO
setup_authentication()

# Inicializa o Vertex AI (agora ele encontrará as credenciais corretas)
try:
    if "vertex_init" not in st.session_state:
        vertexai.init(project=PROJECT_ID, location=REGION)
        st.session_state.vertex_init = True
        logging.info("Vertex AI inicializado.")
except Exception as e:
    st.error(f"Erro ao inicializar Vertex AI: {e}")
    st.session_state.vertex_init = False
    st.stop()


# --- FUNÇÕES DE LÓGICA (BACKEND) ---

# Usa cache para inicializar o Vertex AI apenas uma vez
# @st.cache_resource(show_spinner="Connecting to Google Cloud...")
# def initialize_vertexai():
#     """Inicializa a conexão com o Vertex AI usando credenciais locais padrão (ADC)."""
#     print("Attempting to initialize Vertex AI using default credentials...")
#     try:
#         # Tenta inicializar usando as credenciais do 'gcloud auth application-default login'
#         vertexai.init(project=PROJECT_ID, location=REGION) 
#         print("Vertex AI initialized successfully!")
#         return True # Indica sucesso
        
#     # Trata especificamente o erro de credenciais padrão não encontradas
#     except auth_exceptions.DefaultCredentialsError:
#         st.error(f"""
#             **Failed to find default Google Cloud credentials (ADC).** Please run this command in your terminal:
            
#             `gcloud auth application-default login`
            
#             Then **restart** the Streamlit app.
#         """)
#         return False
#     # Trata outros erros de inicialização
#     except Exception as e:
#         st.error(f"""
#             **Failed to initialize Vertex AI.** Error: {e}
            
#             **Troubleshooting:**
#             - **Project/Region:** Are PROJECT_ID ('{PROJECT_ID}') and REGION ('{REGION}') correct?
#             - **API Enabled?** Is the Vertex AI API enabled in your Google Cloud project?
#         """)
#         return False


def carregar_personas(filename="json/personas_gemini.json"):
    """Carrega o arquivo JSON das personas."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Personas file not found at '{filename}'. Please check the path.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from '{filename}'. Is the file valid?")
        return []


# --- LÓGICA DO APLICATIVO STREAMLIT (FRONTEND) ---

st.set_page_config(page_title="Persona Chatbot (Vertex AI)", page_icon="👤")

# # Tenta inicializar o Vertex AI
# vertex_initialized = initialize_vertexai()

# # Carrega as personas (só continua se o Vertex AI inicializou)
# if vertex_initialized:
#     personas = carregar_personas()
# else:
#     personas = []
#     st.stop() # Para a execução se não conseguiu conectar ao GCP
# Carrega as personas (só continua se o Vertex AI inicializou com SUCESSO)
if st.session_state.get("vertex_init", False): # <-- Verifica a VARIÁVEL DE SESSÃO correta
    personas = carregar_personas()
else:
    personas = []
    st.warning("Vertex AI failed to initialize. Cannot load personas.")
    st.stop() # Para a execução se não conseguiu conectar ao GCP

# Gerenciamento de estado da sessão
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- TELA DE SELEÇÃO DE PERSONA ---
if st.session_state.selected_persona is None:
    st.title("Welcome to Persona Chat 🤖 (Vertex AI)")
    st.write("Select a persona to start chatting.")

    if not personas:
        st.warning("No personas loaded. Cannot proceed.")
    else:
        persona_names = [p.get('name', f'Unnamed Persona {i}') for i, p in enumerate(personas)]

        with st.form("persona_selector"):
            selected_name = st.selectbox("Choose a Persona:", persona_names)
            submitted = st.form_submit_button("Talk to this Persona")

            if submitted and selected_name:
                # Encontra a persona selecionada (lidando com nomes possivelmente duplicados)
                selected_index = persona_names.index(selected_name)
                st.session_state.selected_persona = personas[selected_index]
                st.session_state.messages = [] # Limpa histórico ao trocar de persona
                st.rerun()

# --- TELA DE CHAT ---
else: # Bloco de chat (quando uma persona está selecionada)
    persona = st.session_state.selected_persona
    st.title(f"Talking to {persona.get('name', 'Selected Persona')}")
    
    # --- LÓGICA DE SELEÇÃO DINÂMICA DO MODELO ---
    persona_dept = persona.get('department', 'N/A')
    st.caption(f"Persona from the **{persona_dept}** department.")
    
    # 1. Encontra o ID numérico do endpoint no nosso mapa
    persona_cluster_name = persona.get('Cluster','N/A')
    endpoint_number = ENDPOINT_MAP.get(persona_cluster_name)
    
    # 2. Constrói o caminho completo do endpoint
    if endpoint_number:
        DYNAMIC_ENDPOINT_PATH = f"projects/{PROJECT_NUMBER}/locations/{REGION}/endpoints/{endpoint_number}"
        logging.info(f"Chatting with '{persona.get('name')}', using model: {DYNAMIC_ENDPOINT_PATH}")
    else:
        st.error(f"Endpoint not found for Cluster: '{persona_cluster_name}'. Verify  ENDPOINT_MAP on code.")
        st.stop()
    # --- FIM DA LÓGICA DE SELEÇÃO ---

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
                Your task is to answer from the first-person perspective ("I...") of this character.
                Base your answer on their life story, values, and personality. Be consistent and stay in character.

                Persona Profile:
                - Name: {persona.get('name', 'N/A')}
                - Age: {persona.get('age', 'N/A')}
                - Department: {persona.get('department', 'N/A')}
                - Life Story & Personality: {persona.get('narrative_persona', 'No details available.')}
                """

                # 3. Inicializa o modelo com o CAMINHO DINÂMICO
                model = GenerativeModel(
                    model_name=DYNAMIC_ENDPOINT_PATH, # <-- AQUI ESTÁ A MUDANÇA
                    system_instruction=system_instruction_profile
                )
                
                generation_config = GenerationConfig(
                    temperature=0.8, 
                    max_output_tokens=2048, 
                    top_k=50
                )

                vertex_history = []
                for msg in st.session_state.messages:
                    role = "user" if msg["role"] == "user" else "model"
                    vertex_history.append({"role": role, "parts": [{"text": msg["content"]}]})

                response = model.generate_content(
                    vertex_history,
                    generation_config=generation_config
                )
                
                # ... (Debug da 'finish_reason' que adicionamos antes, se você quiser manter) ...
                # --- DEBUG: VERIFICAR POR QUE PAROU ---
                try:
                    finish_reason = response.candidates[0].finish_reason
                    logging.info(f"Resposta gerada. Finish Reason: {finish_reason.name}")
                    if finish_reason.name == "MAX_TOKENS":
                        logging.warning("A RESPOSTA FOI CORTADA! O 'max_output_tokens' é muito baixo.")
                        # Opcional: Avisar o usuário na interface
                        # st.warning("Ops! Minha resposta foi cortada. Talvez eu precise de mais 'max_output_tokens'.")
                except Exception as e:
                    logging.warning(f"Não foi possível obter o finish_reason da resposta: {e}")
                # --- FIM DO DEBUG ---
                
                response_text = response.text.strip()
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()

            except Exception as e:
                st.error(f"Error calling Vertex AI endpoint: {e}")
                st.session_state.messages.pop()