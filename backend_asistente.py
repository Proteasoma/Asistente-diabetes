import os
import gc
import shutil
import traceback
import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app)

# --- 1. Configuración de API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables de estado global
qa_chain = None
is_loading = False
init_error = None

# --- 2. Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "Archivos PDF")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db_diabetes")

def run_setup():
    """Hilo de fondo para procesar los PDFs sin bloquear el arranque del servidor"""
    global vector_db, qa_chain, is_loading, init_error
    is_loading = True
    print("SISTEMA: Iniciando procesamiento de PDFs en segundo plano...")
    
    try:
        if not API_KEY:
            raise ValueError("GOOGLE_API_KEY no encontrada.")

        genai.configure(api_key=API_KEY)
        
        # FIX: Usar nombre corto del modelo para evitar Error 400
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base anterior si existe para evitar conflictos de versiones
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            time.sleep(1)

        # Carga de documentos
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Cargando {len(pdf_files)} archivos...")
            for filename in pdf_files:
                loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                documents.extend(loader.load())
        
        if not documents:
            raise ValueError(f"No hay PDFs en {PDF_FOLDER_PATH}")

        # Split e Indexación
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configuración RAG
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        
        template = """Eres un profesor de la Universidad Central de Venezuela (UCV).
        Usa el contexto educativo proporcionado para responder.
        Contexto: {context}
        Pregunta: {question}
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        print("SISTEMA: IA lista y base vectorial conectada.")
        
    except Exception as e:
        init_error = str(e)
        print(f"ERROR CRÍTICO: {init_error}")
        traceback.print_exc()
    finally:
        is_loading = False
        gc.collect()

# --- 3. Endpoints ---

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "ia_ready": qa_chain is not None,
        "loading": is_loading,
        "error": init_error
    })

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        if is_loading:
            return jsonify({"response": "Estoy procesando los manuales de la UCV. Dame un momento..."}), 503
        return jsonify({"response": f"Error de inicialización: {init_error}"}), 500

    data = request.get_json()
    user_question = data.get('question')
    
    try:
        # Usar invoke en lugar de run (más moderno)
        result = qa_chain.invoke({"query": user_question})
        return jsonify({"response": result["result"]})
    except Exception as e:
        return jsonify({"response": f"Error en consulta: {str(e)}"}), 500

if __name__ == '__main__':
    # Lanzar el procesamiento en un hilo separado
    threading.Thread(target=run_setup, daemon=True).start()
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
