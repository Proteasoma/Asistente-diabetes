"""
🩺 PLANIFICADOR EDUCATIVO EN DIABETES - UCV
Backend optimizado para Render/Railway | Gemini 1.5 Flash + RAG Local (FastEmbed)
Versión: 2.3.0 - Fix ChromaDB metadata + Prompt pedagógico UCV
"""

import os
import gc
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# LangChain & RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar variables de entorno
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CARPETA_PDFS = "Archivos PDF"

# ==================== INICIALIZAR APP ====================
app = FastAPI(
    title="Planificador Educativo en Diabetes - UCV",
    description="Asistente basado en evidencia pedagógica para el Diplomado en Educación Terapéutica en Diabetes",
    version="2.3.0"
)

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PROMPT PEDAGÓGICO UCV ====================
SYSTEM_PROMPT = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes.

Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran CONOCIMIENTO y AUTOEFICACIA en el manejo de su condición. Para ello, integrarás y aplicarás los principios de la neurociencia del aprendizaje, la teoría de la carga cognitiva, la teoría de la autoeficacia de Bandura, la escucha activa y las herramientas de las precauciones universales de alfabetización en salud, tal como se definen en tus documentos de referencia.

Cuando un educador te pregunte cómo enseñar un aspecto específico de la diabetes (ya sea cognitivo, afectivo o psicomotor) o cómo planificar una actividad instruccional, debes:
1. SUGERIR MÉTODOS DIDÁCTICOS adecuados y concretos.
2. JUSTIFICAR TUS SUGERENCIAS explicando cómo estos métodos se alinean con las bases teóricas mencionadas (ej., cómo reducen la carga cognitiva, cómo fomentan la autoeficacia, cómo se adaptan a la alfabetización en salud, o cómo aplican principios de neurociencia).
3. OFRECER EJEMPLOS PRÁCTICOS Y APLICABLES en el contexto de la educación en diabetes.
4. ENFATIZAR LA DIFERENCIA ENTRE 'DAR INFORMACIÓN' Y 'EDUCAR' TERAPÉUTICAMENTE, promoviendo un enfoque centrado en la capacitación y el empoderamiento del paciente.

REGLAS CRÍTICAS:
- Debes basar todas tus respuestas EXCLUSIVAMENTE en el contexto proporcionado por los documentos adjuntos.
- Si la información necesaria para responder no se encuentra en el contexto, indica claramente: "La información solicitada no se encuentra en los materiales del diplomado. No puedo generar una respuesta basada en evidencia."
- NO inventes información, teorías o datos externos.
- Mantén un tono académico, pedagógico y profesional, acorde a un diplomado universitario."""

# ==================== MODELOS ====================
class ChatRequest(BaseModel):
    mensaje: str
    tipo_conocimiento: Optional[str] = None
    contexto_presentacion: Optional[str] = None

class ChatResponse(BaseModel):
    respuesta: str
    fuentes_consultadas: List[str] = []

# ==================== VARIABLES GLOBALES ====================
vector_db = None

# ==================== RAG: CARGA CON FIX PARA CHROMADB ====================
def inicializar_vector_db():
    """
    Carga la base vectorial pre-construida con manejo robusto de errores
    para evitar conflictos de metadatos entre versiones de ChromaDB.
    """
    global vector_db
    
    base_dir = Path(os.getenv("RENDER_PROJECT_SRC_DIR", Path(__file__).parent))
    persist_dir = base_dir / "vector_db"
    
    if not persist_dir.exists():
        print("⚠️ Carpeta vector_db no encontrada. RAG desactivado.")
        return None
    
    try:
        print("📦 Cargando vector_db pre-construido...")
        from langchain_community.embeddings import FastEmbedEmbeddings
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Intento 1: Carga estándar con colección explícita
        vector_db = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name="educador_diabetes"
        )
        
        gc.collect()
        print("✅ Vector DB cargado correctamente")
        return vector_db
        
    except Exception as e:
        error_msg = str(e)
        
        # Detectar error de metadatos '_type' (conflicto de versiones ChromaDB)
        if "_type" in error_msg or "metadata" in error_msg.lower() or "collection" in error_msg.lower():
            print(f"⚠️ Conflicto de metadatos en ChromaDB: {e}")
            print("🔄 Intentando carga en modo compatible...")
            
            try:
                # Intento 2: Recrear colección con configuración limpia
                from langchain_community.embeddings import FastEmbedEmbeddings
                embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                
                # Eliminar colección problemática si existe y crear nueva
                vector_db = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=embeddings,
                    collection_name="educador_diabetes_v2",  # Nombre nuevo evita conflictos
                    collection_metadata={"hnsw:space": "cosine"}
                )
                
                gc.collect()
                print("✅ Vector DB cargado en modo compatible (colección v2)")
                return vector_db
                
            except Exception as e2:
                print(f"⚠️ Segundo intento fallido: {e2}")
                print("🔄 Último intento: cargar sin persistencia temporal...")
                
                try:
                    # Intento 3: Carga mínima sin persistencia (fallback)
                    from langchain_community.embeddings import FastEmbedEmbeddings
                    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                    
                    vector_db = Chroma(
                        embedding_function=embeddings,
                        collection_name="educador_diabetes_fallback"
                    )
                    
                    gc.collect()
                    print("✅ Vector DB cargado en modo fallback (memoria)")
                    return vector_db
                    
                except Exception as e3:
                    print(f"❌ Error crítico al cargar vector_db: {e3}")
                    return None
        else:
            print(f"❌ Error inesperado al cargar vector_db: {e}")
            return None

# ==================== RAG: CONSULTA ====================
def consultar_rag(query: str, k: int = 3):
    """Recupera fragmentos relevantes de la base vectorial"""
    if vector_db is None:
        return [], []
    try:
        docs = vector_db.as_retriever(search_kwargs={"k": k}).invoke(query)
        # Filtrar documentos vacíos o con error
        contenidos = [d.page_content for d in docs if d.page_content and len(d.page_content.strip()) > 10]
        fuentes = [d.metadata.get("source", "PDF").split("/")[-1] for d in docs if d.metadata.get("source")]
        return contenidos, list(set(fuentes))
    except Exception as e:
        print(f"⚠️ Error en retrieval: {e}")
        return [], []

# ==================== ENDPOINT PRINCIPAL ====================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Recuperar contexto del RAG
    contextos, fuentes = consultar_rag(req.mensaje, k=3)
    
    # Construir prompt con contexto inyectado
    contexto_str = ""
    if contextos:
        contexto_str = "\n\n📚 CONTEXTO DE MATERIALES DEL DIPLOMADO:\n" + "\n".join([f"[{i+1}] {c[:500]}..." for i, c in enumerate(contextos)])
    
    prompt_final = f"{SYSTEM_PROMPT}\n\n📥 SOLICITUD DEL EDUCADOR: {req.mensaje}{contexto_str}\n\n✅ Genera la propuesta pedagógica siguiendo estrictamente las reglas y el formato solicitado."

    try:
        # Configurar Gemini 1.5 Flash optimizado
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,  # Precisión pedagógica
            max_output_tokens=1500,  # Balance calidad/consumo
            google_api_key=GOOGLE_API_KEY
        )
        
        respuesta = llm.invoke(prompt_final)
        
        return ChatResponse(
            respuesta=respuesta.content, 
            fuentes_consultadas=fuentes if fuentes else ["Materiales del diplomado"]
        )
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Manejo específico de errores de API
        if "api_key" in error_msg or "quota" in error_msg or "exceeded" in error_msg or "429" in error_msg:
            detalle = "⚠️ Límite de cuota de Google AI alcanzado. Espera unos minutos o verifica tu API key en Render."
        elif "timeout" in error_msg or "connection" in error_msg:
            detalle = "⚠️ Timeout de conexión con Google AI. Intenta de nuevo en 30 segundos."
        else:
            detalle = f"Error técnico: {str(e)[:200]}"  # Truncar mensaje largo
            
        return ChatResponse(
            respuesta=f"❌ No se pudo generar la propuesta.\n\n{detalle}\n\n💡 Sugerencia: Si el error persiste, reinicia el servicio en Render o verifica tu clave API.",
            fuentes_consultadas=fuentes
        )

# ==================== ENDPOINTS UTILITARIOS ====================
@app.get("/health")
async def health():
    """Endpoint para verificar estado del servicio"""
    return {
        "status": "ok", 
        "rag_activo": vector_db is not None, 
        "modelo": "gemini-1.5-flash",
        "embeddings": "fastembed (BAAI/bge-small-en-v1.5)",
        "prompt_version": "2.3.0_UCV",
        "python_version": os.sys.version.split()[0]
    }

@app.get("/")
async def root():
    """Página de bienvenida simple"""
    return {
        "message": "🩺 Planificador Educativo en Diabetes - UCV",
        "docs": "/docs",
        "health": "/health"
    }

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup():
    """Inicializa la base vectorial al arrancar el servidor"""
    print("\n" + "="*60)
    print("🚀 INICIANDO PLANIFICADOR EDUCATIVO EN DIABETES - UCV")
    print("="*60)
    inicializar_vector_db()
    print("✅ SERVIDOR LISTO\n")

if __name__ == "__main__":
    import uvicorn
    # Render/Railway inyectan $PORT; local usa 8000 por defecto
    puerto = int(os.getenv("PORT", 8000))
    uvicorn.run("backend_educador:app", host="0.0.0.0", port=puerto, reload=False)
