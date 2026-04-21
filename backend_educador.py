"""
🩺 PLANIFICADOR EDUCATIVO EN DIABETES
Backend optimizado para Render | Gemini 1.5 Flash + RAG Local
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

# LangChain & RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar variables de entorno (lee .env local o variables de Render)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CARPETA_PDFS = "Archivos PDF"

# ==================== INICIALIZAR APP ====================
app = FastAPI(
    title="Planificador Educativo en Diabetes",
    description="Asistente basado en evidencia pedagógica para diplomado",
    version="2.0.0"
)

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Render inyecta su dominio automáticamente
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PROMPT PEDAGÓGICO ====================
SYSTEM_PROMPT = """Eres un asistente experto en educación en diabetes para participantes de un diplomado en educación terapéutica.

Tu tarea es ayudar a planificar presentaciones y actividades educativas aplicando RIGUROSAMENTE:

1. PRECAUCIONES UNIVERSALES DE ALFABETIZACIÓN EN SALUD:
- Usa lenguaje claro (nivel 6°-8° grado), evita jerga médica sin explicar
- Organiza la información en MÁXIMO 3 mensajes clave por sesión
- Usa formato "enseñar-devolver" (teach-back) para verificar comprensión
- Diseña materiales con alto contraste y tipografía ≥24pt

2. TEORÍA DE LA AUTOEFICACIA (Bandura) - INCLUIR AL MENOS 2 ESTRATEGIAS:
- Experiencias de dominio: Actividades con dificultad progresiva, logros alcanzables
- Modelado: Demostraciones, videos de pacientes expertos, ejemplos paso a paso
- Persuasión verbal: Retroalimentación específica y alentadora
- Estado fisiológico/emocional: Reduce ansiedad, crea ambiente seguro

3. TEORÍA DE LA CARGA COGNITIVA:
- Minimiza carga extrínseca: Elimina información decorativa, usa señalización visual
- Optimiza carga intrínseca: Activa conocimientos previos, usa analogías familiares
- Maximiza carga germana: Incluye práctica guiada, mapas conceptuales, casos estructurados

4. TÉCNICA DIDÁCTICA SEGÚN TIPO DE CONOCIMIENTO:
- Cognitivo (conocimiento teórico): Analogías, organizadores gráficos, cuestionarios con justificación
- Psicomotor (cómo hacer): Demostración + práctica, listas de verificación, simulaciones
- Afectivo (sentimiento): Narrativas de pacientes, reflexión guiada, role-playing

5. PRINCIPIOS DE DISEÑO DE PRESENTACIONES:
- Máximo 1 idea por diapositiva
- Contraste alto, tipografía ≥24pt, espacios en blanco intencionales
- Imágenes FUNCIONALES (no decorativas), datos visuales simples

FORMATO DE RESPUESTA OBLIGATORIO:

**OBJETIVO EDUCATIVO**
[Objetivo medible alineado a alfabetización en salud]

**TIPO DE CONOCIMIENTO**
[Cognitivo/Psicomotor/Afectivo + justificación]

**ESTRUCTURA DE LA SESIÓN**
- Inicio (X min): [actividad]
- Desarrollo (X min): [actividad]
- Cierre (X min): [actividad]

**DIAPOSITIVAS CLAVE** (máx. 5)
1. [Título]: [contenido mínimo + principio de diseño aplicado]

**ESTRATEGIAS DE AUTOEFICACIA**
- [Fuente]: [estrategia específica]

**GESTIÓN DE CARGA COGNITIVA**
- Extrínseca: [cómo la minimizas]
- Intrínseca: [cómo la optimizas]
- Germana: [cómo la maximizas]

**EVALUACIÓN FORMATIVA (Teach-Back)**
[Pregunta o actividad específica para verificar comprensión]

**MATERIALES DE APOYO**
[Lista de materiales con especificaciones de alfabetización en salud]

Si el usuario no especifica el tipo de conocimiento, INFIERELO del contexto y explica tu elección."""

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

# ==================== RAG: CARGA DE PDFs ====================
def inicializar_vector_db():
    global vector_db
    
    # Compatible con Render y local
    base_dir = Path(os.getenv("RENDER_PROJECT_SRC_DIR", Path(__file__).parent))
    pdf_path = base_dir / CARPETA_PDFS
    
    print(f"📂 Buscando PDFs en: {pdf_path.absolute()}")
    
    if not pdf_path.exists():
        print("⚠️ Carpeta de PDFs no encontrada. RAG desactivado.")
        return None
    
    try:
        print("📚 Cargando documentos...")
        docs = []
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Error al leer {pdf_file.name}: {e}")
        
        if not docs:
            print("⚠️ No se cargaron documentos válidos.")
            return None
        
        print(f"✓ {len(docs)} páginas cargadas")
        
        print("✂️ Fragmentando...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        print(f"✓ {len(chunks)} fragmentos generados")
        
        print("🧠 Generando embeddings locales...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        persist_dir = base_dir / "vector_db"
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )
        print(f"✅ Vector DB lista en {persist_dir}")
        return vector_db
        
    except Exception as e:
        print(f"❌ Error en RAG: {e}")
        return None

# ==================== RAG: CONSULTA ====================
def consultar_rag(query: str, k: int = 4):
    if vector_db is None:
        return [], []
    try:
        docs = vector_db.as_retriever(search_kwargs={"k": k}).invoke(query)
        return [d.page_content for d in docs], [d.metadata.get("source", "PDF") for d in docs]
    except Exception as e:
        print(f"Error en retrieval: {e}")
        return [], []

# ==================== ENDPOINT PRINCIPAL ====================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    contextos, fuentes = consultar_rag(req.mensaje, k=4)
    
    contexto_str = ""
    if contextos:
        contexto_str = "\n\n📚 CONTEXTO DE MATERIALES:\n" + "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contextos)])
    if req.tipo_conocimiento:
        contexto_str += f"\n🎯 TIPO SOLICITADO: {req.tipo_conocimiento.upper()}"
        
    prompt_final = f"{SYSTEM_PROMPT}\n\n📥 SOLICITUD: {req.mensaje}{contexto_str}\n\n✅ Genera la propuesta siguiendo el formato obligatorio."

    try:
        # ✅ Configuración optimizada para Gemini 1.5 Flash
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,  # Precisión pedagógica
            max_output_tokens=2000,
            google_api_key=GOOGLE_API_KEY
        )
        
        respuesta = llm.invoke(prompt_final)
        return ChatResponse(respuesta=respuesta.content, fuentes_consultadas=list(set(fuentes)))
        
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "quota" in error_msg or "exceeded" in error_msg:
            detalle = "⚠️ Error de cuota o clave de Google AI. Verifica tu API key en Render o el límite gratuito diario."
        else:
            detalle = f"Error técnico: {e}"
            
        return ChatResponse(
            respuesta=f"❌ No se pudo generar la propuesta.\n\n{detalle}",
            fuentes_consultadas=fuentes
        )

# ==================== HEALTH & STARTUP ====================
@app.get("/health")
async def health():
    return {"status": "ok", "rag_activo": vector_db is not None, "modelo": "gemini-1.5-flash"}

@app.on_event("startup")
async def startup():
    print("\n🚀 INICIANDO SERVIDOR...")
    inicializar_vector_db()
    print("✅ LISTO\n")

if __name__ == "__main__":
    import uvicorn
    # Render inyecta la variable PORT automáticamente
    puerto = int(os.getenv("PORT", 8000))
    uvicorn.run("backend_educador:app", host="0.0.0.0", port=puerto, reload=False)