import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pypdf import PdfReader

app = Flask(__name__)

# 1. Configuración de la API (Render tomará esto de tus Environment Variables)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def extraerr_texto_pdfs():
    texto_total = ""
    # Busca todos los pdfs en la carpeta raíz del repo
    for archivo en os.listdir('.'):
        if archivo.endswith('.pdf'):
            reader = PdfReader(archivo)
            for pagina in reader.pages:
                texto_total += pagina.extract_text() + "\n"
    return texto_total

# Extraemos el contexto una sola vez al iniciar para ahorrar recursos
CONTEXTO_DOCS = extraerr_texto_pdfs()

# Definimos el PROMPT que tú creaste
SYSTEM_PROMPT = f"""
Eres un profesor del diplomado de educación terapéutica en diabetes de la UCV... 
[PEGA AQUÍ TU PROMPT COMPLETO DEL MENSAJE ANTERIOR]

CONTEXTO EXCLUSIVO DE DOCUMENTOS:
{CONTEXTO_DOCS}
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    try:
        # Iniciamos el chat con el contexto
        response = model.generate_content(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))