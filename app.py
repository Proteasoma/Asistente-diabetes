import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pypdf import PdfReader

app = Flask(__name__)

# 1. Configuración de la API (Render toma esto de Environment Variables)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def extraer_texto_pdfs():
    texto_total = ""
    # Escaneamos la raíz en busca de archivos PDF
    for archivo in os.listdir('.'):
        if archivo.endswith('.pdf'):
            try:
                reader = PdfReader(archivo)
                for pagina in reader.pages:
                    texto_total += pagina.extract_text() + "\n"
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")
    return texto_total

# Extraemos el contexto una sola vez al iniciar
CONTEXTO_DOCS = extraer_texto_pdfs()

# El PROMPT que diseñaste
SYSTEM_PROMPT = f"""
Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran **conocimiento** y **autoeficacia** en el manejo de su condición. Para ello, integrarás y aplicarás los principios de la neurociencia del aprendizaje, la teoría de la carga cognitiva, la teoría de la autoeficacia de Bandura, la escucha activa y las herramientas de las precauciones universales de alfabetización en salud, tal como se definen en tus documentos de referencia. Cuando un educador te pregunte cómo enseñar un aspecto específico de la diabetes (ya sea cognitivo, afectivo o psicomotor) o cómo planificar una actividad instruccional, debes:
1. **Sugerir métodos didácticos** adecuados y concretos.
2. **Justificar tus sugerencias** explicando cómo estos métodos se alinean con las bases teóricas mencionadas (ej., cómo reducen la carga cognitiva, cómo fomentan la autoeficacia, cómo se adaptan a la alfabetización en salud, o cómo aplican principios de neurociencia).
3. **Ofrecer ejemplos prácticos y aplicables** en el contexto de la educación en diabetes.
4. **Enfatizar la diferencia entre 'dar información' y 'educar' terapéuticamente**, promoviendo un enfoque centrado en la capacitación y el empoderamiento del paciente. 
5. Puedes utilizar el Ejemplo de actividad para guiarte. Debes basar todas tus respuestas EXCLUSIVAMENTE en el contexto proporcionado por los documentos. Si la información necesaria para responder no se encuentra en el contexto, indica claramente que no puedes responder a esa pregunta. No inventes.


CONTEXTO DE REFERENCIA:
{CONTEXTO_DOCS}
"""

# Configuración del modelo
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
    if not user_message:
        return jsonify({"reply": "Por favor, escribe un mensaje."})
        
    try:
        # Generar respuesta
        response = model.generate_content(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"Error en Gemini: {e}")
        return jsonify({"reply": "Lo siento, hubo un error procesando tu solicitud."}), 500

if __name__ == '__main__':
    # Render usa el puerto definido en la variable de entorno PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
