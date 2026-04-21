/**
 * 🩺 PLANIFICADOR EDUCATIVO EN DIABETES - Frontend
 * Conecta con el backend en Render/Railway
 */

// ⚠️ CONFIGURACIÓN: Cambia esta URL por la de tu servicio en Render/Railway
const API_BASE_URL = "https://asistente-diabetes.onrender.com"; // ← REEMPLAZA ESTO

// Elementos del DOM
const chatForm = document.getElementById("chatForm");
const inputField = document.getElementById("input");
const outputDiv = document.getElementById("output");
const submitBtn = document.querySelector('button[type="submit"]');

// ==================== EVENT LISTENERS ====================
if (chatForm) {
  chatForm.addEventListener("submit", handleChatSubmit);
}

// ==================== FUNCIONES PRINCIPALES ====================
async function handleChatSubmit(e) {
  e.preventDefault();
  
  const mensaje = inputField.value.trim();
  const tipoConocimiento = null; // ← Siempre null, la IA inferirá automáticamente según el prompt
  
  if (!mensaje) {
    mostrarError("Por favor, describe tu necesidad educativa o pregunta específica.");
    return;
  }
  
  // Validar URL configurada
  if (!API_BASE_URL || API_BASE_URL.includes("onrender.com") && API_BASE_URL.includes("tu-url")) {
    mostrarError("⚠️ Configura la URL del backend en script.js (línea 8). Debe ser la URL de tu servicio en Render o Railway.");
    return;
  }
  
  // Mostrar estado de carga
  mostrarCargando();
  deshabilitarFormulario(true);
  
  try {
    const respuesta = await consultarAPI(mensaje, tipoConocimiento);
    mostrarRespuesta(respuesta);
    inputField.value = ''; // Limpiar input después de respuesta exitosa
    
  } catch (error) {
    console.error("Error en chat:", error);
    mostrarError(`❌ Error de conexión: ${error.message}\n\nVerifica que:\n1. El backend esté corriendo en Render/Railway\n2. La URL en script.js sea correcta\n3. Tengas conexión a internet`);
    
  } finally {
    deshabilitarFormulario(false);
  }
}

async function consultarAPI(mensaje, tipoConocimiento) {
  const url = `${API_BASE_URL}/api/chat`;
  
  const requestBody = {
    mensaje: mensaje,
    tipo_conocimiento: tipoConocimiento,
    contexto_presentacion: null
  };
  
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });
  
  if (!response.ok) {
    if (response.status === 429) {
      throw new Error("Límite de peticiones excedido. Espera unos segundos e intenta de nuevo.");
    } else if (response.status === 500) {
      throw new Error("Error interno del servidor. Intenta más tarde.");
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  }
  
  const data = await response.json();
  return data;
}

// ==================== FUNCIONES DE UI ====================
function mostrarCargando() {
  outputDiv.innerHTML = `
    <div class="loading-container">
      <div class="spinner"></div>
      <h3>🧠 Generando propuesta educativa...</h3>
      <p class="loading-tip">
        Esto puede tomar 15-40 segundos<br>
        <small>(Primera consulta después de inactividad: ~30-60 seg)</small>
      </p>
    </div>
  `;
  outputDiv.className = "output loading";
}

function mostrarRespuesta(data) {
  const { respuesta, fuentes_consultadas } = data;
  
  // Convertir markdown básico a HTML
  const respuestaHTML = formatearMarkdown(respuesta);
  
  // Generar HTML de fuentes
  const fuentesHTML = fuentes_consultadas && fuentes_consultadas.length > 0 
    ? `
      <div class="fuentes-info">
        <h4>📚 Documentos consultados:</h4>
        <ul>
          ${fuentes_consultadas.map(fuente => {
            const nombreArchivo = fuente.split('/').pop() || fuente;
            return `<li>${nombreArchivo}</li>`;
          }).join('')}
        </ul>
      </div>
    ` 
    : '';
  
  outputDiv.innerHTML = `
    <div class="response-container">
      <div class="response-header">
        <h3>✅ Propuesta Educativa Generada</h3>
        <span class="timestamp">${new Date().toLocaleString('es-VE')}</span>
      </div>
      
      <div class="response-content">
        ${respuestaHTML}
      </div>
      
      ${fuentesHTML}
      
      <div class="response-actions">
        <button class="btn-copy" onclick="copiarRespuesta()">
          📋 Copiar propuesta
        </button>
        <button class="btn-new" onclick="nuevaConsulta()">
          ➕ Nueva consulta
        </button>
      </div>
    </div>
  `;
  
  outputDiv.className = "output response";
  
  // Scroll suave hacia la respuesta
  outputDiv.scrollIntoView({ behavior: "smooth", block: "start" });
}

function mostrarError(mensaje) {
  outputDiv.innerHTML = `
    <div class="error-container">
      <h3>❌ Error</h3>
      <div class="error-message">${mensaje.replace(/\n/g, '<br>')}</div>
      <button class="btn-retry" onclick="limpiarError()">
        Entendido
      </button>
    </div>
  `;
  outputDiv.className = "output error";
}

function limpiarError() {
  outputDiv.innerHTML = '';
  outputDiv.className = "output";
  inputField.focus();
}

function formatearMarkdown(texto) {
  if (!texto) return '';
  
  return texto
    // Encabezados
    .replace(/^### (.*$)/gim, '<h4>$1</h4>')
    .replace(/^## (.*$)/gim, '<h3>$1</h3>')
    .replace(/^# (.*$)/gim, '<h2>$1</h2>')
    
    // Negritas y cursivas
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    
    // Listas
    .replace(/^\- (.*$)/gim, '<li>$1</li>')
    .replace(/^\* (.*$)/gim, '<li>$1</li>')
    .replace(/^\d+\. (.*$)/gim, '<li>$1</li>')
    
    // Saltos de línea
    .replace(/\n/g, '<br>');
}

function deshabilitarFormulario(deshabilitar) {
  if (submitBtn) {
    submitBtn.disabled = deshabilitar;
    submitBtn.textContent = deshabilitar ? "Generando..." : "Generar propuesta";
  }
  if (inputField) {
    inputField.disabled = deshabilitar;
  }
}

// ==================== FUNCIONES GLOBALES (para onclick) ====================
window.copiarRespuesta = function() {
  const contenido = document.querySelector('.response-content');
  if (contenido) {
    const texto = contenido.innerText;
    navigator.clipboard.writeText(texto).then(() => {
      const btn = document.querySelector('.btn-copy');
      const textoOriginal = btn.textContent;
      btn.textContent = "✅ ¡Copiado!";
      setTimeout(() => {
        btn.textContent = textoOriginal;
      }, 2000);
    }).catch(err => {
      alert('Error al copiar: ' + err);
    });
  }
};

window.nuevaConsulta = function() {
  inputField.value = '';
  outputDiv.innerHTML = '';
  outputDiv.className = "output";
  inputField.focus();
};

// ==================== UTILIDADES ====================
// Verificar conexión al cargar la página
window.addEventListener('load', () => {
  verificarConexion();
});

async function verificarConexion() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log("✅ Backend conectado:", data);
      
      if (!data.rag_activo) {
        console.warn("⚠️ RAG no está activo. Verifica que vector_db esté cargado.");
      }
    } else {
      console.warn("⚠️ Backend responde pero con estado:", response.status);
    }
  } catch (error) {
    console.error("❌ No se pudo conectar al backend:", error);
    // No mostramos error al usuario aquí para no ser intrusivos
  }
}