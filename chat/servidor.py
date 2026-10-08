"""
Servidor local (FastAPI) que carga el checkpoint afinado para dialogo y
expone un endpoint /api/chat. Sirve tambien el frontend estatico.

Ejecucion:
    uvicorn servidor:app --reload
    (o: python servidor.py)
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modelo"))

import re
import unicodedata
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from generar import cargar_modelo, generar_respuesta

RUTA_CHECKPOINT = os.environ.get(
    "CHECKPOINT_CHAT",
    os.path.join(os.path.dirname(__file__), "..", "modelo", "checkpoint_dialogo.pt"),
)
RUTA_ESTATICOS = os.path.join(os.path.dirname(__file__), "estatico")

MAX_CARACTERES_RESPUESTA = 200
TEMPERATURA = 0.7
TOP_K = 10

app = FastAPI(title="Chat Don Quijote (mini-GPT de aprendizaje)")

# --- Carga del modelo UNA sola vez, al arrancar el servidor ---
if not os.path.exists(RUTA_CHECKPOINT):
    sys.exit(
        f"No se encuentra el checkpoint: {RUTA_CHECKPOINT}\n"
        "Generalo ejecutando entrenamiento/fine_tuning.py"
    )
modelo, tokenizador, longitud_contexto = cargar_modelo(RUTA_CHECKPOINT)
print(f"Modelo cargado. Ventana de contexto: {longitud_contexto} caracteres.")


class Turno(BaseModel):
    rol: Literal["tu", "quijote"]
    texto: str = Field(max_length=500)


class PeticionChat(BaseModel):
    historial: list[Turno] = Field(max_length=40)  # turnos anteriores (sin el mensaje nuevo)
    mensaje: str = Field(max_length=300)            # mensaje nuevo del usuario


def normalizar_texto(texto: str) -> str:
    """Deja el texto en el estilo del dialogo de fine-tuning: minusculas, sin
    tildes ni puntuacion y solo con caracteres que el modelo conoce.

    Por que: la semilla de dialogo no tiene tildes ni signos, asi que una
    entrada como "¿Que opinas?" llegaria al modelo en un formato que nunca vio.
    Ademas evita que alguien meta saltos de linea o ":" y falsee el formato
    tu:/quijote: del prompt. Se aplica a TODOS los turnos, no solo al nuevo:
    sin eso, un caracter desconocido en el historial rompe el chat."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")  # quita tildes
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    texto = re.sub(r" +", " ", texto).strip()
    return "".join(c for c in texto if c in tokenizador.caracter_a_indice)


def construir_transcripcion(historial: list[Turno], mensaje_nuevo: str) -> str:
    """Concatena historial + mensaje nuevo con el MISMO formato exacto del
    fine-tuning (datos/dialogo_formateado.txt):

        tu: <pregunta>
        quijote: <respuesta>
        <linea en blanco>
        tu: <pregunta>
        ...

    Es decir: salto simple entre "tu" y "quijote", y linea en blanco entre
    intercambios. Si el formato difiere del de entrenamiento, el modelo ve
    algo que nunca vio y responde peor."""
    transcripcion = ""
    for turno in historial:
        transcripcion += f"{turno.rol}: {normalizar_texto(turno.texto)}"
        transcripcion += "\n\n" if turno.rol == "quijote" else "\n"
    transcripcion += f"tu: {mensaje_nuevo}\nquijote:"
    return transcripcion


@app.post("/api/chat")
def chat(peticion: PeticionChat):
    mensaje = normalizar_texto(peticion.mensaje)
    if not mensaje:
        raise HTTPException(
            status_code=400,
            detail="El mensaje no tiene letras que el modelo reconozca. Prueba con palabras en espanol.",
        )

    transcripcion = construir_transcripcion(peticion.historial, mensaje)
    # El modelo solo puede atender a los ultimos longitud_contexto caracteres:
    # esta es la "memoria" real de la conversacion, y es deliberadamente visible
    # en el frontend para no generar expectativas falsas.
    contexto_usado = transcripcion[-longitud_contexto:]

    respuesta, truncada = generar_respuesta(
        modelo, tokenizador, longitud_contexto, contexto_usado,
        MAX_CARACTERES_RESPUESTA, TEMPERATURA, TOP_K,
    )

    return {
        "respuesta": respuesta,
        "truncada": truncada,  # True si el modelo no llego a cerrar la linea
        "caracteres_contexto_usados": len(contexto_usado),
        "ventana_contexto": longitud_contexto,
    }


# Sirve el frontend estatico en "/" (debe ir DESPUES de las rutas de /api)
app.mount("/", StaticFiles(directory=RUTA_ESTATICOS, html=True), name="estatico")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)