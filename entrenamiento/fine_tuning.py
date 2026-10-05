import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modelo"))

import torch
import torch.nn as nn

from tokenizador import TokenizadorCaracter
from transformer import ModeloTransformer

RUTA_CHECKPOINT_BASE = os.path.join(os.path.dirname(__file__), "..", "modelo", "checkpoint_final.pt")
RUTA_CHECKPOINT_DIALOGO = os.path.join(os.path.dirname(__file__), "..", "modelo", "checkpoint_dialogo.pt")
RUTA_DIALOGO = os.path.join(os.path.dirname(__file__), "..", "datos", "dialogo_formateado.txt")

# Learning rate MUCHO mas bajo que en el entrenamiento desde cero (3e-4):
# queremos un ajuste suave, no reescribir lo ya aprendido
TASA_APRENDIZAJE = 5e-5
NUM_ITERACIONES = 2000
INTERVALO_IMPRESION = 200


def main():
    # --- Cargar el modelo base ya entrenado ---
    checkpoint = torch.load(RUTA_CHECKPOINT_BASE, map_location="cpu", weights_only=True)
    hp = checkpoint["hiperparametros"]

    tokenizador = TokenizadorCaracter.desde_vocabulario(checkpoint["vocabulario"])
    longitud_maxima = hp["longitud_maxima"]

    modelo = ModeloTransformer(
        tamano_vocabulario=tokenizador.tamano_vocabulario,
        dimension_embedding=hp["dimension_embedding"],
        num_cabezas=hp["num_cabezas"],
        num_capas=hp["num_capas"],
        longitud_maxima=longitud_maxima,
    )
    modelo.load_state_dict(checkpoint["modelo_state"])

    # --- Cargar el dialogo como UN SOLO texto continuo (no ejemplos sueltos) ---
    with open(RUTA_DIALOGO, "r", encoding="utf-8") as f:
        texto_dialogo = f.read()

    desconocidos = sorted({c for c in texto_dialogo if c not in tokenizador.caracter_a_indice})
    if desconocidos:
        sys.exit(
            f"Error: estos caracteres no existen en el vocabulario del modelo base: {desconocidos}\n"
            "(revisa datos/preparar_dialogo.py -- deberia haberlos filtrado)"
        )

    indices = tokenizador.encode(texto_dialogo)
    longitud_ventana = min(longitud_maxima, len(indices) - 1)

    if longitud_ventana < 10:
        sys.exit("El dialogo formateado es demasiado corto para entrenar. Añade mas intercambios.")

    print(f"Texto de dialogo: {len(texto_dialogo):,} caracteres | ventana de entrenamiento: {longitud_ventana}")

    def obtener_ventana_aleatoria():
        inicio = torch.randint(0, len(indices) - longitud_ventana, (1,)).item()
        x = torch.tensor(indices[inicio:inicio + longitud_ventana]).unsqueeze(0)
        y = torch.tensor(indices[inicio + 1:inicio + longitud_ventana + 1]).unsqueeze(0)
        return x, y

    # --- Fine-tuning ---
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    funcion_perdida = nn.CrossEntropyLoss()

    modelo.train()
    for iteracion in range(1, NUM_ITERACIONES + 1):
        x, y = obtener_ventana_aleatoria()

        logits = modelo(x)
        loss = funcion_perdida(logits.view(-1, tokenizador.tamano_vocabulario), y.view(-1))

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        if iteracion % INTERVALO_IMPRESION == 0 or iteracion == 1:
            print(f"Iter {iteracion:5d} | loss {loss.item():.4f}")

    # --- Guardar como checkpoint NUEVO (no sobreescribimos el modelo base) ---
    torch.save(
        {
            "modelo_state": modelo.state_dict(),
            "vocabulario": tokenizador.caracter_a_indice,
            "hiperparametros": hp,
        },
        RUTA_CHECKPOINT_DIALOGO,
    )
    print(f"\nCheckpoint de dialogo guardado en: {RUTA_CHECKPOINT_DIALOGO}")
    print("(el checkpoint_final.pt original NO se ha modificado)")


if __name__ == "__main__":
    main()