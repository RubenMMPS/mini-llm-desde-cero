"""
entrenamiento/entrenar.py

Bucle de entrenamiento del ModeloLSTM sobre un corpus pequeno escrito a mano.

Objetivo de esta fase: verificar que el ciclo completo
(forward -> loss -> backward -> optimizador) funciona y que el modelo
aprende patrones reales del texto (no verificar la calidad final del
texto generado, para eso hara falta un corpus mucho mas grande via scraping,
en la Fase 7).
"""

import sys
import os

# Permite importar los modulos de la carpeta modelo/ desde aqui
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modelo"))

import torch
import torch.nn as nn

from tokenizador import TokenizadorCaracter
from lstm import ModeloLSTM

# -----------------------------------------------------------------
# 1. Corpus de entrenamiento (ampliado respecto al de fases anteriores,
#    con repeticion de patrones para que haya algo real que aprender)
# -----------------------------------------------------------------
CORPUS = """
el gato se sento en la alfombra
el perro se sento en la silla
el gato se escondio detras del sofa
el perro corrio detras del gato
el gato subio al arbol
la nina acaricio al gato
el perro ladro al gato
el gato maullo y se escondio
el gato duerme en la alfombra
el perro duerme en la silla
la nina jugo con el gato
la nina jugo con el perro
el gato y el perro corrieron juntos
""".strip().lower()

tokenizador = TokenizadorCaracter(CORPUS)
print(f"Tamano del corpus: {len(CORPUS)} caracteres")
print(f"Tamano del vocabulario: {tokenizador.tamano_vocabulario}\n")

# -----------------------------------------------------------------
# 2. Preparacion de datos: entrada y objetivo desplazados una posicion
# -----------------------------------------------------------------
indices_completos = tokenizador.encode(CORPUS)

secuencia_entrada = indices_completos[:-1]
secuencia_objetivo = indices_completos[1:]

# Anadimos la dimension de batch (aqui, batch de tamano 1: todo el corpus
# como una unica secuencia larga)
tensor_entrada = torch.tensor(secuencia_entrada).unsqueeze(0)
tensor_objetivo = torch.tensor(secuencia_objetivo).unsqueeze(0)

print(f"Shape entrada:  {tensor_entrada.shape}")
print(f"Shape objetivo: {tensor_objetivo.shape}\n")

# -----------------------------------------------------------------
# 3. Modelo, funcion de perdida y optimizador
# -----------------------------------------------------------------
modelo = ModeloLSTM(tamano_vocabulario=tokenizador.tamano_vocabulario)
funcion_perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.005)

# -----------------------------------------------------------------
# 4. Bucle de entrenamiento
# -----------------------------------------------------------------
NUM_EPOCAS = 300

for epoca in range(1, NUM_EPOCAS + 1):
    # --- Forward pass ---
    logits, _ = modelo(tensor_entrada)  # shape: (1, longitud_secuencia, vocabulario)

    # CrossEntropyLoss espera (N, num_clases) y (N,), asi que "aplanamos"
    # la dimension de batch y secuencia en una sola
    logits_aplanados = logits.view(-1, tokenizador.tamano_vocabulario)
    objetivo_aplanado = tensor_objetivo.view(-1)

    loss = funcion_perdida(logits_aplanados, objetivo_aplanado)

    # --- Backward pass + actualizacion ---
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()

    if epoca % 30 == 0 or epoca == 1:
        print(f"Epoca {epoca:4d} | loss = {loss.item():.4f}")

# -----------------------------------------------------------------
# 5. Generacion de texto con el modelo ya entrenado
# -----------------------------------------------------------------
def generar_texto(modelo, tokenizador, texto_inicial, num_caracteres=60, temperatura=0.8):
    """Genera texto caracter a caracter, muestreando de la distribucion del modelo."""
    modelo.eval()  # modo evaluacion (desactiva comportamientos especificos de entrenamiento)

    indices = tokenizador.encode(texto_inicial)
    tensor_actual = torch.tensor(indices).unsqueeze(0)

    estado_oculto = None
    resultado = texto_inicial

    with torch.no_grad():  # no necesitamos gradientes para generar texto
        # Procesamos el texto inicial para "poner al dia" el estado oculto
        logits, estado_oculto = modelo(tensor_actual, estado_oculto)

        for _ in range(num_caracteres):
            # Tomamos los logits del ultimo caracter procesado
            ultimo_logit = logits[0, -1, :] / temperatura
            probabilidades = torch.softmax(ultimo_logit, dim=0)

            siguiente_indice = torch.multinomial(probabilidades, num_samples=1).item()
            resultado += tokenizador.decode([siguiente_indice])

            # El siguiente caracter generado se convierte en la nueva entrada
            tensor_actual = torch.tensor([[siguiente_indice]])
            logits, estado_oculto = modelo(tensor_actual, estado_oculto)

    modelo.train()  # volvemos a modo entrenamiento por si se seguiria usando el modelo
    return resultado


print("\n--- Texto generado tras el entrenamiento ---")
torch.manual_seed(0)
for inicio in ["el gato", "el perro", "la nina"]:
    print(f"'{inicio}' -> {generar_texto(modelo, tokenizador, inicio)!r}")