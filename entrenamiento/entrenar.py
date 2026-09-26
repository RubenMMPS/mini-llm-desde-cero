import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modelo"))

import torch
import torch.nn as nn

from tokenizador import TokenizadorCaracter
from lstm import ModeloLSTM
from transformer import ModeloTransformer

# -----------------------------------------------------------------
# 1. Corpus y preparacion de datos (igual que antes, compartido por ambos modelos)
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

indices_completos = tokenizador.encode(CORPUS)
tensor_entrada = torch.tensor(indices_completos[:-1]).unsqueeze(0)
tensor_objetivo = torch.tensor(indices_completos[1:]).unsqueeze(0)

funcion_perdida = nn.CrossEntropyLoss()
NUM_EPOCAS = 300


# -----------------------------------------------------------------
# 2. Forward pass generico: cada arquitectura devuelve algo ligeramente
#    distinto (la LSTM tambien devuelve el estado oculto), asi que
#    unificamos aqui la parte que varia.
# -----------------------------------------------------------------
def obtener_logits(modelo, tensor_entrada, tipo_modelo):
    if tipo_modelo == "lstm":
        logits, _ = modelo(tensor_entrada)
    else:  # "transformer"
        logits = modelo(tensor_entrada)
    return logits


def entrenar_modelo(modelo, tipo_modelo, nombre):
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.005)

    print(f"\n=== Entrenando {nombre} ===")
    for epoca in range(1, NUM_EPOCAS + 1):
        logits = obtener_logits(modelo, tensor_entrada, tipo_modelo)

        logits_aplanados = logits.view(-1, tokenizador.tamano_vocabulario)
        objetivo_aplanado = tensor_objetivo.view(-1)
        loss = funcion_perdida(logits_aplanados, objetivo_aplanado)

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        if epoca % 60 == 0 or epoca == 1:
            print(f"Epoca {epoca:4d} | loss = {loss.item():.4f}")

    return loss.item()


# -----------------------------------------------------------------
# 3. Generacion de texto: la LSTM puede arrastrar estado oculto paso a
#    paso (eficiente); el Transformer no tiene estado oculto y tiene que
#    reprocesar la secuencia completa generada hasta el momento en cada paso.
# -----------------------------------------------------------------
def generar_texto_lstm(modelo, texto_inicial, num_caracteres=60, temperatura=0.8):
    modelo.eval()
    indices = tokenizador.encode(texto_inicial)
    tensor_actual = torch.tensor(indices).unsqueeze(0)
    estado_oculto = None
    resultado = texto_inicial

    with torch.no_grad():
        logits, estado_oculto = modelo(tensor_actual, estado_oculto)
        for _ in range(num_caracteres):
            ultimo_logit = logits[0, -1, :] / temperatura
            probabilidades = torch.softmax(ultimo_logit, dim=0)
            siguiente_indice = torch.multinomial(probabilidades, num_samples=1).item()
            resultado += tokenizador.decode([siguiente_indice])
            tensor_actual = torch.tensor([[siguiente_indice]])
            logits, estado_oculto = modelo(tensor_actual, estado_oculto)

    modelo.train()
    return resultado


def generar_texto_transformer(modelo, texto_inicial, num_caracteres=60, temperatura=0.8):
    modelo.eval()
    indices = tokenizador.encode(texto_inicial)

    with torch.no_grad():
        for _ in range(num_caracteres):
            # Reprocesamos toda la secuencia generada hasta ahora (sin estado oculto)
            tensor_actual = torch.tensor(indices).unsqueeze(0)
            logits = modelo(tensor_actual)

            ultimo_logit = logits[0, -1, :] / temperatura
            probabilidades = torch.softmax(ultimo_logit, dim=0)
            siguiente_indice = torch.multinomial(probabilidades, num_samples=1).item()
            indices.append(siguiente_indice)

    modelo.train()
    return tokenizador.decode(indices)


# -----------------------------------------------------------------
# 4. Entrenamos y comparamos ambas arquitecturas
# -----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    modelo_lstm = ModeloLSTM(tamano_vocabulario=tokenizador.tamano_vocabulario)
    loss_final_lstm = entrenar_modelo(modelo_lstm, "lstm", "LSTM")

    # longitud_maxima debe cubrir la secuencia de entrenamiento completa (373 caracteres)
    # mas margen para la generacion posterior
    modelo_transformer = ModeloTransformer(
        tamano_vocabulario=tokenizador.tamano_vocabulario,
        longitud_maxima=512,
    )
    loss_final_transformer = entrenar_modelo(modelo_transformer, "transformer", "Transformer")

    print("\n=== Comparacion de loss final ===")
    print(f"LSTM:        {loss_final_lstm:.4f}")
    print(f"Transformer: {loss_final_transformer:.4f}")

    print("\n=== Texto generado: LSTM ===")
    torch.manual_seed(1)
    for inicio in ["el gato", "el perro", "la nina"]:
        print(f"'{inicio}' -> {generar_texto_lstm(modelo_lstm, inicio)!r}")

    print("\n=== Texto generado: Transformer ===")
    torch.manual_seed(1)
    for inicio in ["el gato", "el perro", "la nina"]:
        print(f"'{inicio}' -> {generar_texto_transformer(modelo_transformer, inicio)!r}")