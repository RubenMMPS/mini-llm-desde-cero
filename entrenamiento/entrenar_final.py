import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modelo"))

import torch
import torch.nn as nn

from tokenizador import TokenizadorCaracter
from transformer import ModeloTransformer

# -----------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------
RUTA_CORPUS = os.path.join(os.path.dirname(__file__), "..", "datos", "corpus_limpio.txt")
RUTA_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "modelo", "checkpoint_final.pt")

# -----------------------------------------------------------------
# Hiperparametros
# -----------------------------------------------------------------
LONGITUD_SECUENCIA = 128     # "ventana de contexto" que el modelo aprende a usar
TAMANO_BATCH = 32
DIMENSION_EMBEDDING = 128
NUM_CABEZAS = 4
NUM_CAPAS = 4
NUM_ITERACIONES = 10000
INTERVALO_EVALUACION = 200
ITERACIONES_EVALUACION = 50  # cuantos lotes promediar al medir train/val loss
TASA_APRENDIZAJE = 3e-4

# Modo de prueba rapida: activa con la variable de entorno PRUEBA_RAPIDA=1
# Reduce drasticamente las iteraciones para verificar que todo funciona
# en un par de minutos, antes de lanzar el entrenamiento completo (horas).
if os.environ.get("PRUEBA_RAPIDA") == "1":
    NUM_ITERACIONES = 100
    INTERVALO_EVALUACION = 25
    ITERACIONES_EVALUACION = 10
    print(">>> MODO PRUEBA RAPIDA ACTIVADO (PRUEBA_RAPIDA=1) <<<\n")

# Deteccion automatica de dispositivo: usa GPU con CUDA si esta disponible,
# si no, CPU. 
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cargar_datos():
    with open(RUTA_CORPUS, "r", encoding="utf-8") as f:
        corpus = f.read()

    tokenizador = TokenizadorCaracter(corpus)
    datos = torch.tensor(tokenizador.encode(corpus), dtype=torch.long)

    # Split 90/10: el 10% final del corpus nunca se usa para entrenar
    n = int(0.9 * len(datos))
    datos_entrenamiento = datos[:n]
    datos_validacion = datos[n:]

    return tokenizador, datos_entrenamiento, datos_validacion


def obtener_batch(datos_entrenamiento, datos_validacion, split):
    """Muestrea un lote de fragmentos aleatorios de longitud fija."""
    datos_split = datos_entrenamiento if split == "train" else datos_validacion

    indices_inicio = torch.randint(len(datos_split) - LONGITUD_SECUENCIA, (TAMANO_BATCH,))
    x = torch.stack([datos_split[i:i + LONGITUD_SECUENCIA] for i in indices_inicio])
    y = torch.stack([datos_split[i + 1:i + LONGITUD_SECUENCIA + 1] for i in indices_inicio])

    return x.to(DISPOSITIVO), y.to(DISPOSITIVO)


@torch.no_grad()
def estimar_perdida(modelo, datos_entrenamiento, datos_validacion, funcion_perdida, tamano_vocabulario):
    """Promedia el loss sobre varios lotes, en modo evaluacion, sin gradientes."""
    modelo.eval()
    perdidas = {}

    for split in ["train", "val"]:
        valores = torch.zeros(ITERACIONES_EVALUACION)
        for k in range(ITERACIONES_EVALUACION):
            x, y = obtener_batch(datos_entrenamiento, datos_validacion, split)
            logits = modelo(x)
            loss = funcion_perdida(logits.view(-1, tamano_vocabulario), y.view(-1))
            valores[k] = loss.item()
        perdidas[split] = valores.mean().item()

    modelo.train()
    return perdidas


@torch.no_grad()
def generar_texto(modelo, tokenizador, texto_inicial, num_caracteres=200, temperatura=0.8):
    """Genera texto, recortando el contexto a LONGITUD_SECUENCIA (el Transformer
    no puede atender a mas contexto del que se le definio en longitud_maxima)."""
    modelo.eval()
    indices = tokenizador.encode(texto_inicial)

    for _ in range(num_caracteres):
        # Si la secuencia generada supera la ventana de contexto, recortamos
        # por delante y nos quedamos solo con los ultimos LONGITUD_SECUENCIA caracteres
        contexto = indices[-LONGITUD_SECUENCIA:]
        tensor_actual = torch.tensor(contexto).unsqueeze(0).to(DISPOSITIVO)

        logits = modelo(tensor_actual)
        ultimo_logit = logits[0, -1, :] / temperatura
        probabilidades = torch.softmax(ultimo_logit, dim=0)
        siguiente_indice = torch.multinomial(probabilidades, num_samples=1).item()
        indices.append(siguiente_indice)

    modelo.train()
    return tokenizador.decode(indices)


def guardar_checkpoint(modelo, tokenizador):
    os.makedirs(os.path.dirname(RUTA_CHECKPOINT), exist_ok=True)
    torch.save(
        {
            "modelo_state": modelo.state_dict(),
            "vocabulario": tokenizador.caracter_a_indice,
            "hiperparametros": {
                "dimension_embedding": DIMENSION_EMBEDDING,
                "num_cabezas": NUM_CABEZAS,
                "num_capas": NUM_CAPAS,
                "longitud_maxima": LONGITUD_SECUENCIA,
            },
        },
        RUTA_CHECKPOINT,
    )


def main():
    print(f"Usando dispositivo: {DISPOSITIVO}")

    tokenizador, datos_entrenamiento, datos_validacion = cargar_datos()
    print(f"Corpus: {len(datos_entrenamiento) + len(datos_validacion):,} caracteres "
          f"({len(datos_entrenamiento):,} entrenamiento / {len(datos_validacion):,} validacion)")
    print(f"Tamano del vocabulario: {tokenizador.tamano_vocabulario}\n")

    modelo = ModeloTransformer(
        tamano_vocabulario=tokenizador.tamano_vocabulario,
        dimension_embedding=DIMENSION_EMBEDDING,
        num_cabezas=NUM_CABEZAS,
        num_capas=NUM_CAPAS,
        longitud_maxima=LONGITUD_SECUENCIA,
    ).to(DISPOSITIVO)

    num_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Numero de parametros del modelo: {num_parametros:,}\n")

    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    funcion_perdida = nn.CrossEntropyLoss()

    tiempo_inicio = time.time()
    for iteracion in range(1, NUM_ITERACIONES + 1):
        x, y = obtener_batch(datos_entrenamiento, datos_validacion, "train")
        logits = modelo(x)
        loss = funcion_perdida(logits.view(-1, tokenizador.tamano_vocabulario), y.view(-1))

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        if iteracion % INTERVALO_EVALUACION == 0 or iteracion == 1:
            perdidas = estimar_perdida(modelo, datos_entrenamiento, datos_validacion, funcion_perdida, tokenizador.tamano_vocabulario)
            perplejidad_val = torch.exp(torch.tensor(perdidas["val"])).item()
            tiempo_transcurrido = time.time() - tiempo_inicio
            print(
                f"Iter {iteracion:5d} | train loss {perdidas['train']:.4f} | "
                f"val loss {perdidas['val']:.4f} | perplejidad val {perplejidad_val:.2f} | "
                f"{tiempo_transcurrido:.0f}s"
            )

            # Checkpoint periodico: si el entrenamiento se interrumpe, no se pierde
            # todo el progreso (sobreescribe el mismo archivo cada vez)
            guardar_checkpoint(modelo, tokenizador)

    # Checkpoint final (redundante con el ultimo periodico, pero explicito)
    guardar_checkpoint(modelo, tokenizador)
    print(f"\nCheckpoint guardado en: {RUTA_CHECKPOINT}")

    # --- Generacion final de muestra ---
    print("\n=== Texto generado con el modelo final ===")
    for inicio in ["capitulo", "don quijote", "sancho"]:
        print(f"\n'{inicio}' ->\n{generar_texto(modelo, tokenizador, inicio)}")


if __name__ == "__main__":
    main()