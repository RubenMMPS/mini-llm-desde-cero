"""
Carga el checkpoint entrenado (checkpoint_final.pt) y genera texto,
sin necesidad de reentrenar ni de tener el corpus a mano.

Ejemplo de Uso:
    python modelo/generar.py --texto "don quijote" --num-caracteres 300
    python modelo/generar.py --texto "sancho" --temperatura 0.7 --top-k 10
"""

import argparse
import os
import sys

import torch

from tokenizador import TokenizadorCaracter
from transformer import ModeloTransformer

RUTA_CHECKPOINT_POR_DEFECTO = os.path.join(os.path.dirname(__file__), "checkpoint_final.pt")


def cargar_modelo(ruta_checkpoint):
    """Reconstruye modelo y tokenizador a partir de un checkpoint guardado."""
    checkpoint = torch.load(ruta_checkpoint, map_location="cpu", weights_only=True)

    tokenizador = TokenizadorCaracter.desde_vocabulario(checkpoint["vocabulario"])
    hp = checkpoint["hiperparametros"]

    modelo = ModeloTransformer(
        tamano_vocabulario=tokenizador.tamano_vocabulario,
        dimension_embedding=hp["dimension_embedding"],
        num_cabezas=hp["num_cabezas"],
        num_capas=hp["num_capas"],
        longitud_maxima=hp["longitud_maxima"],
    )
    modelo.load_state_dict(checkpoint["modelo_state"])
    modelo.eval()  # modo evaluacion: sin comportamientos propios del entrenamiento

    return modelo, tokenizador, hp["longitud_maxima"]


@torch.no_grad()
def siguiente_indice(modelo, indices, longitud_contexto, temperatura, top_k):
    """Muestrea el indice del siguiente caracter dado el texto visto hasta ahora.

    Es el paso elemental de la generacion; lo comparten generar() (CLI) y
    generar_respuesta() (chat), para no duplicar la logica de muestreo.

    temperatura: <1 hace el modelo mas conservador, >1 mas arriesgado/caotico.
    top_k: si se indica, solo se puede elegir entre los k caracteres mas probables
           (descarta la 'cola' de caracteres muy improbables que producen errores).
    """
    # El Transformer solo puede atender a longitud_contexto caracteres
    contexto = indices[-longitud_contexto:]
    entrada = torch.tensor(contexto).unsqueeze(0)

    logits = modelo(entrada)[0, -1, :] / temperatura

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        valores_top, _ = torch.topk(logits, k)
        logits[logits < valores_top[-1]] = float("-inf")

    probabilidades = torch.softmax(logits, dim=0)
    return torch.multinomial(probabilidades, num_samples=1).item()


@torch.no_grad()
def generar(modelo, tokenizador, longitud_contexto, texto_inicial, num_caracteres, temperatura, top_k):
    """Genera exactamente num_caracteres caracteres a continuacion de texto_inicial."""
    indices = tokenizador.encode(texto_inicial)

    for _ in range(num_caracteres):
        indices.append(siguiente_indice(modelo, indices, longitud_contexto, temperatura, top_k))

    return tokenizador.decode(indices)


@torch.no_grad()
def generar_respuesta(modelo, tokenizador, longitud_contexto, prompt, max_caracteres, temperatura, top_k):
    """Genera la respuesta a un prompt de dialogo, parando en el primer salto de linea.

    En el formato del fine-tuning cada respuesta ocupa una sola linea, asi que el
    salto de linea es la "senal de fin de turno". Parar ahi es mas robusto (y mas
    barato) que generar una longitud fija y recortar despues.

    Devuelve (respuesta, truncada): truncada=True si se alcanzo max_caracteres
    sin que el modelo cerrara la linea.
    """
    indices = tokenizador.encode(prompt)
    generados = []

    for _ in range(max_caracteres):
        indice = siguiente_indice(modelo, indices, longitud_contexto, temperatura, top_k)
        indices.append(indice)
        caracter = tokenizador.indice_a_caracter[indice]

        # Ignoramos saltos de linea iniciales: solo cuenta como "fin" si ya hay contenido
        if caracter == "\n" and "".join(generados).strip():
            return "".join(generados).strip(), False
        generados.append(caracter)

    return "".join(generados).strip(), True


def main():
    parser = argparse.ArgumentParser(description="Genera texto con el mini-GPT entrenado")
    parser.add_argument("--texto", default="don quijote", help="texto inicial (se pasa a minusculas)")
    parser.add_argument("--num-caracteres", type=int, default=300)
    parser.add_argument("--temperatura", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--semilla", type=int, default=None, help="para resultados reproducibles")
    parser.add_argument("--checkpoint", default=RUTA_CHECKPOINT_POR_DEFECTO)
    parser.add_argument(
        "--dialogo", action="store_true",
        help="envuelve --texto como 'tu: <texto>\\nquijote:' y recorta la respuesta "
             "en el siguiente turno (requiere un checkpoint afinado para dialogo)",
    )
    args = parser.parse_args()

    if args.temperatura <= 0:
        sys.exit("Error: --temperatura debe ser mayor que 0")

    if args.semilla is not None:
        torch.manual_seed(args.semilla)

    modelo, tokenizador, longitud_contexto = cargar_modelo(args.checkpoint)

    # El corpus de entrenamiento estaba en minusculas (lo hizo limpieza.py)
    texto = args.texto.lower()
    desconocidos = sorted({c for c in texto if c not in tokenizador.caracter_a_indice})
    if desconocidos:
        sys.exit(f"Error: estos caracteres no estan en el vocabulario del modelo: {desconocidos}")

    if args.dialogo:
        texto = f"tu: {texto}\nquijote:"

    resultado = generar(modelo, tokenizador, longitud_contexto, texto,
                         args.num_caracteres, args.temperatura, args.top_k)

    if args.dialogo:
        # Cortamos justo antes de que el modelo "invente" el siguiente turno
        # del usuario (patron que debería haber aprendido en el fine-tuning)
        marcador = "\ntu:"
        posicion_corte = resultado.find(marcador, len(texto))
        if posicion_corte != -1:
            resultado = resultado[:posicion_corte]

    print(resultado)


if __name__ == "__main__":
    main()