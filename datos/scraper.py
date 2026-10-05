import os
import time
import requests

# Cada libro: (nombre_archivo, url_texto_plano)
# Ambos son de dominio publico y estan en espanol.
LIBROS = [
    ("don_quijote.txt", "https://www.gutenberg.org/ebooks/2000.txt.utf-8"),
    ("vida_de_don_quijote_y_sancho.txt", "https://www.gutenberg.org/ebooks/75472.txt.utf-8"),
    ("azul.txt", "https://www.gutenberg.org/ebooks/52894.txt.utf-8"),
]

CARPETA_SALIDA = os.path.join(os.path.dirname(__file__), "raw")

CABECERAS = {
    # Identificarse con un User-Agent real es buena practica al hacer scraping
    "User-Agent": "Mozilla/5.0 (proyecto de aprendizaje personal - mini-llm-desde-cero)"
}


def descargar_libro(url: str) -> str:
    """Descarga el contenido en texto plano de una URL de Gutenberg."""
    respuesta = requests.get(url, headers=CABECERAS, timeout=30)
    respuesta.raise_for_status()  # lanza un error si la descarga fallo (404, 500, etc.)
    respuesta.encoding = "utf-8"
    return respuesta.text


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    for nombre_archivo, url in LIBROS:
        ruta_destino = os.path.join(CARPETA_SALIDA, nombre_archivo)

        if os.path.exists(ruta_destino):
            print(f"Ya existe {nombre_archivo}, no se vuelve a descargar.")
            continue

        print(f"Descargando {nombre_archivo} desde {url} ...")
        texto = descargar_libro(url)

        with open(ruta_destino, "w", encoding="utf-8") as f:
            f.write(texto)

        print(f"  Guardado en {ruta_destino} ({len(texto):,} caracteres)")

        # Pequena pausa entre descargas: buena practica para no saturar el servidor
        time.sleep(1)


if __name__ == "__main__":
    main()