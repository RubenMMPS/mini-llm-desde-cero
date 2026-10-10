import os
import time
import requests

IDS_LIBROS = [
    2000,   # Don Quijote
    75472,  # Vida de Don Quijote y Sancho
    52894,  # Azul (Ruben Dario)
    63412,  # Lo prohibido (novela completa, Galdos)
    61202,  # Novelas ejemplares (Cervantes)
    52392, 17340, 29506, 18005, 17223, 66263, 77337,
    5937, 50216, 38124, 17013, 55589, 54521, 29468,
]
LIBROS = [(f"gutenberg_{i}.txt", f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8") for i in IDS_LIBROS]

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
        try:
            texto = descargar_libro(url)
        except requests.RequestException as error:
            print(f"  FALLO ({error}); se salta este libro.")
            continue

        with open(ruta_destino, "w", encoding="utf-8") as f:
            f.write(texto)

        print(f"  Guardado en {ruta_destino} ({len(texto):,} caracteres)")

        # Pequena pausa entre descargas: buena practica para no saturar el servidor
        time.sleep(1)


if __name__ == "__main__":
    main()