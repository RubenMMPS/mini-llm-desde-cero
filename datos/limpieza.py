import os
import re
import glob

CARPETA_RAW = os.path.join(os.path.dirname(__file__), "raw")
RUTA_CORPUS_FINAL = os.path.join(os.path.dirname(__file__), "corpus_limpio.txt")

# Los marcadores de inicio/fin de Project Gutenberg son siempre de esta forma,
# solo cambia el titulo del libro en medio de los asteriscos.
PATRON_INICIO = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
PATRON_FIN = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

# Vocabulario permitido: letras (incluyendo acentos y enie), digitos, puntuacion
# basica y espacios en blanco. Cualquier otro caracter se elimina.
CARACTERES_PERMITIDOS = re.compile(r"[^a-zñáéíóúü0-9.,;:!?¡¿'\"\-\s]")


def extraer_texto_del_libro(texto_crudo: str) -> str:
    """Recorta la cabecera y el pie legal de Project Gutenberg, dejando solo la obra."""
    inicio = PATRON_INICIO.search(texto_crudo)
    fin = PATRON_FIN.search(texto_crudo)

    if not inicio or not fin:
        raise ValueError(
            "No se encontraron los marcadores de inicio/fin de Gutenberg. "
            "¿El archivo se descargo correctamente?"
        )

    # Nos quedamos con el texto que hay ENTRE el final del marcador de inicio
    # y el principio del marcador de fin
    return texto_crudo[inicio.end():fin.start()]


def limpiar_texto(texto: str) -> str:
    """Normaliza el texto: minusculas, restringe caracteres, colapsa espacios."""
    texto = texto.lower()
    texto = CARACTERES_PERMITIDOS.sub(" ", texto)   # fuera simbolos no deseados
    texto = re.sub(r"[ \t]+", " ", texto)           # varios espacios/tabs -> uno
    texto = re.sub(r"\n\s*\n+", "\n\n", texto)      # varias lineas en blanco -> una
    texto = re.sub(r" *\n *", "\n", texto)           # sin espacios alrededor de saltos de linea
    return texto.strip()


def procesar_todos_los_libros() -> str:
    """Procesa cada archivo en raw/, los limpia y los combina en un solo corpus."""
    rutas_libros = sorted(glob.glob(os.path.join(CARPETA_RAW, "*.txt")))

    if not rutas_libros:
        raise FileNotFoundError(
            f"No hay archivos .txt en {CARPETA_RAW}. Ejecuta antes scraper.py"
        )

    textos_limpios = []
    for ruta in rutas_libros:
        with open(ruta, "r", encoding="utf-8") as f:
            texto_crudo = f.read()

        texto_obra = extraer_texto_del_libro(texto_crudo)
        texto_limpio = limpiar_texto(texto_obra)

        print(f"{os.path.basename(ruta)}: {len(texto_crudo):,} caracteres crudos -> {len(texto_limpio):,} limpios")
        textos_limpios.append(texto_limpio)

    return "\n\n".join(textos_limpios)


if __name__ == "__main__":
    corpus_final = procesar_todos_los_libros()

    with open(RUTA_CORPUS_FINAL, "w", encoding="utf-8") as f:
        f.write(corpus_final)

    print(f"\nCorpus final combinado: {len(corpus_final):,} caracteres")
    print(f"Guardado en: {RUTA_CORPUS_FINAL}")