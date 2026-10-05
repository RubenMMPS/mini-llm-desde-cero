import os

from limpieza import limpiar_texto

RUTA_SEMILLA = os.path.join(os.path.dirname(__file__), "dialogo_semilla.txt")
RUTA_SALIDA = os.path.join(os.path.dirname(__file__), "dialogo_formateado.txt")


def main():
    with open(RUTA_SEMILLA, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques_crudos = [b.strip() for b in contenido.split("---") if b.strip()]
    bloques_limpios = [limpiar_texto(b) for b in bloques_crudos]

    # Cada intercambio separado por una linea en blanco: asi
    # fine_tuning.py puede dividirlos de nuevo en ejemplos independientes
    texto_final = "\n\n".join(bloques_limpios)

    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        f.write(texto_final)

    print(f"{len(bloques_limpios)} intercambios procesados -> {RUTA_SALIDA}")


if __name__ == "__main__":
    main()