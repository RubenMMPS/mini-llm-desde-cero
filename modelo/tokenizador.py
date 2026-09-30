class TokenizadorCaracter:
    """Tokenizador que trata cada carácter individual como un token."""

    def __init__(self, corpus: str):
        """
        Construye el vocabulario a partir de un corpus de texto.

        Args:
            corpus: texto completo sobre el que se construye el vocabulario.
                    Debe ser representativo de todo el texto que se usará
                    después (si aparece un carácter nuevo más adelante que
                    no estaba aquí, el tokenizador no sabrá codificarlo).
        """
        caracteres_unicos = sorted(set(corpus))

        # Diccionarios de conversión en ambas direcciones
        self.caracter_a_indice = {c: i for i, c in enumerate(caracteres_unicos)}
        self.indice_a_caracter = {i: c for i, c in enumerate(caracteres_unicos)}
        self.tamano_vocabulario = len(caracteres_unicos)

    @classmethod
    def desde_vocabulario(cls, caracter_a_indice: dict):
        """Reconstruye un tokenizador a partir de un vocabulario ya guardado
        (por ejemplo, el que se almacena en el checkpoint), sin necesitar el corpus."""
        tokenizador = cls.__new__(cls)
        tokenizador.caracter_a_indice = dict(caracter_a_indice)
        tokenizador.indice_a_caracter = {i: c for c, i in caracter_a_indice.items()}
        tokenizador.tamano_vocabulario = len(caracter_a_indice)
        return tokenizador

    def encode(self, texto: str) -> list[int]:
        """Convierte una cadena de texto en una lista de índices enteros."""
        return [self.caracter_a_indice[c] for c in texto]

    def decode(self, indices: list[int]) -> str:
        """Convierte una lista de índices enteros de vuelta a texto."""
        return "".join(self.indice_a_caracter[i] for i in indices)


if __name__ == "__main__":
    # Demo rápida para verificar que el tokenizador funciona como se espera
    corpus_ejemplo = "el gato se sentó en la alfombra"
    tokenizador = TokenizadorCaracter(corpus_ejemplo)

    print(f"Tamaño del vocabulario: {tokenizador.tamano_vocabulario}")
    print(f"Vocabulario: {tokenizador.caracter_a_indice}\n")

    texto_prueba = "el gato"
    codificado = tokenizador.encode(texto_prueba)
    decodificado = tokenizador.decode(codificado)

    print(f"Texto original:  {texto_prueba!r}")
    print(f"Codificado:      {codificado}")
    print(f"Decodificado:    {decodificado!r}")
    assert decodificado == texto_prueba, "El decode debería recuperar el texto original"
    print("\nVerificación correcta: decode(encode(texto)) == texto")