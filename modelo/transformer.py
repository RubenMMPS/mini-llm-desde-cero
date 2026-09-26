import torch
import torch.nn as nn

from tokenizador import TokenizadorCaracter


class ModeloTransformer(nn.Module):
    """Transformer decoder-only minimo para prediccion del siguiente caracter."""

    def __init__(
        self,
        tamano_vocabulario,
        dimension_embedding=32,
        num_cabezas=4,
        num_capas=2,
        longitud_maxima=256,
    ):
        """
        Args:
            tamano_vocabulario: numero de caracteres distintos.
            dimension_embedding: tamano del vector de cada token (debe ser
                divisible entre num_cabezas, porque se reparte entre ellas).
            num_cabezas: numero de "cabezas" de atencion en paralelo.
            num_capas: cuantos bloques Transformer se apilan.
            longitud_maxima: longitud maxima de secuencia soportada
                (necesaria para crear la tabla de embeddings posicionales).
        """
        super().__init__()

        self.longitud_maxima = longitud_maxima

        self.embedding_token = nn.Embedding(tamano_vocabulario, dimension_embedding)
        self.embedding_posicion = nn.Embedding(longitud_maxima, dimension_embedding)

        capa_transformer = nn.TransformerEncoderLayer(
            d_model=dimension_embedding,
            nhead=num_cabezas,
            dim_feedforward=dimension_embedding * 4,
            batch_first=True,
        )
        self.bloques_transformer = nn.TransformerEncoder(capa_transformer, num_layers=num_capas)

        self.capa_salida = nn.Linear(dimension_embedding, tamano_vocabulario)

    def forward(self, indices_entrada):
        """
        Args:
            indices_entrada: tensor de shape (batch, longitud_secuencia).

        Returns:
            logits: tensor de shape (batch, longitud_secuencia, tamano_vocabulario).
        """
        batch, longitud_secuencia = indices_entrada.shape
        assert longitud_secuencia <= self.longitud_maxima, (
            f"La secuencia ({longitud_secuencia}) supera longitud_maxima "
            f"({self.longitud_maxima})"
        )

        posiciones = torch.arange(longitud_secuencia, device=indices_entrada.device)

        # Sumamos embedding de token + embedding de posicion
        x = self.embedding_token(indices_entrada) + self.embedding_posicion(posiciones)

        # Mascara causal: impide que cada posicion "vea" posiciones futuras
        mascara_causal = nn.Transformer.generate_square_subsequent_mask(longitud_secuencia).to(
            indices_entrada.device
        )

        x = self.bloques_transformer(x, mask=mascara_causal, is_causal=True)

        logits = self.capa_salida(x)
        return logits


if __name__ == "__main__":
    corpus_ejemplo = "el gato se sento en la alfombra"
    tokenizador = TokenizadorCaracter(corpus_ejemplo)

    modelo = ModeloTransformer(tamano_vocabulario=tokenizador.tamano_vocabulario)

    texto = "el gato"
    indices = tokenizador.encode(texto)
    tensor_entrada = torch.tensor(indices).unsqueeze(0)

    print(f"Shape de entrada: {tensor_entrada.shape}  (1 secuencia, {len(indices)} caracteres)")

    logits = modelo(tensor_entrada)
    print(f"Shape de logits:  {logits.shape}  (1 secuencia, {len(indices)} caracteres, {tokenizador.tamano_vocabulario} puntuaciones por caracter)")

    # A diferencia de la LSTM, aqui no hay estado oculto que arrastrar:
    # cada llamada procesa la secuencia completa de una vez.
    logits_ultimo_caracter = logits[0, -1, :]
    caracter_mas_probable = tokenizador.decode([logits_ultimo_caracter.argmax().item()])
    print(f"\nCaracter con mayor puntuacion ahora mismo (sin entrenar): {caracter_mas_probable!r}")

    num_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"\nNumero total de parametros del modelo: {num_parametros:,}")