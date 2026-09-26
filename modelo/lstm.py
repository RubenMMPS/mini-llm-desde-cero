import torch
import torch.nn as nn

from tokenizador import TokenizadorCaracter


class ModeloLSTM(nn.Module):
    """Modelo generativo a nivel carácter: predice el siguiente token dado el contexto."""

    def __init__(self, tamano_vocabulario, dimension_embedding=32, tamano_oculto=64):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=tamano_vocabulario,
            embedding_dim=dimension_embedding,
        )

        self.lstm = nn.LSTM(
            input_size=dimension_embedding,
            hidden_size=tamano_oculto,
            batch_first=True,
        )

        self.capa_salida = nn.Linear(tamano_oculto, tamano_vocabulario)

    def forward(self, indices_entrada, estado_oculto=None):
        vectores = self.embedding(indices_entrada)
        salida_lstm, nuevo_estado_oculto = self.lstm(vectores, estado_oculto)
        logits = self.capa_salida(salida_lstm)
        return logits, nuevo_estado_oculto


if __name__ == "__main__":
    corpus_ejemplo = "el gato se sento en la alfombra"
    tokenizador = TokenizadorCaracter(corpus_ejemplo)

    modelo = ModeloLSTM(tamano_vocabulario=tokenizador.tamano_vocabulario)

    texto = "el gato"
    indices = tokenizador.encode(texto)

    tensor_entrada = torch.tensor(indices).unsqueeze(0)
    print(f"Shape de entrada:  {tensor_entrada.shape}  (1 secuencia, {len(indices)} caracteres)")

    logits, estado_oculto = modelo(tensor_entrada)
    print(f"Shape de logits:   {logits.shape}")

    h, c = estado_oculto
    print(f"Shape estado oculto (h): {h.shape}")
    print(f"Shape estado de celda (c): {c.shape}")

    logits_ultimo_caracter = logits[0, -1, :]
    print(f"\nLogits para predecir el caracter siguiente a {texto!r} (sin entrenar, son ruido aleatorio):")
    print(logits_ultimo_caracter)

    caracter_mas_probable = tokenizador.decode([logits_ultimo_caracter.argmax().item()])
    print(f"\nCaracter con mayor puntuacion ahora mismo: {caracter_mas_probable!r}")
    print("(sin sentido todavia -- el modelo no ha visto ni un solo ejemplo de entrenamiento)")