"""
modelo/embeddings_demo.py

Primer contacto con PyTorch: convertimos los índices que produce nuestro
TokenizadorCaracter en vectores mediante nn.Embedding, y observamos su forma
y su comportamiento antes de que se entrenen (todavía son aleatorios).
"""

import torch
import torch.nn as nn

from tokenizador import TokenizadorCaracter

# -----------------------------------------------------------------
# 1. Preparamos el tokenizador (igual que antes)
# -----------------------------------------------------------------
corpus_ejemplo = "el gato se sentó en la alfombra"
tokenizador = TokenizadorCaracter(corpus_ejemplo)

texto = "el gato"
indices = tokenizador.encode(texto)
print(f"Texto:   {texto!r}")
print(f"Índices: {indices}")

# -----------------------------------------------------------------
# 2. Convertimos la lista de índices en un tensor de PyTorch
#    (la estructura de datos base con la que trabaja PyTorch, similar
#    a un array de NumPy pero con soporte para gradientes y GPU)
# -----------------------------------------------------------------
tensor_indices = torch.tensor(indices)
print(f"\nTensor de índices: {tensor_indices}")
print(f"Shape del tensor:  {tensor_indices.shape}  (7 tokens)")

# -----------------------------------------------------------------
# 3. Creamos la capa de embedding
#    - num_embeddings: cuántos tokens distintos existen (tamaño del vocabulario)
#    - embedding_dim: cuántos números describen a cada token (elegido por nosotros)
# -----------------------------------------------------------------
DIMENSION_EMBEDDING = 8  # valor pequeño a propósito, solo para verlo con claridad

capa_embedding = nn.Embedding(
    num_embeddings=tokenizador.tamano_vocabulario,
    embedding_dim=DIMENSION_EMBEDDING,
)

# -----------------------------------------------------------------
# 4. Pasamos los índices por la capa: cada índice se convierte en su vector
# -----------------------------------------------------------------
vectores = capa_embedding(tensor_indices)

print(f"\nShape de los vectores resultantes: {vectores.shape}")
print("(7 tokens x 8 números por token — cada fila es el 'significado' actual de un carácter)\n")

print("Vector correspondiente al primer carácter ('e'):")
print(vectores[0])

print("\nNota: estos vectores son aleatorios ahora mismo — nn.Embedding los")
print("inicializa así. Solo adquieren un significado útil DESPUÉS de entrenar")
print("el modelo completo, cuando el gradiente los ajuste para minimizar el")
print("error de predicción (exactamente igual que 'w' y 'b' en la regresión lineal).")