# Autoencoder Variacional (VAE) no CIFAR-10 com Keras

Este projeto treina um Autoencoder Variacional (VAE) nas imagens do CIFAR-10 para aprender a gerar imagens novas. Diferente de um autoencoder comum, que só comprime e reconstrói, o VAE aprende uma distribuição de probabilidade no espaço latente, e por isso é possível sortear um ponto qualquer dessa distribuição, passar pelo decoder e obter uma imagem que não existia na base. Um exemplo prático seria sortear 256 vetores de ruído e receber 256 imagens 32x32 de objetos do dia a dia, como aviões, carros e animais.

## Como funciona

O encoder é uma rede convolucional (convoluções de passo 2 com 128, 64 e 32 filtros, uma camada densa de 256 neurônios) que, em vez de devolver um vetor único, devolve dois: a média (zMean) e o log da variância (zLogVar) de uma distribuição normal no espaço latente de 16 dimensões. O ponto que vai para o decoder é sorteado dessa distribuição pelo truque da reparametrização, z = média + desvio padrão * epsilon, com epsilon vindo de uma normal padrão, o que permite o gradiente passar pelo sorteio e treinar a rede de ponta a ponta.

O decoder faz o caminho inverso, com uma camada densa que vira um mapa 4x4 com 32 canais e convoluções transpostas que dobram a resolução até 32x32, com sigmoid na saída para ficar na faixa de 0 a 1. A loss total soma duas partes: a de reconstrução (binary crossentropy somada sobre os pixels), que cobra que a imagem de saída seja parecida com a de entrada, e a divergência KL, que cobra que a distribuição aprendida fique perto de uma normal padrão. O peso da KL foi de 4, valor que dá mais peso à regularização do espaço latente e deixa a geração por sorteio mais consistente, em troca de reconstruções menos nítidas.

## Dados

A base é o CIFAR-10, carregada direto pelo Keras, com imagens coloridas de 32x32 pixels em 10 classes. Como o VAE não usa rótulos para treinar, as 50 mil imagens de treino e as 10 mil de teste foram juntadas, 60 mil no total, e normalizadas para a faixa de 0 a 1. Os rótulos só são usados no final, para colorir o gráfico do espaço latente.

## Resultados

O modelo foi treinado por 128 épocas com Adam (taxa de 0,001) e batch de 128. Na última época as losses ficaram assim:

| Loss | Valor |
| --- | --- |
| Reconstrução | 631,56 |
| KL | 5,64 |
| Total (reconstrução + 4 x KL) | 654,14 |

O notebook mostra o gráfico das três losses ao longo do treino, uma grade de 256 imagens geradas a partir de ruído e o espaço latente (as duas primeiras dimensões da média) colorido pela classe do CIFAR-10. A KL baixa, perto de 5,6, mostra que o espaço latente carrega pouca informação por imagem, o que provavelmente se deve ao peso 4 na KL, e junto da resolução pequena do CIFAR-10 isso faz as imagens geradas saírem borradas, característica conhecida dos VAEs.

## Como rodar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois abra o notebook e execute as células em ordem. O CIFAR-10 é baixado pelo Keras na primeira execução.

```bash
jupyter notebook notebook/vae_cifar10.ipynb
```

## Estrutura do projeto

```
vae-cifar10-keras/
├── notebook/
│   └── vae_cifar10.ipynb   # encoder, decoder, treino e visualizacoes
├── requirements.txt
└── .gitignore
```

## Observações e próximos passos

Como próximos passos, vale testar pesos menores para a KL (como 1) e ver o equilíbrio entre nitidez e qualidade da geração, aumentar o espaço latente, usar uma arquitetura com mais camadas e comparar com a GAN do mesmo tipo de imagem, que costuma gerar imagens mais nítidas mas é mais difícil de treinar.

Os resultados deste projeto podem ser melhores. O foco foi o aprendizado do conteúdo do bootcamp e não obter o melhor desempenho possível, por isso não foi feita uma busca exaustiva por hiperparâmetros, arquiteturas ou dados.
