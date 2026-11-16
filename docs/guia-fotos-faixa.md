# Guia de fotos: faixa de pedestre (nítida × apagada)

O modelo **não** olha o frame inteiro. Ele recebe só o **recorte da faixa**, uma região fixa por câmera
definida no `config/cameras.json`. Por isso as fotos de treino precisam parecer esse recorte:
a faixa ocupando quase toda a imagem, vista de cima e de frente, como uma câmera de poste veria.

## Quantas fotos

| Classe | Mínimo | Ideal |
|---|---|---|
| `nitida` (listras brancas inteiras e contrastadas) | 60 recortes de **pelo menos 12 faixas diferentes** | 100+ de 20 faixas |
| `apagada` (listras gastas, falhadas ou quase sumidas) | 60 recortes de **pelo menos 12 faixas diferentes** | 100+ de 20 faixas |

O que importa mais é o **número de faixas diferentes**, não o de fotos. Dez fotos da mesma faixa
ensinam pouco: o modelo decora aquela faixa em vez de aprender o que é "apagada".

**Na dúvida, não use.** Faixa "meio apagada" confunde o modelo. Fotografe as claramente nítidas e as
claramente apagadas, e deixe as intermediárias de fora (ou numa pasta `duvida/`, que não entra no treino).

## Como fotografar

- **Posição:** do alto, se possível (passarela, janela, calçada elevada) ou em pé na calçada, com o
  celular na altura dos olhos e inclinado uns 45° para baixo. **Nunca** no meio da rua.
- **Enquadramento:** a faixa inteira, ou pelo menos 4 listras, ocupando a maior parte da foto.
- **Variação** (por faixa, 3 a 6 fotos): 2 ou 3 ângulos (de frente e um pouco de lado), 2 distâncias
  e, se der, 2 horários (sol forte e nublado/sombra). Isso ensina que a iluminação não muda a classe.
- **Evite:** pessoas e placas em destaque (se aparecerem, tudo bem: o recorte e o borrão cuidam disso,
  mas prefira fotos sem gente), carro em cima da faixa, chuva forte, foto tremida.
- **Segurança:** só fotografe com o sinal fechado para os carros, da calçada.

## Como recortar

Recorte um retângulo **só com a faixa** (pode sobrar um pouco de asfalto em volta). Não precisa ter
tamanho exato: o treino redimensiona para 224 × 224. Se a foto já estiver bem enquadrada, não recorte.

## Pastas

Uma subpasta **por faixa**, com um nome que identifique o lugar. É essa pasta que garante a separação
sem vazamento.

```
data/raw/crosswalk/
  nitida/
    av-paulista-1578/        ← uma faixa = uma pasta
      IMG_0001.jpg
      IMG_0002.jpg
    rua-augusta-900/
      ...
  apagada/
    rua-vergueiro-2100/
      ...
```

`data/raw/` está no `.gitignore`: as fotos originais não vão para o GitHub (podem ter gente e placas).

## Treino, validação e teste POR FAIXA (sem data leakage)

**Data leakage** é o modelo "ver a prova antes": se uma foto da faixa da Paulista vai para o treino e
outra foto **da mesma faixa** vai para o teste, o modelo acerta porque reconhece o lugar (o meio-fio,
a sombra da árvore), e não porque aprendeu o que é faixa apagada. A métrica fica bonita e mentirosa.

Por isso o script de treino sorteia **pastas (faixas)**, e não fotos:

| Conjunto | Parte das faixas | Para quê |
|---|---|---|
| Treino | ~70% | o modelo aprende |
| Validação | ~15% | escolher a época e o limiar, sem tocar no teste |
| Teste | ~15% | a nota final (precisão, recall, F1, matriz de confusão), usada **uma vez só** |

O sorteio usa uma semente fixa (`seed=42`), então qualquer pessoa que rodar o script obtém a mesma
divisão. A lista de faixas de cada conjunto é salva em `docs/avaliacao/split-faixa.json`.

## Checklist antes de me mandar

- [ ] Pelo menos 12 faixas em `nitida/` e 12 em `apagada/`
- [ ] Cada faixa na sua própria pasta
- [ ] Nenhuma foto "meio apagada" misturada
- [ ] Fotos recortadas ou já bem enquadradas na faixa
