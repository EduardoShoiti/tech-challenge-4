# Tech Challenge 4

## 1. Objetivo do projeto

A proposta é permitir que um usuário faça perguntas em linguagem natural sobre as avaliações de clientes da Olist, como:

- Quais são os principais problemas relacionados à entrega?
- Quais reclamações aparecem em avaliações de nota baixa?
- O que os clientes mais elogiam?
- Quais problemas aparecem com maior frequência em determinada categoria?

O sistema deverá recuperar as avaliações mais relevantes e, posteriormente, utilizar um LLM para gerar respostas fundamentadas nas evidências encontradas.

## 2. Arquitetura desenvolvida

A solução foi organizada em dois fluxos principais:

1. **Pipeline de indexação (offline):** prepara a base e constrói o índice vetorial.
2. **Pipeline de consulta (online):** recebe uma pergunta e recupera os documentos semanticamente mais próximos.

```mermaid
flowchart LR
    A[Dataset Olist] --> B[Preparação dos dados]
    B --> C[Document Builder]
    C --> D[Documentos: id + texto + metadata]
    D --> E[Modelo de Embeddings]
    E --> F[ChromaDB / Vector Store]

    Q[Pergunta do usuário] --> G[Embedding da pergunta]
    G --> F
    F --> H[Top-K reviews relevantes]
    H --> V[Validação de evidência]

    V -. próxima etapa .-> I[Prompt + LLM]
    I -. próxima etapa .-> J[Resposta + evidências]
```

### Estrutura lógica

```text
DataFrame processado
        ↓
document_builder.py
        ↓
Documentos
{id, text, metadata}
        ↓
embeddings.py
        ↓
Vetores semânticos
        ↓
build_vector_store.py
        ↓
ChromaDB
        ↓
retriever.py
        ↓
Top-K reviews relevantes
        ↓
Validação de evidência
        ↓
Prompt + LLM
        ↓
Resposta + Evidências
```

## 3. Tecnologias e bibliotecas utilizadas

### Linguagem

- **Python**

### Manipulação e preparação dos dados

- **pandas** — leitura, limpeza e transformação da base.
- **numpy** — suporte a tipos numéricos e conversões auxiliares.
- **pyarrow** — leitura/escrita de arquivos Parquet.

### Embeddings

- **sentence-transformers**
- Modelo utilizado inicialmente: **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**

O modelo foi escolhido como baseline por ser:

- multilíngue, incluindo português;
- leve e rápido para execução local;
- adequado para similaridade semântica entre sentenças e textos curtos;
- simples de integrar ao pipeline;
- compatível com execução em CPU.

O modelo gera embeddings de 384 dimensões.

### Vector Store

- **ChromaDB**

Foi escolhido por ser simples de executar localmente, persistente em disco e permitir armazenar, para cada documento:

- ID;
- texto original;
- metadados;
- embedding.

Além disso, permite recuperação semântica e filtros por metadata.


## 4. Preparação da base

### 4.1. Tratamento dos campos de texto

A base de reviews contém dois campos principais de conteúdo textual:

- `review_comment_title`
- `review_comment_message`

A estratégia adotada foi preservar informação existente em qualquer um deles.

Quando os dois campos possuem conteúdo, são concatenados para formar `review_text`.

Exemplo:

```text
Título: "Entrega ruim"
Mensagem: "Produto chegou duas semanas atrasado."

↓

review_text:
"Entrega ruim. Produto chegou duas semanas atrasado."
```

Quando apenas um dos campos existe, seu conteúdo é preservado.

Avaliações sem título **e** sem mensagem são removidas da base utilizada pelo RAG, pois não possuem conteúdo textual para indexação.

### 4.2. Reviews curtas

Reviews curtas com duas ou mais palavras podem ser semanticamente úteis e são mantidas, por exemplo:

```text
"não chegou"
"produto quebrado"
"entrega atrasada"
```

São removidas as reviews:

- sem conteúdo textual;
- formadas por apenas uma palavra;
- que correspondem à lista de expressões genéricas definida no pipeline.

Exemplos de expressões genéricas:

```text
"bom"
"ótimo"
"muito bom"
```

Também são removidos textos duplicados, mantendo apenas a primeira ocorrência.

### 4.3. Limpeza textual

A limpeza é conservadora, pois embeddings modernos funcionam melhor com o texto natural preservado.

São tratados:

- valores nulos;
- strings vazias;
- espaços no início e no fim dos campos;
- pontuação usada na união entre título e mensagem.

Uma versão normalizada do texto é criada apenas para aplicar o filtro de expressões genéricas. O `review_text` original é preservado para a geração dos embeddings.

Não são aplicadas transformações agressivas como:

- remoção de stopwords;
- stemming;
- lematização obrigatória;
- remoção de negações;
- remoção indiscriminada de pontuação.

Isso evita alterar o significado original das avaliações.

### 4.4. Dataset processado e Vector Store são artefatos diferentes

O DataFrame processado continua existindo para análise, auditoria e reprodutibilidade.

Exemplo de estrutura:

```text
review_id
review_text
review_score
```

Os embeddings não precisam ser mantidos permanentemente no DataFrame.

A Vector Store é uma segunda representação dos mesmos registros, criada especificamente para retrieval semântico.

```text
Dataset processado
        ↓
Documentos
        ↓
Embeddings
        ↓
Vector Store
```

## 5. Estratégia utilizada para construção do pipeline RAG

A estratégia atual separa claramente as responsabilidades de cada componente.

### 5.1. Preparação dos dados

Nesta etapa, o projeto utiliza o arquivo de avaliações da Olist. O processamento transforma os títulos e comentários em um DataFrame específico para o RAG, preservando também a nota da avaliação.

### 5.2. Construção dos documentos

Foi adotada a regra:

> **1 review = 1 documento**

Como as avaliações são curtas, não foi necessário dividir cada review em chunks menores.

Cada documento segue a estrutura conceitual:

```python
{
    "id": "R001",
    "text": "Produto chegou atrasado e com a embalagem danificada.",
    "metadata": {
        "review_score": 2,
    }
}
```

O texto é a informação utilizada para gerar o embedding. Os metadados são usados para rastreabilidade, filtros e enriquecimento da consulta.

### 5.3. Geração dos embeddings

Os textos dos documentos são transformados em vetores usando o mesmo modelo de embeddings.

```text
texto da review
      ↓
modelo de embedding
      ↓
vetor numérico
```

Exemplo conceitual:

```text
"Produto demorou muito para chegar"
              ↓
[0.031, -0.218, 0.441, ...]
```

O mesmo modelo deve ser usado depois para transformar a pergunta do usuário. Isso garante que documentos e consultas estejam no mesmo espaço vetorial.

### 5.4. Indexação na Vector Store

Os documentos e seus respectivos embeddings são armazenados no ChromaDB.

Para cada registro, a Vector Store mantém:

```text
ID
+
texto
+
metadata
+
embedding
```

### 5.5. Retrieval

A pergunta do usuário é transformada em embedding e comparada aos vetores armazenados no ChromaDB.

O sistema retorna os **Top-K documentos semanticamente mais próximos**.

Exemplo:

```text
Pergunta:
"problemas relacionados ao atraso na entrega"

↓

Possíveis resultados:
- "Meu pedido levou quase um mês para chegar."
- "Entrega realizada depois do prazo informado."
- "Ainda não recebi meu produto."
```

A recuperação é semântica, portanto não depende apenas da presença exata das mesmas palavras.

Além dos resultados, o retriever calcula:

- a menor distância encontrada;
- a distância média dos três primeiros resultados;
- se há evidência suficiente de acordo com os thresholds configurados.

Atualmente são recuperadas cinco reviews por consulta. Os limites iniciais são `0.65` para a melhor distância e `0.66` para a média das três primeiras distâncias.


## 6. Principais decisões técnicas

### 6.1. Uma review por documento

As avaliações da Olist são textos curtos e representam naturalmente uma unidade independente da experiência do cliente.

Por isso, não foi aplicado chunking tradicional.

**Vantagens:**

- preserva o significado completo da avaliação;
- simplifica o pipeline;
- evita fragmentação desnecessária;
- melhora a rastreabilidade da evidência.

### 6.2. Título + mensagem

Reviews com mensagem nula não são automaticamente descartadas, pois o título isolado pode conter informação relevante.

Exemplo:

```text
"Produto não chegou"
```

Esse conteúdo é útil para recuperação mesmo sem comentário adicional.

### 6.3. Embeddings separados da estrutura do documento

O documento representa a informação original:

```text
id + texto + metadata
```

O embedding representa matematicamente esse documento.

Essa separação reduz acoplamento e facilita trocar o modelo de embeddings no futuro.

### 6.4. Modelo multilíngue

O dataset contém avaliações em português, portanto foi priorizado um modelo multilíngue.

O `paraphrase-multilingual-MiniLM-L12-v2` foi adotado como baseline devido ao bom equilíbrio entre:

- qualidade;
- velocidade;
- consumo de memória;
- facilidade de execução local.

### 6.5. ChromaDB local

Nesta etapa, não há necessidade de infraestrutura cloud ou banco vetorial gerenciado.

O ChromaDB é persistido localmente em disco, reduzindo complexidade e custo.

### 6.6. Separação entre indexação e consulta

A geração dos embeddings de todos os documentos ocorre apenas na construção/reconstrução do índice.

Não é necessário recalcular todos os embeddings a cada pergunta.

```text
Indexação:
reviews → embeddings → ChromaDB

Consulta:
pergunta → embedding → ChromaDB → Top-K
```

---

### Responsabilidades dos arquivos

| Arquivo | Responsabilidade |
|---|---|
| `data_preparation.py` | Prepara o CSV de avaliações e gera o dataset processado |
| `config.py` | Centraliza modelo, caminhos e parâmetros do retrieval |
| `document_builder.py` | Converte cada linha do DataFrame em documento RAG |
| `embeddings.py` | Carrega o modelo e gera embeddings |
| `build_vector_store.py` | Cria/persiste a collection no ChromaDB |
| `build_index.py` | Orquestra todo o pipeline de indexação |
| `retriever.py` | Recupera os Top-K documentos e valida a evidência |
| `evaluation.py` | Executa perguntas de avaliação e resume as distâncias |
| `test_evaluation.py` | Carrega os componentes e executa a avaliação do retrieval |

---

## 7. Avaliação do retrieval

O projeto possui um conjunto inicial com seis perguntas em `evaluation/retrieval_questions.json`:

- três perguntas dentro do escopo das avaliações;
- três perguntas fora do escopo.

A avaliação compara a melhor distância e a média das três primeiras distâncias entre os dois grupos. Os resultados iniciais indicam separação parcial, mas o conjunto ainda é pequeno e os thresholds devem ser tratados como valores iniciais para teste.

Perguntas amplas, como aquelas sobre os principais problemas relatados, podem apresentar distância maior que perguntas específicas sobre produto quebrado ou produto errado.

---

## 8. Como reproduzir a solução

### 8.1. Pré-requisitos

- Python
- Git
- Acesso à internet na primeira execução para instalar dependências e baixar o modelo de embeddings

### 8.2. Criar ambiente virtual

No Windows / PowerShell:

```powershell
python -m venv .venv
```

Ativar:

```powershell
.\.venv\Scripts\Activate.ps1
```

Atualizar o `pip`:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 8.3. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 8.4. Preparar os dados

Executar o pipeline de preparação:

```powershell
python src/data_preparation.py
```

O comando gera:

```text
data/processed/df_reviews_rag.parquet
```

O arquivo deve conter, no mínimo:

```text
review_id
review_text
```

O metadado utilizado atualmente é:

```text
review_score
```

### 8.5. Construir o índice vetorial

Quando o dataset, o texto dos documentos ou o modelo de embeddings mudar, executar:

```powershell
python src/build_index.py
```

O índice é persistido na pasta `chroma_db/`. Não é necessário reconstruí-lo para cada consulta.

### 8.6. Executar a avaliação

Com o índice já construído, executar:

```powershell
python src/test_evaluation.py
```

O script utiliza as perguntas em `evaluation/retrieval_questions.json` e apresenta as distâncias de cada consulta e o resumo por grupo.


## 9. Dataset

**Brazilian E-Commerce Public Dataset by Olist**  
Disponível no Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data
