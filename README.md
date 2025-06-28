# Knowledge Graph com LLM - Alternativa ao Neo4j

**Desenvolvido por:**

- Arthur Henrique Tscha Vieira
- Rafael Rodrigues Ferreira de Andrade

> **Projeto de Algoritmos Avançados**: Sistema RAG com NetworkX + LangChain para consultas em linguagem natural

## Descrição do Projeto

Este projeto implementa um **Knowledge Graph** usando **NetworkX** como alternativa ao Neo4j, integrado com **LangChain** e **Google Gemini** para permitir consultas em linguagem natural. O sistema utiliza técnicas de **RAG (Retrieval Augmented Generation)** para fornecer respostas precisas baseadas no grafo de conhecimento.

### Domínio Escolhido: Universo Star Wars

O projeto utiliza o universo de **Star Wars** como domínio de conhecimento, incluindo:

- **Personagens** (Luke Skywalker, Darth Vader, Princess Leia, etc.)
- **Planetas** (Tatooine, Alderaan, Coruscant, Dagobah)
- **Organizações** (Jedi Order, Sith, Rebel Alliance, Galactic Empire)
- **Veículos** (Millennium Falcon, Death Star, X-wing)
- **Relacionamentos** (família, mentoria, afiliações, etc.)

## Tecnologias Utilizadas

### Core

- **NetworkX**: Alternativa ao Neo4j para modelagem de grafos
- **LangChain**: Framework para integração com LLMs
- **Google Gemini**: Modelo de linguagem para consultas naturais
- **Python**: Linguagem principal do projeto

### Funcionalidades Avançadas

- **Busca Semântica**: Implementação simples de similaridade de texto
- **Google Embeddings**: Embeddings via API do Gemini
- **Streamlit**: Interface web interativa
- **Plotly**: Visualizações interativas do grafo
- **Pandas**: Manipulação de dados

## Instalação e Configuração

### 1. Clone o Repositório

```bash
git clone https://github.com/arthurvieira2003/adv-alg-n3.git
cd adv-alg-n3
```

### 2. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 3. Configure a API Key

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env e adicione sua Google API Key
GOOGLE_API_KEY=sua_chave_aqui
```

### 4. Execute o Sistema

```bash
streamlit run streamlit_app.py
```

O sistema será iniciado em `http://localhost:8501` e estará pronto para uso!

## Estrutura do Projeto

```
adv-alg-n3/
├── README.md                 # Documentação principal
├── requirements.txt          # Dependências Python
├── config.py                # Configurações do sistema
├── knowledge_graph.py       # Implementação do Knowledge Graph
├── llm_integration.py       # Integração com LangChain/Google Gemini
├── data_loader.py           # Carregamento de dados Star Wars
├── streamlit_app.py         # Interface web interativa
├── data/                    # Dados do Knowledge Graph
│   ├── star_wars_knowledge_graph.json
│   ├── entities.csv
│   └── relationships.csv
└── .env.example             # Exemplo de configuração
```

## Como Usar

### Interface Web Streamlit

Após executar `streamlit run streamlit_app.py`, acesse a interface web onde você pode:

- **Chat com IA**: Faça perguntas em linguagem natural sobre Star Wars
- **Estatísticas**: Visualize métricas do Knowledge Graph
- **Explorar Grafo**: Navegue pelas entidades e relacionamentos
- **Visualização**: Veja o grafo de forma interativa
- **Consultas Avançadas**: Execute consultas específicas no grafo

### Funcionalidades da Interface Web

1. **Chat Inteligente**: Interface conversacional para consultas em linguagem natural
2. **Visualização Interativa**: Grafos dinâmicos com filtros e zoom
3. **Análise de Dados**: Estatísticas detalhadas do Knowledge Graph
4. **Exploração de Entidades**: Navegação por personagens, planetas, organizações e veículos
5. **Métricas de Confiança**: Indicadores de qualidade das respostas

### Exemplos de Consultas

**Consultas em Linguagem Natural (via Chat):**

- "Quem é Luke Skywalker?"
- "Qual é a relação entre Obi-Wan e Darth Vader?"
- "Quais planetas aparecem na saga?"
- "Quem são os membros da Ordem Jedi?"
- "Onde Yoda viveu em exílio?"
- "Qual é o caminho mais curto entre Luke e Yoda?"
- "Quantos personagens estão conectados ao planeta Tatooine?"

**Funcionalidades Disponíveis:**

- Busca semântica por entidades
- Análise de relacionamentos
- Cálculo de caminhos no grafo
- Estatísticas em tempo real
- Visualização interativa

## Arquitetura do Sistema

### 1. Knowledge Graph (NetworkX)

```python
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Grafo direcionado
        self.entities = {}              # Dicionário de entidades
        self.relationships = []         # Lista de relacionamentos
```

### 2. Sistema RAG

```python
class GraphRAGSystem:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.llm = ChatGoogleGenerativeAI()  # Modelo de linguagem
        self.embeddings = GoogleGenerativeAIEmbeddings()  # Embeddings
        self.vector_store = SimpleVectorStore()  # Vector store simples
```

### 3. Fluxo de Consulta

1. **Entrada**: Pergunta em linguagem natural
2. **Embedding**: Conversão para vetor semântico
3. **Retrieval**: Busca no vector store
4. **Augmentation**: Contexto do grafo
5. **Generation**: Resposta do LLM
6. **Validação**: Verificação com o grafo

## Métricas e Validação

### Estatísticas do Grafo

- **Entidades**: 18 (6 personagens, 4 planetas, 4 organizações, 4 veículos)
- **Relacionamentos**: 25 (família, mentoria, afiliações, etc.)
- **Densidade**: ~0.082 (grafo esparso, realista)
- **Conectividade**: Grafo fracamente conectado

### Validação de Dados

- Verificação de relacionamentos órfãos
- Identificação de entidades isoladas
- Consistência de tipos e propriedades

## Referências

- [NetworkX Documentation](https://networkx.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API Reference](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Papers and Tutorials](https://arxiv.org/abs/2005.11401)
