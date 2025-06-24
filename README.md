# 🌌 Knowledge Graph com LLM - Alternativa ao Neo4j

> **Projeto de Algoritmos Avançados**: Sistema RAG com NetworkX + LangChain para consultas em linguagem natural

## 📋 Descrição do Projeto

Este projeto implementa um **Knowledge Graph** usando **NetworkX** como alternativa ao Neo4j, integrado com **LangChain** e **Google Gemini** para permitir consultas em linguagem natural. O sistema utiliza técnicas de **RAG (Retrieval Augmented Generation)** para fornecer respostas precisas baseadas no grafo de conhecimento.

### 🎯 Objetivos

- ✅ Modelar um domínio de conhecimento como grafo (entidades, relacionamentos e propriedades)
- ✅ Importar dados estruturados para o sistema
- ✅ Permitir consultas com linguagem natural via LLM + RAG
- ✅ Demonstrar como a base de conhecimento evita alucinações da IA
- ✅ Fornecer alternativa viável ao Neo4j usando NetworkX

### 🌟 Domínio Escolhido: Universo Star Wars

O projeto utiliza o universo de **Star Wars** como domínio de conhecimento, incluindo:
- 👥 **Personagens** (Luke Skywalker, Darth Vader, Princess Leia, etc.)
- 🪐 **Planetas** (Tatooine, Alderaan, Coruscant, Dagobah)
- 🏛️ **Organizações** (Jedi Order, Sith, Rebel Alliance, Galactic Empire)
- 🚀 **Veículos** (Millennium Falcon, Death Star, X-wing)
- 🔗 **Relacionamentos** (família, mentoria, afiliações, etc.)

## 🛠️ Tecnologias Utilizadas

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

## 🚀 Instalação e Configuração

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/adv-alg-n3.git
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

> 📖 **Guia Detalhado**: Consulte [GOOGLE_GEMINI_SETUP.md](GOOGLE_GEMINI_SETUP.md) para instruções completas de configuração da API Google Gemini.

### 4. Execute o Sistema

#### Modo Console (Demonstração)
```bash
python main.py
```

#### Interface Web (Streamlit)
```bash
streamlit run streamlit_app.py
```

## 📊 Estrutura do Projeto

```
adv-alg-n3/
├── 📄 README.md                 # Documentação principal
├── 📋 requirements.txt          # Dependências Python
├── ⚙️ config.py                # Configurações do sistema
├── 🔗 knowledge_graph.py       # Implementação do Knowledge Graph
├── 🤖 llm_integration.py       # Integração com LangChain/OpenAI
├── 📊 data_loader.py           # Carregamento de dados Star Wars
├── 🖥️ main.py                  # Script principal (modo console)
├── 🌐 streamlit_app.py         # Interface web interativa
├── 📁 data/                    # Dados do Knowledge Graph
│   ├── star_wars_knowledge_graph.json
│   ├── entities.csv
│   └── relationships.csv
└── 🔐 .env.example             # Exemplo de configuração
```

## 🎮 Como Usar

### 1. Modo Console Interativo

Após executar `python main.py`, você pode:

```
🔍 Sua pergunta: Quem é o pai de Luke Skywalker?
🤖 Darth Vader é o pai de Luke Skywalker...

🔍 Sua pergunta: stats
📊 Estatísticas:
   - Entidades: 18
   - Relacionamentos: 25
   - Densidade: 0.082
```

### 2. Interface Web (Streamlit)

A interface web oferece:
- 💬 **Chat com IA**: Consultas em linguagem natural
- 📊 **Estatísticas**: Métricas do grafo
- 🔍 **Explorar Grafo**: Busca e navegação
- 📈 **Visualização**: Grafos interativos
- ⚙️ **Consultas Avançadas**: Análises específicas

### 3. Exemplos de Consultas

```python
# Consultas em linguagem natural
"Quem é Luke Skywalker?"
"Qual é a relação entre Obi-Wan e Darth Vader?"
"Quais planetas aparecem na saga?"
"Quem são os membros da Ordem Jedi?"
"Onde Yoda viveu em exílio?"

# Consultas estruturadas
kg.search_entities("Skywalker")
kg.get_neighbors("luke_skywalker")
kg.find_path("luke_skywalker", "yoda")
```

## 🏗️ Arquitetura do Sistema

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

## 📈 Funcionalidades Principais

### 🔍 Consultas Semânticas
- Busca por similaridade usando embeddings
- Recuperação de contexto relevante do grafo
- Respostas fundamentadas em dados estruturados

### 📊 Análise de Grafos
- Cálculo de métricas (densidade, conectividade)
- Análise de caminhos mais curtos
- Identificação de entidades centrais

### 🎨 Visualizações
- Grafos interativos com Plotly
- Filtros por tipo de entidade
- Análise visual de relacionamentos

### 🔒 Prevenção de Alucinações
- Respostas baseadas apenas em dados do grafo
- Indicadores de confiança
- Rastreabilidade das fontes

## 🧪 Exemplos de Uso

### Consulta Básica
```python
from knowledge_graph import KnowledgeGraph
from data_loader import StarWarsDataLoader

# Carregar dados
loader = StarWarsDataLoader()
kg = loader.create_sample_data()

# Buscar entidade
luke = kg.get_entity("luke_skywalker")
print(f"Nome: {luke.properties['name']}")
print(f"Planeta: {luke.properties['homeworld']}")

# Encontrar relacionamentos
neighbors = kg.get_neighbors("luke_skywalker")
print(f"Conectado com: {neighbors}")
```

### Consulta com LLM
```python
from llm_integration import GraphRAGSystem

# Configurar sistema RAG
rag_system = GraphRAGSystem(kg)
rag_system.build_vector_store()
rag_system.setup_qa_chain()

# Fazer pergunta
result = rag_system.query("Quem treinou Luke Skywalker?")
print(f"Resposta: {result['answer']}")
print(f"Confiança: {result['confidence']:.1%}")
```

## 📊 Métricas e Validação

### Estatísticas do Grafo
- **Entidades**: 18 (6 personagens, 4 planetas, 4 organizações, 4 veículos)
- **Relacionamentos**: 25 (família, mentoria, afiliações, etc.)
- **Densidade**: ~0.082 (grafo esparso, realista)
- **Conectividade**: Grafo fracamente conectado

### Validação de Dados
- Verificação de relacionamentos órfãos
- Identificação de entidades isoladas
- Consistência de tipos e propriedades

## 🔧 Personalização

### Adicionar Novo Domínio
1. Crie um novo data loader seguindo o padrão de `StarWarsDataLoader`
2. Defina suas entidades, tipos e relacionamentos
3. Implemente os métodos de carregamento
4. Configure o sistema RAG

### Modificar Configurações
Edite `config.py` para ajustar:
- Modelos de LLM e embeddings
- Parâmetros de chunking
- Limites de resultados
- Configurações do grafo

## 🚨 Limitações e Considerações

### Limitações do NetworkX vs Neo4j
- **Performance**: NetworkX é menos otimizado para grafos muito grandes
- **Persistência**: Requer serialização manual (JSON/pickle)
- **Consultas**: Não possui linguagem de consulta nativa como Cypher
- **Escalabilidade**: Limitado pela memória RAM

### Vantagens da Abordagem
- **Simplicidade**: Não requer instalação de banco de dados
- **Flexibilidade**: Fácil integração com Python
- **Portabilidade**: Funciona em qualquer ambiente Python
- **Desenvolvimento**: Ideal para prototipagem e projetos acadêmicos

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Equipe

- **Desenvolvedor Principal**: [Seu Nome]
- **Disciplina**: Algoritmos Avançados
- **Instituição**: [Sua Instituição]

## 📚 Referências

- [NetworkX Documentation](https://networkx.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API Reference](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Papers and Tutorials](https://arxiv.org/abs/2005.11401)

---

**🌟 Projeto desenvolvido como alternativa viável ao Neo4j para Knowledge Graphs acadêmicos e de pequeno/médio porte.**