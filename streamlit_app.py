import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from knowledge_graph import KnowledgeGraph
from llm_integration import GraphRAGSystem, GraphQueryProcessor
from data_loader import StarWarsDataLoader, DataValidator
from config import Config
import os
import json

# Configuração da página
st.set_page_config(
    page_title="Knowledge Graph com LLM",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">Knowledge Graph Star Wars</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Sistema RAG com NetworkX + LangChain para consultas em linguagem natural</p>', unsafe_allow_html=True)

# Inicialização do estado da sessão
if 'kg' not in st.session_state:
    st.session_state.kg = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'query_processor' not in st.session_state:
    st.session_state.query_processor = None

# Sidebar
with st.sidebar:
    st.header("Configurações")
    
    # Verificar se a API key está configurada
    api_key = st.text_input(
        "Google API Key",
        value="",
        help="Insira sua chave da API Google Gemini"
    )
    
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
        Config.GOOGLE_API_KEY = api_key
    
    st.divider()
    
    # Carregar dados
    st.subheader("Dados")
    
    if st.button("Carregar Dados Star Wars", use_container_width=True):
        with st.spinner("Carregando dados..."):
            success = False
            try:
                loader = StarWarsDataLoader()
                st.session_state.kg = loader.create_sample_data()
                
                if api_key:
                    # Inicializar RAG system sem criar conexões assíncronas imediatamente
                    st.session_state.rag_system = GraphRAGSystem(st.session_state.kg)
                    st.session_state.query_processor = GraphQueryProcessor(st.session_state.kg)
                    st.success("Dados carregados com sucesso! LLM configurado.")
                else:
                    st.success("Dados carregados com sucesso! (Sem LLM - adicione API key para consultas em linguagem natural)")
                
                success = True
                    
            except Exception as e:
                st.error(f"Erro ao carregar dados: {str(e)}")
                st.error("Verifique se sua API key está correta e tente novamente.")
                
        if success:
            st.rerun()
    


# Conteúdo principal
if st.session_state.kg is None:
    st.info("Use a barra lateral para carregar os dados do Knowledge Graph")
    
    # Mostrar informações sobre o projeto
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="sub-header">Objetivo do Projeto</h2>', unsafe_allow_html=True)
        st.markdown("""
        Este projeto demonstra como criar um **Knowledge Graph** usando **NetworkX** 
        como alternativa ao Neo4j, integrado com **LangChain** para consultas em 
        linguagem natural.
        
        **Características:**
        - Modelagem de entidades e relacionamentos
        - Consultas via LLM com RAG
        - Visualizações interativas
        - Busca semântica
        - Análise de grafos
        """)
    
    with col2:
        st.markdown('<h2 class="sub-header">Tecnologias Utilizadas</h2>', unsafe_allow_html=True)
        st.markdown("""
        **Core:**
        - NetworkX (alternativa ao Neo4j)
        - LangChain + Google Gemini
        - Python + Streamlit
        
        **Funcionalidades:**
        - Vector Store (Busca Simples)
        - Embeddings semânticos
        - RAG (Retrieval Augmented Generation)
        - Visualização com Plotly
        """)
        
else:
    # Input global do chat (fora das tabs para evitar erro do Streamlit)
    chat_prompt = None
    if api_key:
        chat_prompt = st.chat_input("Faça uma pergunta sobre Star Wars...")
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Chat com IA",
        "Estatísticas",
        "Explorar Grafo",
        "Visualização",
        "Consultas Avançadas"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Consultas em Linguagem Natural</h2>', unsafe_allow_html=True)
        
        if not api_key:
            st.warning("Configure sua Google API Key na barra lateral para usar o chat")
        else:
            # Interface de chat
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Processar input do chat se houver
            if chat_prompt:
                prompt = chat_prompt
                # Adicionar mensagem do usuário
                st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Mostrar histórico de mensagens
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Processar nova mensagem se houver
            if chat_prompt:
                with st.chat_message("user"):
                    st.markdown(chat_prompt)
                
                # Gerar resposta
                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            result = st.session_state.rag_system.query(chat_prompt)
                            
                            # Adicionar resposta ao histórico
                            response_content = result["answer"]
                            st.session_state.messages.append({"role": "assistant", "content": response_content})
                            
                            # Mostrar resposta
                            st.markdown(response_content)
                            
                            # Mostrar confiança
                            confidence = result["confidence"]
                            if confidence > 0.7:
                                st.success(f"Confiança: {confidence:.1%}")
                            elif confidence > 0.4:
                                st.warning(f"Confiança: {confidence:.1%}")
                            else:
                                st.error(f"Confiança baixa: {confidence:.1%}")
                            
                            # Mostrar entidades mencionadas
                            if result["mentioned_entities"]:
                                with st.expander("Entidades Relacionadas"):
                                    for entity in result["mentioned_entities"]:
                                        st.write(f"- {entity}")
                            
                            # Sugerir perguntas relacionadas
                            try:
                                suggestions = st.session_state.rag_system.suggest_related_questions(chat_prompt)
                                if suggestions:
                                    with st.expander("Perguntas Sugeridas"):
                                        for suggestion in suggestions:
                                            if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                                                st.session_state.messages.append({"role": "user", "content": suggestion})
                                                st.rerun()
                            except Exception:
                                pass  # Sugestões são opcionais
                            
                        except Exception as e:
                            error_msg = str(e)
                            if "event loop" in error_msg.lower():
                                st.error("Erro de configuração do LLM. Tente recarregar a página e configurar novamente.")
                            elif "api" in error_msg.lower() or "key" in error_msg.lower():
                                st.error("Erro na API Key. Verifique se sua chave do Google Gemini está correta.")
                            else:
                                st.error(f"Erro: {error_msg}")
                            st.info("Dica: Tente usar as funcionalidades básicas do grafo na aba 'Explorar Grafo'")
            
            # Exemplos de perguntas
            with st.expander("Exemplos de Perguntas"):
                examples = [
                    "Quem é o pai de Luke Skywalker?",
                    "Quais planetas aparecem na saga?",
                    "Qual é a relação entre Obi-Wan e Darth Vader?",
                    "Quem são os membros da Ordem Jedi?",
                    "Que veículos Han Solo pilota?",
                    "Onde Yoda viveu em exílio?"
                ]
                
                for example in examples:
                    if st.button(example, key=f"example_{hash(example)}"):
                        st.session_state.messages.append({"role": "user", "content": example})
                        st.rerun()
    
    with tab2:
        st.markdown('<h2 class="sub-header">Estatísticas do Knowledge Graph</h2>', unsafe_allow_html=True)
        
        stats = st.session_state.kg.get_statistics()
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Entidades", stats['total_entities'])
        with col2:
            st.metric("Total de Relacionamentos", stats['total_relationships'])
        with col3:
            st.metric("Densidade do Grafo", f"{stats['density']:.3f}")
        with col4:
            conectado = "Sim" if stats['is_connected'] else "Não"
            st.metric("Conectado", conectado)
        
        # Distribuição por tipos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tipos de Entidades")
            entity_df = pd.DataFrame(
                list(stats['entity_types'].items()),
                columns=['Tipo', 'Quantidade']
            )
            st.dataframe(entity_df, use_container_width=True)
            
            # Gráfico de pizza
            fig_entities = go.Figure(data=[go.Pie(
                labels=entity_df['Tipo'],
                values=entity_df['Quantidade'],
                hole=0.3
            )])
            fig_entities.update_layout(title="Distribuição de Entidades")
            st.plotly_chart(fig_entities, use_container_width=True)
        
        with col2:
            st.subheader("Tipos de Relacionamentos")
            rel_df = pd.DataFrame(
                list(stats['relationship_types'].items()),
                columns=['Tipo', 'Quantidade']
            )
            st.dataframe(rel_df, use_container_width=True)
            
            # Gráfico de barras
            fig_rels = go.Figure(data=[go.Bar(
                x=rel_df['Tipo'],
                y=rel_df['Quantidade'],
                marker_color='lightblue'
            )])
            fig_rels.update_layout(
                title="Distribuição de Relacionamentos",
                xaxis_title="Tipo de Relacionamento",
                yaxis_title="Quantidade"
            )
            st.plotly_chart(fig_rels, use_container_width=True)
        
        # Validação dos dados
        st.subheader("Validação dos Dados")
        validation = DataValidator.validate_graph(st.session_state.kg)
        
        if validation['valid']:
            st.success("Grafo válido - sem problemas detectados")
        else:
            st.error("Problemas encontrados no grafo:")
            for issue in validation['issues']:
                st.write(f"- {issue}")
        
        if validation['warnings']:
            st.warning("Avisos:")
            for warning in validation['warnings']:
                st.write(f"- {warning}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Explorar Entidades e Relacionamentos</h2>', unsafe_allow_html=True)
        
        # Busca de entidades
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("Buscar entidades:", placeholder="Digite o nome de uma entidade...")
        
        with col2:
            entity_types = list(set(entity.type for entity in st.session_state.kg.entities.values()))
            selected_type = st.selectbox("Filtrar por tipo:", ["Todos"] + entity_types)
        
        if search_query:
            filter_type = None if selected_type == "Todos" else selected_type
            results = st.session_state.kg.search_entities(search_query, filter_type)
            
            if results:
                st.success(f"Encontradas {len(results)} entidades")
                
                for entity in results:
                    with st.expander(f"{entity.id} ({entity.type})"):
                        # Propriedades da entidade
                        st.write("**Propriedades:**")
                        for key, value in entity.properties.items():
                            st.write(f"- **{key}**: {value}")
                        
                        # Relacionamentos
                        neighbors = st.session_state.kg.get_neighbors(entity.id)
                        if neighbors:
                            st.write("**Conectado com:**")
                            for neighbor in neighbors:
                                neighbor_entity = st.session_state.kg.get_entity(neighbor)
                                if neighbor_entity:
                                    st.write(f"- {neighbor} ({neighbor_entity.type})")
            else:
                st.info("Nenhuma entidade encontrada")
        
        # Lista todas as entidades
        st.subheader("Todas as Entidades")
        
        entities_data = []
        for entity in st.session_state.kg.entities.values():
            entities_data.append({
                'ID': entity.id,
                'Tipo': entity.type,
                'Nome': entity.properties.get('name', entity.id),
                'Conexões': len(st.session_state.kg.get_neighbors(entity.id))
            })
        
        entities_df = pd.DataFrame(entities_data)
        st.dataframe(entities_df, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Visualização do Knowledge Graph</h2>', unsafe_allow_html=True)
        
        # Opções de visualização
        col1, col2, col3 = st.columns(3)
        
        with col1:
            layout_type = st.selectbox(
                "Layout do Grafo:",
                ["spring", "circular", "random"]
            )
        
        with col2:
            show_labels = st.checkbox("Mostrar rótulos", value=True)
        
        with col3:
            filter_by_type = st.multiselect(
                "Filtrar tipos de entidade:",
                entity_types,
                default=entity_types
            )
        
        # Gerar visualização
        if st.button("Gerar Visualização", use_container_width=True):
            with st.spinner("Gerando visualização..."):
                try:
                    # Filtrar entidades se necessário
                    if filter_by_type and len(filter_by_type) < len(entity_types):
                        filtered_entities = [
                            eid for eid, entity in st.session_state.kg.entities.items()
                            if entity.type in filter_by_type
                        ]
                        subgraph = st.session_state.kg.get_subgraph(filtered_entities)
                    else:
                        subgraph = st.session_state.kg.graph
                    
                    # Criar visualização interativa
                    fig = st.session_state.kg.visualize_interactive()
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erro na visualização: {e}")
        
        # Análise de caminhos
        st.subheader("Análise de Caminhos")
        
        col1, col2 = st.columns(2)
        
        entity_ids = list(st.session_state.kg.entities.keys())
        
        with col1:
            source_entity = st.selectbox("Entidade de origem:", entity_ids, key="source")
        
        with col2:
            target_entity = st.selectbox("Entidade de destino:", entity_ids, key="target")
        
        if st.button("Encontrar Caminho") and source_entity != target_entity:
            path_result = st.session_state.query_processor.find_shortest_path(source_entity, target_entity)
            
            if path_result["exists"]:
                st.success(f"Caminho encontrado! Distância: {path_result['length']} passos")
                
                st.write("**Caminho:**")
                for i, step in enumerate(path_result["path_details"]):
                    st.write(f"{i+1}. {step['from']} --[{step['relationship']}]--> {step['to']}")
            else:
                st.warning("Nenhum caminho encontrado entre essas entidades")
    
    with tab5:
        st.markdown('<h2 class="sub-header">Consultas Avançadas</h2>', unsafe_allow_html=True)
        
        # Consultas estruturadas
        st.subheader("Consultas Estruturadas")
        
        query_type = st.selectbox(
            "Tipo de consulta:",
            [
                "Buscar por tipo de entidade",
                "Buscar relacionamentos específicos",
                "Análise de centralidade",
                "Entidades mais conectadas"
            ]
        )
        
        if query_type == "Buscar por tipo de entidade":
            selected_entity_type = st.selectbox("Selecione o tipo:", entity_types)
            
            if st.button("Executar Busca"):
                results = [
                    entity for entity in st.session_state.kg.entities.values()
                    if entity.type == selected_entity_type
                ]
                
                st.write(f"**Encontradas {len(results)} entidades do tipo '{selected_entity_type}':**")
                for entity in results:
                    st.write(f"- {entity.id}: {entity.properties.get('name', 'N/A')}")
        
        elif query_type == "Buscar relacionamentos específicos":
            rel_types = list(set(rel.type for rel in st.session_state.kg.relationships))
            selected_rel_type = st.selectbox("Selecione o tipo de relacionamento:", rel_types)
            
            if st.button("Executar Busca"):
                results = [
                    rel for rel in st.session_state.kg.relationships
                    if rel.type == selected_rel_type
                ]
                
                st.write(f"**Encontrados {len(results)} relacionamentos do tipo '{selected_rel_type}':**")
                for rel in results:
                    st.write(f"- {rel.source} --[{rel.type}]--> {rel.target}")
        
        elif query_type == "Entidades mais conectadas":
            if st.button("Analisar Conectividade"):
                connectivity = {}
                for entity_id in st.session_state.kg.entities.keys():
                    connectivity[entity_id] = len(st.session_state.kg.get_neighbors(entity_id))
                
                # Ordenar por conectividade
                sorted_entities = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)
                
                st.write("**Top 10 entidades mais conectadas:**")
                for i, (entity_id, connections) in enumerate(sorted_entities[:10], 1):
                    entity = st.session_state.kg.get_entity(entity_id)
                    name = entity.properties.get('name', entity_id) if entity else entity_id
                    st.write(f"{i}. **{name}** ({entity_id}): {connections} conexões")
        


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Knowledge Graph Star Wars - Projeto de Algoritmos Avançados</p>",
    unsafe_allow_html=True
)