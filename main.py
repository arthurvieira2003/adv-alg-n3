#!/usr/bin/env python3
"""
Script principal para demonstrar o Knowledge Graph com LLM
Alternativa ao Neo4j usando NetworkX + LangChain
"""

import os
import sys
from typing import Optional
from knowledge_graph import KnowledgeGraph
from llm_integration import GraphRAGSystem, GraphQueryProcessor
from data_loader import StarWarsDataLoader, DataValidator
from config import Config

def setup_environment():
    """Configura o ambiente e verifica dependências"""
    print("🚀 Iniciando Knowledge Graph System...")
    
    # Verificar se a API key está configurada
    if not Config.GOOGLE_API_KEY:
        print("⚠️  Google API Key não encontrada!")
        api_key = input("Digite sua Google API Key (ou pressione Enter para pular): ")
        if api_key.strip():
            os.environ['GOOGLE_API_KEY'] = api_key
            Config.GOOGLE_API_KEY = api_key
            print("✅ API Key configurada!")
        else:
            print("⚠️  Continuando sem LLM (apenas funcionalidades do grafo)")
    
    return Config.GOOGLE_API_KEY is not None

def create_sample_knowledge_graph() -> KnowledgeGraph:
    """Cria e retorna um Knowledge Graph de exemplo"""
    print("\n📊 Carregando dados do universo Star Wars...")
    
    loader = StarWarsDataLoader()
    kg = loader.create_sample_data()
    
    # Salvar dados para uso futuro
    loader.save_sample_data()
    
    print(f"✅ Knowledge Graph criado com sucesso!")
    
    # Mostrar estatísticas
    stats = kg.get_statistics()
    print(f"   📈 {stats['total_entities']} entidades")
    print(f"   🔗 {stats['total_relationships']} relacionamentos")
    print(f"   🌐 Densidade: {stats['density']:.3f}")
    
    return kg

def demonstrate_graph_queries(kg: KnowledgeGraph):
    """Demonstra consultas básicas no grafo"""
    print("\n🔍 Demonstrando consultas no grafo...")
    
    query_processor = GraphQueryProcessor(kg)
    
    # 1. Buscar entidades por tipo
    print("\n1️⃣ Personagens principais:")
    characters = [e for e in kg.entities.values() if e.type == "Character"]
    for char in characters[:5]:
        print(f"   - {char.properties.get('name', char.id)}")
    
    # 2. Encontrar relacionamentos
    print("\n2️⃣ Relacionamentos familiares:")
    family_rels = [r for r in kg.relationships if "FATHER_OF" in r.type or "SIBLING_OF" in r.type]
    for rel in family_rels:
        source_name = kg.get_entity(rel.source).properties.get('name', rel.source)
        target_name = kg.get_entity(rel.target).properties.get('name', rel.target)
        print(f"   - {source_name} --[{rel.type}]--> {target_name}")
    
    # 3. Análise de caminhos
    print("\n3️⃣ Caminho entre Luke Skywalker e Darth Vader:")
    path_result = query_processor.find_shortest_path("luke_skywalker", "darth_vader")
    if path_result["exists"]:
        print(f"   📏 Distância: {path_result['length']} passos")
        for step in path_result["path_details"]:
            print(f"   🛤️  {step['from']} --[{step['relationship']}]--> {step['to']}")
    
    # 4. Entidades mais conectadas
    print("\n4️⃣ Entidades mais conectadas:")
    connectivity = {}
    for entity_id in kg.entities.keys():
        connectivity[entity_id] = len(kg.get_neighbors(entity_id))
    
    top_connected = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)[:3]
    for entity_id, connections in top_connected:
        entity = kg.get_entity(entity_id)
        name = entity.properties.get('name', entity_id)
        print(f"   🌟 {name}: {connections} conexões")

def demonstrate_llm_integration(kg: KnowledgeGraph):
    """Demonstra integração com LLM"""
    if not Config.GOOGLE_API_KEY:
        print("\n⚠️  Pulando demonstração do LLM (API Key não configurada)")
        return
    
    print("\n🤖 Demonstrando integração com LLM...")
    
    try:
        # Inicializar sistema RAG
        rag_system = GraphRAGSystem(kg)
        print("   ⚙️  Construindo vector store...")
        rag_system.build_vector_store()
        rag_system.setup_qa_chain()
        print("   ✅ Sistema RAG configurado!")
        
        # Perguntas de exemplo
        sample_questions = [
            "Quem é o pai de Luke Skywalker?",
            "Quais planetas aparecem na saga?",
            "Qual é a relação entre Obi-Wan e Darth Vader?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n{i}️⃣ Pergunta: {question}")
            
            try:
                result = rag_system.query(question)
                print(f"   🤖 Resposta: {result['answer']}")
                print(f"   🎯 Confiança: {result['confidence']:.1%}")
                
                if result['mentioned_entities']:
                    print(f"   🔗 Entidades: {', '.join(result['mentioned_entities'])}")
                    
            except Exception as e:
                print(f"   ❌ Erro na consulta: {e}")
                
    except Exception as e:
        print(f"❌ Erro na configuração do LLM: {e}")
        print("   💡 Verifique sua API Key e conexão com a internet")

def interactive_mode(kg: KnowledgeGraph, rag_system: Optional[GraphRAGSystem] = None):
    """Modo interativo para consultas"""
    print("\n💬 Modo Interativo Ativado!")
    print("Digite suas perguntas (ou 'quit' para sair)")
    print("Comandos especiais:")
    print("  - 'stats': Mostrar estatísticas do grafo")
    print("  - 'entities': Listar todas as entidades")
    print("  - 'help': Mostrar esta ajuda")
    
    while True:
        try:
            user_input = input("\n🔍 Sua pergunta: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'sair']:
                print("👋 Até logo!")
                break
            
            elif user_input.lower() == 'stats':
                stats = kg.get_statistics()
                print(f"📊 Estatísticas:")
                print(f"   - Entidades: {stats['total_entities']}")
                print(f"   - Relacionamentos: {stats['total_relationships']}")
                print(f"   - Densidade: {stats['density']:.3f}")
                print(f"   - Conectado: {'Sim' if stats['is_connected'] else 'Não'}")
            
            elif user_input.lower() == 'entities':
                print("📋 Entidades disponíveis:")
                for entity_type, count in kg.get_statistics()['entity_types'].items():
                    print(f"   - {entity_type}: {count}")
                    entities_of_type = [e for e in kg.entities.values() if e.type == entity_type]
                    for entity in entities_of_type[:3]:  # Mostrar apenas 3 exemplos
                        name = entity.properties.get('name', entity.id)
                        print(f"     • {name}")
                    if len(entities_of_type) > 3:
                        print(f"     ... e mais {len(entities_of_type) - 3}")
            
            elif user_input.lower() == 'help':
                print("💡 Exemplos de perguntas:")
                print("   - Quem é Luke Skywalker?")
                print("   - Qual é a relação entre Luke e Vader?")
                print("   - Quais planetas existem no universo?")
                print("   - Quem são os Jedi?")
            
            elif user_input:
                if rag_system:
                    try:
                        result = rag_system.query(user_input)
                        print(f"\n🤖 {result['answer']}")
                        
                        if result['confidence'] < 0.5:
                            print("⚠️  Resposta com baixa confiança - pode não ser precisa")
                            
                    except Exception as e:
                        print(f"❌ Erro na consulta LLM: {e}")
                        print("💡 Tentando busca simples no grafo...")
                        
                        # Fallback para busca simples
                        results = kg.search_entities(user_input)
                        if results:
                            print(f"🔍 Encontrei {len(results)} entidades relacionadas:")
                            for entity in results[:3]:
                                name = entity.properties.get('name', entity.id)
                                print(f"   - {name} ({entity.type})")
                        else:
                            print("🤷 Não encontrei informações relacionadas")
                else:
                    # Busca simples sem LLM
                    results = kg.search_entities(user_input)
                    if results:
                        print(f"🔍 Encontrei {len(results)} entidades:")
                        for entity in results:
                            name = entity.properties.get('name', entity.id)
                            print(f"   - {name} ({entity.type})")
                            
                            # Mostrar relacionamentos
                            neighbors = kg.get_neighbors(entity.id)
                            if neighbors:
                                print(f"     Conectado com: {', '.join(neighbors[:3])}")
                                if len(neighbors) > 3:
                                    print(f"     ... e mais {len(neighbors) - 3}")
                    else:
                        print("🤷 Não encontrei entidades relacionadas")
                        
        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

def main():
    """Função principal"""
    print("🌌 Knowledge Graph com LLM - Alternativa ao Neo4j")
    print("📚 Projeto: Algoritmos Avançados")
    print("🛠️  Tecnologias: NetworkX + LangChain + Google Gemini")
    
    # Configurar ambiente
    has_llm = setup_environment()
    
    # Criar Knowledge Graph
    kg = create_sample_knowledge_graph()
    
    # Validar dados
    print("\n🔍 Validando dados...")
    validation = DataValidator.validate_graph(kg)
    if validation['valid']:
        print("✅ Dados válidos!")
    else:
        print("⚠️  Problemas encontrados:")
        for issue in validation['issues']:
            print(f"   - {issue}")
    
    # Demonstrações
    demonstrate_graph_queries(kg)
    
    if has_llm:
        demonstrate_llm_integration(kg)
        
        # Configurar sistema RAG para modo interativo
        try:
            rag_system = GraphRAGSystem(kg)
            rag_system.build_vector_store()
            rag_system.setup_qa_chain()
        except Exception as e:
            print(f"⚠️  Erro ao configurar RAG: {e}")
            rag_system = None
    else:
        rag_system = None
    
    # Modo interativo
    print("\n" + "="*50)
    interactive_mode(kg, rag_system)

if __name__ == "__main__":
    main()