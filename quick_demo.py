#!/usr/bin/env python3
"""
Demonstração rápida do Knowledge Graph Star Wars
Script simples para mostrar as funcionalidades principais
"""

import os
import sys

# Adicionar diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import KnowledgeGraph
from data_loader import StarWarsDataLoader
from config import Config

def main():
    print("🌌 Knowledge Graph Star Wars - Demonstração Rápida")
    print("="*60)
    
    # 1. Carregar dados
    print("\n📊 1. Carregando dados do universo Star Wars...")
    loader = StarWarsDataLoader()
    kg = loader.create_sample_data()
    
    stats = kg.get_statistics()
    print(f"✅ Carregado: {stats['total_entities']} entidades, {stats['total_relationships']} relacionamentos")
    
    # 2. Explorar entidades
    print("\n👥 2. Personagens principais:")
    characters = [e for e in kg.entities.values() if e.type == "Character"]
    for char in characters:
        name = char.properties.get('name', char.id)
        homeworld = char.properties.get('homeworld', 'Desconhecido')
        print(f"   - {name} (de {homeworld})")
    
    # 3. Relacionamentos familiares
    print("\n👨‍👩‍👧‍👦 3. Relacionamentos familiares:")
    family_rels = [r for r in kg.relationships if "FATHER_OF" in r.type or "SIBLING_OF" in r.type]
    for rel in family_rels:
        source_name = kg.get_entity(rel.source).properties.get('name', rel.source)
        target_name = kg.get_entity(rel.target).properties.get('name', rel.target)
        rel_type = rel.type.replace('_', ' ').lower()
        print(f"   - {source_name} é {rel_type} {target_name}")
    
    # 4. Análise de Luke Skywalker
    print("\n🌟 4. Análise detalhada: Luke Skywalker")
    luke = kg.get_entity("luke_skywalker")
    if luke:
        print(f"   Nome: {luke.properties.get('name')}")
        print(f"   Espécie: {luke.properties.get('species')}")
        print(f"   Planeta natal: {luke.properties.get('homeworld')}")
        print(f"   Sensível à Força: {'Sim' if luke.properties.get('force_sensitive') else 'Não'}")
        
        # Conexões
        neighbors = kg.get_neighbors("luke_skywalker")
        print(f"   Conectado com {len(neighbors)} entidades:")
        for neighbor in neighbors:
            neighbor_entity = kg.get_entity(neighbor)
            if neighbor_entity:
                name = neighbor_entity.properties.get('name', neighbor)
                print(f"     • {name} ({neighbor_entity.type})")
    
    # 5. Caminho entre Luke e Yoda
    print("\n🛤️  5. Caminho entre Luke Skywalker e Yoda:")
    from llm_integration import GraphQueryProcessor
    query_processor = GraphQueryProcessor(kg)
    
    path_result = query_processor.find_shortest_path("luke_skywalker", "yoda")
    if path_result["exists"]:
        print(f"   Distância: {path_result['length']} passos")
        print("   Caminho:")
        for i, step in enumerate(path_result["path_details"], 1):
            source_name = kg.get_entity(step['from']).properties.get('name', step['from'])
            target_name = kg.get_entity(step['to']).properties.get('name', step['to'])
            print(f"     {i}. {source_name} --[{step['relationship']}]--> {target_name}")
    else:
        print("   Nenhum caminho direto encontrado")
    
    # 6. Entidades mais conectadas
    print("\n🌐 6. Entidades mais conectadas:")
    connectivity = {}
    for entity_id in kg.entities.keys():
        connectivity[entity_id] = len(kg.get_neighbors(entity_id))
    
    top_connected = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (entity_id, connections) in enumerate(top_connected, 1):
        entity = kg.get_entity(entity_id)
        name = entity.properties.get('name', entity_id)
        print(f"   {i}. {name}: {connections} conexões")
    
    # 7. Busca por texto
    print("\n🔍 7. Busca por 'Skywalker':")
    results = kg.search_entities("Skywalker")
    for entity in results:
        name = entity.properties.get('name', entity.id)
        print(f"   - {name} ({entity.type})")
    
    # 8. Estatísticas finais
    print("\n📈 8. Estatísticas do grafo:")
    print(f"   - Densidade: {stats['density']:.3f}")
    print(f"   - Conectado: {'Sim' if stats['is_connected'] else 'Não'}")
    print(f"   - Tipos de entidades: {len(stats['entity_types'])}")
    print(f"   - Tipos de relacionamentos: {len(stats['relationship_types'])}")
    
    # 9. Informações sobre LLM
    print("\n🤖 9. Funcionalidades de LLM:")
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print("   ✅ Google API Key configurada")
        print("   💡 Execute 'python main.py' para chat interativo")
        print("   🌐 Execute 'streamlit run streamlit_app.py' para interface web")
    else:
        print("   ⚠️  Google API Key não configurada")
        print("   💡 Configure GOOGLE_API_KEY para usar funcionalidades de LLM")
        print("   📝 Copie .env.example para .env e adicione sua chave")
    
    print("\n" + "="*60)
    print("🎉 Demonstração concluída!")
    print("\n📚 Próximos passos:")
    print("   1. Configure sua Google API Key")
    print("   2. Execute 'python main.py' para modo interativo")
    print("   3. Execute 'streamlit run streamlit_app.py' para interface web")
    print("   4. Execute 'python test_system.py' para testes completos")
    

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demonstração interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro na demonstração: {e}")
        print("💡 Execute 'python test_system.py' para diagnosticar problemas")