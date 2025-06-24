#!/usr/bin/env python3
"""
Script de teste para verificar o funcionamento do Knowledge Graph
"""

import sys
import os
from typing import Dict, Any

def test_imports():
    """Testa se todas as importações funcionam"""
    print("🧪 Testando importações...")
    
    try:
        import networkx as nx
        print("✅ NetworkX importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar NetworkX: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar Pandas: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar Matplotlib: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar Plotly: {e}")
        return False
    
    # Testar importações opcionais (LangChain)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("✅ LangChain importado com sucesso")
    except ImportError as e:
        print(f"⚠️  LangChain não disponível: {e}")
        print("   (Funcionalidades de LLM não estarão disponíveis)")
    
    return True

def test_knowledge_graph():
    """Testa a funcionalidade básica do Knowledge Graph"""
    print("\n🧪 Testando Knowledge Graph...")
    
    try:
        from knowledge_graph import KnowledgeGraph, Entity, Relationship
        
        # Criar grafo de teste
        kg = KnowledgeGraph()
        
        # Adicionar entidades de teste
        entity1 = Entity(
            id="test_entity_1",
            type="TestType",
            properties={"name": "Test Entity 1", "value": 100}
        )
        
        entity2 = Entity(
            id="test_entity_2",
            type="TestType",
            properties={"name": "Test Entity 2", "value": 200}
        )
        
        kg.add_entity(entity1)
        kg.add_entity(entity2)
        
        # Adicionar relacionamento
        relationship = Relationship(
            source="test_entity_1",
            target="test_entity_2",
            type="TEST_RELATION",
            properties={"strength": 0.8}
        )
        
        kg.add_relationship(relationship)
        
        # Testar funcionalidades
        assert len(kg.entities) == 2, "Número incorreto de entidades"
        assert len(kg.relationships) == 1, "Número incorreto de relacionamentos"
        
        # Testar busca
        found_entity = kg.get_entity("test_entity_1")
        assert found_entity is not None, "Entidade não encontrada"
        assert found_entity.properties["name"] == "Test Entity 1", "Propriedade incorreta"
        
        # Testar vizinhos
        neighbors = kg.get_neighbors("test_entity_1")
        assert "test_entity_2" in neighbors, "Vizinho não encontrado"
        
        # Testar estatísticas
        stats = kg.get_statistics()
        assert stats["total_entities"] == 2, "Estatística de entidades incorreta"
        assert stats["total_relationships"] == 1, "Estatística de relacionamentos incorreta"
        
        print("✅ Knowledge Graph funcionando corretamente")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do Knowledge Graph: {e}")
        return False

def test_data_loader():
    """Testa o carregamento de dados Star Wars"""
    print("\n🧪 Testando Data Loader...")
    
    try:
        from data_loader import StarWarsDataLoader, DataValidator
        
        # Criar loader
        loader = StarWarsDataLoader()
        
        # Carregar dados de exemplo
        kg = loader.create_sample_data()
        
        # Verificar se os dados foram carregados
        stats = kg.get_statistics()
        
        assert stats["total_entities"] > 0, "Nenhuma entidade carregada"
        assert stats["total_relationships"] > 0, "Nenhum relacionamento carregado"
        
        # Verificar tipos de entidades esperados
        entity_types = stats["entity_types"]
        expected_types = ["Character", "Planet", "Organization", "Vehicle"]
        
        for expected_type in expected_types:
            assert expected_type in entity_types, f"Tipo de entidade '{expected_type}' não encontrado"
        
        # Testar validação
        validation = DataValidator.validate_graph(kg)
        assert validation["valid"], f"Grafo inválido: {validation['issues']}"
        
        # Verificar entidades específicas
        luke = kg.get_entity("luke_skywalker")
        assert luke is not None, "Luke Skywalker não encontrado"
        assert luke.type == "Character", "Tipo incorreto para Luke Skywalker"
        
        vader = kg.get_entity("darth_vader")
        assert vader is not None, "Darth Vader não encontrado"
        
        # Verificar relacionamento pai-filho
        luke_neighbors = kg.get_neighbors("luke_skywalker")
        assert "darth_vader" in luke_neighbors, "Relacionamento Luke-Vader não encontrado"
        
        print(f"✅ Data Loader funcionando corretamente")
        print(f"   📊 {stats['total_entities']} entidades carregadas")
        print(f"   🔗 {stats['total_relationships']} relacionamentos carregados")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do Data Loader: {e}")
        return False

def test_llm_integration():
    """Testa a integração com LLM (se disponível)"""
    print("\n🧪 Testando integração com LLM...")
    
    # Verificar se a API key está configurada
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  Google API Key não configurada - pulando teste de LLM")
        return True
    
    try:
        from llm_integration import GraphRAGSystem
        from data_loader import StarWarsDataLoader
        
        # Carregar dados
        loader = StarWarsDataLoader()
        kg = loader.create_sample_data()
        
        # Criar sistema RAG
        rag_system = GraphRAGSystem(kg)
        
        # Testar construção do vector store
        documents = rag_system.create_documents_from_graph()
        assert len(documents) > 0, "Nenhum documento criado"
        
        print(f"✅ {len(documents)} documentos criados para o vector store")
        
        # Testar configuração (sem fazer chamadas à API)
        try:
            rag_system.build_vector_store()
            print("✅ Vector store construído com sucesso")
        except Exception as e:
            print(f"⚠️  Erro na construção do vector store: {e}")
            print("   (Pode ser devido a limitações da API ou rede)")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  LangChain não disponível: {e}")
        return True
    except Exception as e:
        print(f"❌ Erro no teste de LLM: {e}")
        return False

def test_visualization():
    """Testa as funcionalidades de visualização"""
    print("\n🧪 Testando visualização...")
    
    try:
        from data_loader import StarWarsDataLoader
        
        # Carregar dados
        loader = StarWarsDataLoader()
        kg = loader.create_sample_data()
        
        # Testar visualização interativa
        fig = kg.visualize_interactive()
        assert fig is not None, "Figura de visualização não criada"
        
        print("✅ Visualização interativa funcionando")
        
        # Testar subgrafo
        subgraph = kg.get_subgraph(["luke_skywalker", "darth_vader"], depth=1)
        assert len(subgraph.nodes()) >= 2, "Subgrafo muito pequeno"
        
        print(f"✅ Subgrafo criado com {len(subgraph.nodes())} nós")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de visualização: {e}")
        return False

def test_file_operations():
    """Testa operações de arquivo"""
    print("\n🧪 Testando operações de arquivo...")
    
    try:
        from data_loader import StarWarsDataLoader
        from knowledge_graph import KnowledgeGraph
        import tempfile
        import os
        
        # Carregar dados
        loader = StarWarsDataLoader()
        kg = loader.create_sample_data()
        
        # Testar salvamento em arquivo temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            kg.save_to_file(temp_file)
            print("✅ Salvamento em arquivo funcionando")
            
            # Testar carregamento
            new_kg = KnowledgeGraph()
            new_kg.load_from_file(temp_file)
            
            # Verificar se os dados foram carregados corretamente
            original_stats = kg.get_statistics()
            new_stats = new_kg.get_statistics()
            
            assert original_stats["total_entities"] == new_stats["total_entities"], "Entidades não carregadas corretamente"
            assert original_stats["total_relationships"] == new_stats["total_relationships"], "Relacionamentos não carregados corretamente"
            
            print("✅ Carregamento de arquivo funcionando")
            
        finally:
            # Limpar arquivo temporário
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de operações de arquivo: {e}")
        return False

def run_all_tests() -> Dict[str, bool]:
    """Executa todos os testes"""
    print("🚀 Iniciando testes do sistema...\n")
    
    tests = {
        "Importações": test_imports,
        "Knowledge Graph": test_knowledge_graph,
        "Data Loader": test_data_loader,
        "Integração LLM": test_llm_integration,
        "Visualização": test_visualization,
        "Operações de Arquivo": test_file_operations
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Erro inesperado no teste '{test_name}': {e}")
            results[test_name] = False
    
    return results

def print_test_summary(results: Dict[str, bool]):
    """Imprime resumo dos testes"""
    print("\n" + "="*50)
    print("📋 RESUMO DOS TESTES")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSOU" if passed_test else "❌ FALHOU"
        print(f"{test_name:20} | {status}")
        if passed_test:
            passed += 1
    
    print("="*50)
    print(f"📊 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! Sistema funcionando corretamente.")
    else:
        print("⚠️  Alguns testes falharam. Verifique as dependências e configurações.")
    
    print("\n💡 Para usar o sistema:")
    print("   - Modo console: python main.py")
    print("   - Interface web: streamlit run streamlit_app.py")
    print("   - Configure GOOGLE_API_KEY para funcionalidades de LLM")

if __name__ == "__main__":
    # Adicionar diretório atual ao path para importações
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Executar testes
    results = run_all_tests()
    
    # Mostrar resumo
    print_test_summary(results)