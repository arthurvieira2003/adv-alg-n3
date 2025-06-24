from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# FAISS removed due to Windows compilation issues
# Using simple text similarity instead
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import json
import re
from difflib import SequenceMatcher
from knowledge_graph import KnowledgeGraph, Entity, Relationship
from config import Config

class SimpleVectorStore:
    """Vector store simples usando similaridade de texto"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Busca por similaridade usando SequenceMatcher"""
        scores = []
        query_lower = query.lower()
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            # Calcular similaridade
            similarity = SequenceMatcher(None, query_lower, content_lower).ratio()
            
            # Bonus para palavras-chave exatas
            query_words = set(re.findall(r'\w+', query_lower))
            content_words = set(re.findall(r'\w+', content_lower))
            word_overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
            
            final_score = similarity * 0.7 + word_overlap * 0.3
            scores.append((final_score, doc))
            
        # Ordenar por score e retornar top k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:k]]

class GraphRAGSystem:
    """Sistema RAG integrado com Knowledge Graph"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.vector_store = None
        self.qa_chain = None
        
    def create_documents_from_graph(self) -> List[Document]:
        """Converte o grafo em documentos para o vector store"""
        documents = []
        
        # Documentos das entidades
        for entity in self.kg.entities.values():
            content = f"Entidade: {entity.id}\n"
            content += f"Tipo: {entity.type}\n"
            
            for key, value in entity.properties.items():
                content += f"{key}: {value}\n"
                
            # Adicionar relacionamentos
            neighbors = self.kg.get_neighbors(entity.id)
            if neighbors:
                content += f"Relacionado com: {', '.join(neighbors)}\n"
                
            documents.append(Document(
                page_content=content,
                metadata={
                    'entity_id': entity.id,
                    'entity_type': entity.type,
                    'source': 'entity'
                }
            ))
            
        # Documentos dos relacionamentos
        for rel in self.kg.relationships:
            content = f"Relacionamento: {rel.source} -{rel.type}-> {rel.target}\n"
            
            for key, value in rel.properties.items():
                content += f"{key}: {value}\n"
                
            documents.append(Document(
                page_content=content,
                metadata={
                    'source_entity': rel.source,
                    'target_entity': rel.target,
                    'relationship_type': rel.type,
                    'source': 'relationship'
                }
            ))
            
        return documents
        
    def build_vector_store(self) -> None:
        """Constrói o vector store a partir do grafo"""
        documents = self.create_documents_from_graph()
        
        # Dividir documentos se necessário
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Criar vector store simples
        self.vector_store = SimpleVectorStore(split_docs)
        
    def setup_qa_chain(self) -> None:
        """Configura a cadeia de QA"""
        if not self.vector_store:
            self.build_vector_store()
            
        # Template de prompt personalizado
        prompt_template = """
Você é um assistente especializado em responder perguntas baseado em um Knowledge Graph.
Use apenas as informações fornecidas no contexto para responder às perguntas.
Se a informação não estiver disponível no contexto, diga que não sabe.

Contexto do Knowledge Graph:
{context}

Pergunta: {question}

Resposta detalhada:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Criar cadeia de QA personalizada (sem retriever)
        self.prompt = PROMPT
        self.qa_chain = True  # Flag para indicar que está configurado
        
    def query(self, question: str) -> Dict[str, Any]:
        """Executa uma consulta em linguagem natural"""
        if not self.qa_chain:
            self.setup_qa_chain()
            
        # Buscar documentos relevantes
        relevant_docs = self.vector_store.similarity_search(question, k=Config.TOP_K_RESULTS)
        
        # Criar contexto a partir dos documentos
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Gerar resposta usando o LLM
        prompt_text = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_text)
        
        # Extrair entidades mencionadas na resposta
        answer_text = response.content if hasattr(response, 'content') else str(response)
        mentioned_entities = self._extract_mentioned_entities(answer_text)
        
        # Buscar informações adicionais do grafo
        graph_context = self._get_graph_context(mentioned_entities)
        
        return {
            "answer": answer_text,
            "source_documents": relevant_docs,
            "mentioned_entities": mentioned_entities,
            "graph_context": graph_context,
            "confidence": len(relevant_docs) / Config.TOP_K_RESULTS  # Confiança simples
        }
        
    def _extract_mentioned_entities(self, text: str) -> List[str]:
        """Extrai entidades mencionadas no texto"""
        mentioned = []
        text_lower = text.lower()
        
        for entity_id in self.kg.entities.keys():
            if entity_id.lower() in text_lower:
                mentioned.append(entity_id)
                
        return mentioned
        
    def _get_graph_context(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Obtém contexto adicional do grafo para as entidades mencionadas"""
        context = {}
        
        for entity_id in entity_ids:
            entity = self.kg.get_entity(entity_id)
            if entity:
                neighbors = self.kg.get_neighbors(entity_id)
                context[entity_id] = {
                    "type": entity.type,
                    "properties": entity.properties,
                    "neighbors": neighbors
                }
                
        return context
        
    # Método _calculate_confidence removido - confiança agora calculada diretamente no query()
        
    def explain_reasoning(self, question: str) -> Dict[str, Any]:
        """Explica o raciocínio por trás de uma resposta"""
        result = self.query(question)
        
        explanation = {
            "question": question,
            "answer": result["answer"],
            "reasoning_steps": [],
            "graph_evidence": [],
            "confidence": result["confidence"]
        }
        
        # Analisar documentos fonte
        for doc in result["source_documents"]:
            if doc.metadata.get("source") == "entity":
                explanation["reasoning_steps"].append(
                    f"Consultei informações sobre a entidade: {doc.metadata.get('entity_id')}"
                )
            elif doc.metadata.get("source") == "relationship":
                explanation["reasoning_steps"].append(
                    f"Analisei o relacionamento: {doc.metadata.get('source_entity')} -> {doc.metadata.get('target_entity')}"
                )
                
        # Evidências do grafo
        for entity_id, context in result["graph_context"].items():
            explanation["graph_evidence"].append({
                "entity": entity_id,
                "type": context["type"],
                "connections": len(context["neighbors"])
            })
            
        return explanation
        
    def suggest_related_questions(self, question: str) -> List[str]:
        """Sugere perguntas relacionadas baseadas no grafo"""
        result = self.query(question)
        suggestions = []
        
        # Baseado nas entidades mencionadas
        for entity_id in result["mentioned_entities"]:
            entity = self.kg.get_entity(entity_id)
            if entity:
                neighbors = self.kg.get_neighbors(entity_id)
                
                for neighbor in neighbors[:3]:  # Limitar a 3 sugestões por entidade
                    suggestions.append(
                        f"Qual é a relação entre {entity_id} e {neighbor}?"
                    )
                    
                # Sugestões baseadas no tipo da entidade
                same_type_entities = [
                    e.id for e in self.kg.entities.values() 
                    if e.type == entity.type and e.id != entity_id
                ][:2]
                
                for similar_entity in same_type_entities:
                    suggestions.append(
                        f"Compare {entity_id} com {similar_entity}"
                    )
                    
        return list(set(suggestions))[:5]  # Remover duplicatas e limitar a 5

class GraphQueryProcessor:
    """Processador de consultas específicas do grafo"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        
    def process_cypher_like_query(self, query: str) -> List[Dict[str, Any]]:
        """Processa consultas similares ao Cypher do Neo4j"""
        # Implementação simplificada de consultas estruturadas
        results = []
        
        # Exemplo: "MATCH (n:Person) RETURN n"
        if "MATCH" in query.upper() and "RETURN" in query.upper():
            # Parse básico da consulta
            if ":" in query:
                entity_type = query.split(":")[1].split(")")[0]
                
                for entity in self.kg.entities.values():
                    if entity.type.lower() == entity_type.lower():
                        results.append({
                            "id": entity.id,
                            "type": entity.type,
                            "properties": entity.properties
                        })
                        
        return results
        
    def find_shortest_path(self, source: str, target: str) -> Dict[str, Any]:
        """Encontra o caminho mais curto entre duas entidades"""
        path = self.kg.find_path(source, target)
        
        if not path:
            return {"path": [], "length": 0, "exists": False}
            
        # Adicionar informações dos relacionamentos no caminho
        path_details = []
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Encontrar tipo de relacionamento
            edge_data = self.kg.graph.get_edge_data(current, next_node)
            rel_type = "unknown"
            if edge_data:
                rel_type = list(edge_data.values())[0].get("type", "unknown")
                
            path_details.append({
                "from": current,
                "to": next_node,
                "relationship": rel_type
            })
            
        return {
            "path": path,
            "path_details": path_details,
            "length": len(path) - 1,
            "exists": True
        }