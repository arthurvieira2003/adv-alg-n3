import networkx as nx
import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class Entity:
    """Representa uma entidade no grafo"""
    id: str
    type: str
    properties: Dict[str, Any]
    
@dataclass
class Relationship:
    """Representa um relacionamento no grafo"""
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class KnowledgeGraph:
    """Knowledge Graph usando NetworkX como alternativa ao Neo4j"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relationships = []
        
    def add_entity(self, entity: Entity) -> None:
        """Adiciona uma entidade ao grafo"""
        self.entities[entity.id] = entity
        # Evita conflito se 'type' estiver em properties
        node_attrs = entity.properties.copy()
        node_attrs['type'] = entity.type
        self.graph.add_node(entity.id, **node_attrs)
        
    def add_relationship(self, relationship: Relationship) -> None:
        """Adiciona um relacionamento ao grafo"""
        self.relationships.append(relationship)
        # Evita conflito se 'type' estiver em properties
        edge_attrs = relationship.properties.copy()
        edge_attrs['type'] = relationship.type
        self.graph.add_edge(
            relationship.source,
            relationship.target,
            **edge_attrs
        )
        
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Recupera uma entidade pelo ID"""
        return self.entities.get(entity_id)
        
    def get_neighbors(self, entity_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """Recupera vizinhos de uma entidade (considera ambas as direções)"""
        if entity_id not in self.graph:
            return []
            
        neighbors = set()
        
        # Vizinhos de saída (outgoing)
        for neighbor in self.graph.neighbors(entity_id):
            if relationship_type:
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if any(data.get('type') == relationship_type for data in edge_data.values()):
                    neighbors.add(neighbor)
            else:
                neighbors.add(neighbor)
        
        # Vizinhos de entrada (incoming)
        for predecessor in self.graph.predecessors(entity_id):
            if relationship_type:
                edge_data = self.graph.get_edge_data(predecessor, entity_id)
                if any(data.get('type') == relationship_type for data in edge_data.values()):
                    neighbors.add(predecessor)
            else:
                neighbors.add(predecessor)
                
        return list(neighbors)
        
    def find_path(self, source: str, target: str, max_length: int = 5) -> List[str]:
        """Encontra caminho entre duas entidades"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []
            
    def get_subgraph(self, entity_ids: List[str], depth: int = 1) -> nx.Graph:
        """Extrai subgrafo centrado em entidades específicas"""
        nodes_to_include = set(entity_ids)
        
        for _ in range(depth):
            new_nodes = set()
            for node in nodes_to_include:
                if node in self.graph:
                    new_nodes.update(self.graph.neighbors(node))
            nodes_to_include.update(new_nodes)
            
        return self.graph.subgraph(nodes_to_include)
        
    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Busca entidades por texto"""
        results = []
        query_lower = query.lower()
        
        for entity in self.entities.values():
            if entity_type and entity.type != entity_type:
                continue
                
            # Busca no ID e propriedades
            if query_lower in entity.id.lower():
                results.append(entity)
                continue
                
            for value in entity.properties.values():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(entity)
                    break
                    
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do grafo"""
        entity_types = {}
        relationship_types = {}
        
        for entity in self.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
            
        for rel in self.relationships:
            relationship_types[rel.type] = relationship_types.get(rel.type, 0) + 1
            
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'entity_types': entity_types,
            'relationship_types': relationship_types,
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }
        
    def visualize(self, layout: str = 'spring', figsize: Tuple[int, int] = (12, 8)) -> None:
        """Visualiza o grafo usando matplotlib"""
        plt.figure(figsize=figsize)
        
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.random_layout(self.graph)
            
        # Cores por tipo de entidade
        entity_types = list(set(entity.type for entity in self.entities.values()))
        colors = plt.cm.Set3(range(len(entity_types)))
        type_color_map = dict(zip(entity_types, colors))
        
        node_colors = [type_color_map[self.entities[node].type] for node in self.graph.nodes()]
        
        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=500,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                arrows=True,
                edge_color='gray',
                alpha=0.7)
        
        plt.title('Knowledge Graph Visualization')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def visualize_interactive(self) -> go.Figure:
        """Visualização interativa usando Plotly"""
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Preparar dados dos nós
        node_trace = go.Scatter(
            x=[pos[node][0] for node in self.graph.nodes()],
            y=[pos[node][1] for node in self.graph.nodes()],
            mode='markers+text',
            text=[node for node in self.graph.nodes()],
            textposition='middle center',
            hovertemplate='<b>%{text}</b><br>Type: %{customdata}<extra></extra>',
            customdata=[self.entities[node].type for node in self.graph.nodes()],
            marker=dict(
                size=20,
                color=[hash(self.entities[node].type) % 10 for node in self.graph.nodes()],
                colorscale='Viridis',
                line=dict(width=2, color='black')
            )
        )
        
        # Preparar dados das arestas
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Knowledge Graph - Interactive Visualization',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        return fig
        
    def save_to_file(self, filepath: str) -> None:
        """Salva o grafo em arquivo JSON"""
        data = {
            'entities': {
                eid: {
                    'type': entity.type,
                    'properties': entity.properties
                } for eid, entity in self.entities.items()
            },
            'relationships': [
                {
                    'source': rel.source,
                    'target': rel.target,
                    'type': rel.type,
                    'properties': rel.properties
                } for rel in self.relationships
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def load_from_file(self, filepath: str) -> None:
        """Carrega o grafo de arquivo JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Limpar grafo atual
        self.graph.clear()
        self.entities.clear()
        self.relationships.clear()
        
        # Carregar entidades
        for eid, entity_data in data['entities'].items():
            entity = Entity(
                id=eid,
                type=entity_data['type'],
                properties=entity_data['properties']
            )
            self.add_entity(entity)
            
        # Carregar relacionamentos
        for rel_data in data['relationships']:
            relationship = Relationship(
                source=rel_data['source'],
                target=rel_data['target'],
                type=rel_data['type'],
                properties=rel_data['properties']
            )
            self.add_relationship(relationship)