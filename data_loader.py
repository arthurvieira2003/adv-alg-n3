import json
import pandas as pd
from typing import Dict, List, Any
from knowledge_graph import KnowledgeGraph, Entity, Relationship
import os

class StarWarsDataLoader:
    """Carregador de dados do universo Star Wars"""
    
    def __init__(self):
        self.kg = KnowledgeGraph()
        
    def create_sample_data(self) -> KnowledgeGraph:
        """Cria dados de exemplo do universo Star Wars"""
        
        # Personagens
        characters = [
            {
                "id": "luke_skywalker",
                "name": "Luke Skywalker",
                "species": "Human",
                "homeworld": "Tatooine",
                "affiliation": "Rebel Alliance",
                "force_sensitive": True,
                "description": "Jedi Knight who destroyed the Death Star"
            },
            {
                "id": "darth_vader",
                "name": "Darth Vader",
                "species": "Human",
                "homeworld": "Tatooine",
                "affiliation": "Galactic Empire",
                "force_sensitive": True,
                "description": "Sith Lord, former Jedi Anakin Skywalker"
            },
            {
                "id": "princess_leia",
                "name": "Princess Leia Organa",
                "species": "Human",
                "homeworld": "Alderaan",
                "affiliation": "Rebel Alliance",
                "force_sensitive": True,
                "description": "Leader of the Rebel Alliance"
            },
            {
                "id": "han_solo",
                "name": "Han Solo",
                "species": "Human",
                "homeworld": "Corellia",
                "affiliation": "Rebel Alliance",
                "force_sensitive": False,
                "description": "Smuggler and pilot of the Millennium Falcon"
            },
            {
                "id": "obi_wan_kenobi",
                "name": "Obi-Wan Kenobi",
                "species": "Human",
                "homeworld": "Stewjon",
                "affiliation": "Jedi Order",
                "force_sensitive": True,
                "description": "Jedi Master and mentor to Luke Skywalker"
            },
            {
                "id": "yoda",
                "name": "Yoda",
                "species": "Unknown",
                "homeworld": "Unknown",
                "affiliation": "Jedi Order",
                "force_sensitive": True,
                "description": "Grand Master of the Jedi Order"
            }
        ]
        
        # Planetas
        planets = [
            {
                "id": "tatooine",
                "name": "Tatooine",
                "climate": "Arid",
                "terrain": "Desert",
                "population": "200000",
                "description": "Desert planet with twin suns"
            },
            {
                "id": "alderaan",
                "name": "Alderaan",
                "climate": "Temperate",
                "terrain": "Grasslands, mountains",
                "population": "2000000000",
                "description": "Peaceful planet destroyed by Death Star"
            },
            {
                "id": "coruscant",
                "name": "Coruscant",
                "climate": "Temperate",
                "terrain": "Cityscape",
                "population": "1000000000000",
                "description": "Capital of the Galactic Republic and Empire"
            },
            {
                "id": "dagobah",
                "name": "Dagobah",
                "climate": "Murky",
                "terrain": "Swamp, jungles",
                "population": "Unknown",
                "description": "Swamp planet where Yoda lived in exile"
            },
            {
                "id": "corellia",
                "name": "Corellia",
                "climate": "Temperate",
                "terrain": "Plains, urban areas",
                "population": "3000000000",
                "description": "Industrial world known for shipbuilding and smugglers"
            }
        ]
        
        # Organizações
        organizations = [
            {
                "id": "jedi_order",
                "name": "Jedi Order",
                "type": "Religious Order",
                "alignment": "Light Side",
                "description": "Ancient order of Force-sensitive peacekeepers"
            },
            {
                "id": "sith",
                "name": "Sith",
                "type": "Religious Order",
                "alignment": "Dark Side",
                "description": "Ancient order of Force-sensitive dark warriors"
            },
            {
                "id": "rebel_alliance",
                "name": "Rebel Alliance",
                "type": "Military Organization",
                "alignment": "Good",
                "description": "Resistance movement against the Galactic Empire"
            },
            {
                "id": "galactic_empire",
                "name": "Galactic Empire",
                "type": "Government",
                "alignment": "Evil",
                "description": "Authoritarian regime ruling the galaxy"
            }
        ]
        
        # Veículos/Naves
        vehicles = [
            {
                "id": "millennium_falcon",
                "name": "Millennium Falcon",
                "type": "Light Freighter",
                "manufacturer": "Corellian Engineering Corporation",
                "description": "Fast smuggling ship owned by Han Solo"
            },
            {
                "id": "death_star",
                "name": "Death Star",
                "type": "Space Station",
                "manufacturer": "Galactic Empire",
                "description": "Moon-sized battle station with planet-destroying capability"
            },
            {
                "id": "x_wing",
                "name": "X-wing Starfighter",
                "type": "Starfighter",
                "manufacturer": "Incom Corporation",
                "description": "Versatile starfighter used by the Rebel Alliance"
            }
        ]
        
        # Adicionar entidades ao grafo
        for char in characters:
            entity = Entity(
                id=char["id"],
                type="Character",
                properties=char
            )
            self.kg.add_entity(entity)
            
        for planet in planets:
            entity = Entity(
                id=planet["id"],
                type="Planet",
                properties=planet
            )
            self.kg.add_entity(entity)
            
        for org in organizations:
            entity = Entity(
                id=org["id"],
                type="Organization",
                properties=org
            )
            self.kg.add_entity(entity)
            
        for vehicle in vehicles:
            entity = Entity(
                id=vehicle["id"],
                type="Vehicle",
                properties=vehicle
            )
            self.kg.add_entity(entity)
            
        # Relacionamentos
        relationships = [
            # Família
            ("darth_vader", "luke_skywalker", "FATHER_OF", {"relationship": "father-son"}),
            ("darth_vader", "princess_leia", "FATHER_OF", {"relationship": "father-daughter"}),
            ("luke_skywalker", "princess_leia", "SIBLING_OF", {"relationship": "twin siblings"}),
            
            # Mentoria
            ("obi_wan_kenobi", "luke_skywalker", "MENTOR_OF", {"relationship": "Jedi training"}),
            ("yoda", "luke_skywalker", "MENTOR_OF", {"relationship": "Jedi training"}),
            ("obi_wan_kenobi", "darth_vader", "FORMER_MENTOR_OF", {"relationship": "former Padawan"}),
            
            # Origem/Nascimento
            ("luke_skywalker", "tatooine", "BORN_ON", {"relationship": "homeworld"}),
            ("darth_vader", "tatooine", "BORN_ON", {"relationship": "homeworld"}),
            ("princess_leia", "alderaan", "BORN_ON", {"relationship": "homeworld"}),
            ("han_solo", "corellia", "BORN_ON", {"relationship": "homeworld"}),
            
            # Afiliações
            ("luke_skywalker", "jedi_order", "MEMBER_OF", {"relationship": "Jedi Knight"}),
            ("luke_skywalker", "rebel_alliance", "MEMBER_OF", {"relationship": "pilot"}),
            ("darth_vader", "sith", "MEMBER_OF", {"relationship": "Sith Lord"}),
            ("darth_vader", "galactic_empire", "MEMBER_OF", {"relationship": "enforcer"}),
            ("princess_leia", "rebel_alliance", "LEADER_OF", {"relationship": "leader"}),
            ("han_solo", "rebel_alliance", "MEMBER_OF", {"relationship": "smuggler"}),
            ("obi_wan_kenobi", "jedi_order", "MEMBER_OF", {"relationship": "Jedi Master"}),
            ("yoda", "jedi_order", "LEADER_OF", {"relationship": "Grand Master"}),
            
            # Veículos
            ("han_solo", "millennium_falcon", "OWNS", {"relationship": "pilot and owner"}),
            ("luke_skywalker", "x_wing", "PILOTS", {"relationship": "pilot"}),
            ("galactic_empire", "death_star", "OWNS", {"relationship": "built and operated"}),
            
            # Eventos importantes
            ("luke_skywalker", "death_star", "DESTROYED", {"event": "Battle of Yavin"}),
            ("darth_vader", "obi_wan_kenobi", "KILLED", {"event": "Death Star duel"}),
            
            # Localizações
            ("yoda", "dagobah", "LIVED_ON", {"relationship": "exile location"}),
            ("galactic_empire", "coruscant", "CAPITAL_AT", {"relationship": "seat of power"}),
        ]
        
        for source, target, rel_type, properties in relationships:
            relationship = Relationship(
                source=source,
                target=target,
                type=rel_type,
                properties=properties
            )
            self.kg.add_relationship(relationship)
            
        return self.kg
        
    def load_from_csv(self, entities_file: str, relationships_file: str) -> KnowledgeGraph:
        """Carrega dados de arquivos CSV"""
        # Carregar entidades
        if os.path.exists(entities_file):
            entities_df = pd.read_csv(entities_file)
            
            for _, row in entities_df.iterrows():
                properties = row.to_dict()
                entity_id = properties.pop('id')
                entity_type = properties.pop('type')
                
                entity = Entity(
                    id=entity_id,
                    type=entity_type,
                    properties=properties
                )
                self.kg.add_entity(entity)
                
        # Carregar relacionamentos
        if os.path.exists(relationships_file):
            relationships_df = pd.read_csv(relationships_file)
            
            for _, row in relationships_df.iterrows():
                properties = row.to_dict()
                source = properties.pop('source')
                target = properties.pop('target')
                rel_type = properties.pop('type')
                
                relationship = Relationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    properties=properties
                )
                self.kg.add_relationship(relationship)
                
        return self.kg
        
    def export_to_csv(self, kg: KnowledgeGraph, entities_file: str, relationships_file: str) -> None:
        """Exporta o grafo para arquivos CSV"""
        # Exportar entidades
        entities_data = []
        for entity in kg.entities.values():
            data = {'id': entity.id, 'type': entity.type}
            data.update(entity.properties)
            entities_data.append(data)
            
        entities_df = pd.DataFrame(entities_data)
        entities_df.to_csv(entities_file, index=False)
        
        # Exportar relacionamentos
        relationships_data = []
        for rel in kg.relationships:
            data = {
                'source': rel.source,
                'target': rel.target,
                'type': rel.type
            }
            data.update(rel.properties)
            relationships_data.append(data)
            
        relationships_df = pd.DataFrame(relationships_data)
        relationships_df.to_csv(relationships_file, index=False)
        
    def create_data_directory(self) -> None:
        """Cria diretório de dados se não existir"""
        os.makedirs('data', exist_ok=True)
        
    def save_sample_data(self) -> None:
        """Salva dados de exemplo em arquivos"""
        self.create_data_directory()
        kg = self.create_sample_data()
        
        # Salvar como JSON
        kg.save_to_file('data/star_wars_knowledge_graph.json')
        
        # Salvar como CSV
        self.export_to_csv(kg, 'data/entities.csv', 'data/relationships.csv')
        
        print("Dados de exemplo salvos em:")
        print("- data/star_wars_knowledge_graph.json")
        print("- data/entities.csv")
        print("- data/relationships.csv")
        
class DataValidator:
    """Validador de dados do Knowledge Graph"""
    
    @staticmethod
    def validate_graph(kg: KnowledgeGraph) -> Dict[str, Any]:
        """Valida a integridade do grafo"""
        issues = []
        warnings = []
        
        # Verificar relacionamentos órfãos
        for rel in kg.relationships:
            if rel.source not in kg.entities:
                issues.append(f"Relacionamento órfão: entidade fonte '{rel.source}' não existe")
            if rel.target not in kg.entities:
                issues.append(f"Relacionamento órfão: entidade alvo '{rel.target}' não existe")
                
        # Verificar entidades isoladas
        connected_entities = set()
        for rel in kg.relationships:
            connected_entities.add(rel.source)
            connected_entities.add(rel.target)
            
        isolated_entities = set(kg.entities.keys()) - connected_entities
        if isolated_entities:
            warnings.append(f"Entidades isoladas: {list(isolated_entities)}")
            
        # Estatísticas
        stats = kg.get_statistics()
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "statistics": stats
        }