import argparse
import json
import re
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import random

class NERExtractor:
    def __init__(self):
        """Initialize the NER pipeline using a multilingual NER model."""
        self.ner_pipe = pipeline(
            task="ner",
            model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple",
            device_map="auto"
        )

    def extract_entities(self, text: str) -> dict:
        """
        Extract named entities from a given text using the NER pipeline and regex.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        dict
            A dictionary with keys representing entity types 
            (e.g., PER, ORG, LOC, DATE, IBAN, CODICE FISCALE, EMAIL, PHONE, MISC),
            and values as lists of unique detected entities.
        """
        result = {
            "PER": [],
            "ORG": [],
            "LOC": [],
            "DATE": [],
            "IBAN": [],
            "CODICE FISCALE": [],
            "EMAIL": [],
            "PHONE": [],
            "MISC": []
        }

        # NER model extraction
        for ent in self.ner_pipe(text):
            tag = ent["entity_group"]
            word = ent["word"].strip()
            if tag in result:
                result[tag].append(word)
            else:
                result["MISC"].append(word)

        # Regex-based extraction for (IBAN, CODICE FISCALE, EMAIL, PHONE, DATE)
        result["IBAN"].extend(re.findall(r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b', text, flags=re.IGNORECASE))
        result["CODICE FISCALE"].extend(re.findall(r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b', text, flags=re.IGNORECASE))
        result["EMAIL"].extend(re.findall(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text))
        result["PHONE"].extend(re.findall(r'\+?\d[\d\s\-]{7,}\d', text))
        result["DATE"].extend(re.findall(r'\b(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}|(?:\d{1,2}\s)?(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s\d{4})\b', text, flags=re.IGNORECASE))

        # Deduplicate (case insensitive)
        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x.lower() in seen or seen.add(x.lower()))]

        return result

    def extract_from_file(self, file_path: str) -> dict:
        """
        Load text from a file and extract entities using the NER pipeline.

        Parameters
        ----------
        file_path : str
            Path to the text file containing input data.

        Returns
        -------
        dict
            Dictionary of extracted named entities grouped by type.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.extract_entities(text)

    def normalize_entity(self, entity: str, entity_type: str) -> str:
        """
        Normalize and clean entity values based on their type.
        """
        if entity_type == "DATE":
            # Try to parse and format date to YYYY-MM-DD
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d", "%d %B %Y", "%d %b %Y", "%B %Y", "%Y"):
                try:
                    dt = datetime.strptime(entity, fmt)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    continue
            return entity  # fallback
        elif entity_type == "ORG":
            return entity.strip().lower().replace('spa', '').replace('srl', '').replace('s.p.a.', '').replace('s.r.l.', '').strip()
        elif entity_type == "PER":
            return entity.title().strip()
        elif entity_type == "IBAN":
            return entity.replace(' ', '').upper()
        elif entity_type == "EMAIL":
            return entity.lower().strip()
        elif entity_type == "PHONE":
            return re.sub(r'\D', '', entity)
        elif entity_type == "CODICE FISCALE":
            return entity.upper().strip()
        else:
            return entity.strip()

    def clean_and_normalize_entities(self, entities: dict) -> dict:
        """
        Normalize and deduplicate entities for each type.
        """
        cleaned = {}
        for ent_type, ent_list in entities.items():
            normed = [self.normalize_entity(e, ent_type) for e in ent_list]
            # Remove empty and deduplicate
            seen = set()
            cleaned[ent_type] = [x for x in normed if x and not (x.lower() in seen or seen.add(x.lower()))]
        return cleaned

    def assign_entity_types(self, entities: dict) -> dict:
        """
        Assigns a semantic type to each entity (e.g., Cliente, Fornitore, Data, Importo, etc.).
        For demo, map NER types to example business types.
        """
        type_map = {
            "PER": "Persona",
            "ORG": "Azienda",
            "LOC": "Luogo",
            "DATE": "Data",
            "IBAN": "IBAN",
            "CODICE FISCALE": "Codice Fiscale",
            "EMAIL": "Email",
            "PHONE": "Telefono",
            "MISC": "Altro"
        }
        typed_entities = {}
        for k, v in entities.items():
            mapped_type = type_map.get(k, k)
            typed_entities[mapped_type] = v
        return typed_entities

    def infer_relations(self, entities: dict) -> list:
        """
        Infer simple relations between entities (demo: link Persona->Azienda, Persona->Email, etc.).
        Returns a list of (source, relation, target).
        """
        relations = []
        # Example: Persona works at Azienda
        for person in entities.get("Persona", []):
            for org in entities.get("Azienda", []):
                relations.append((person, "lavora presso", org))
            for email in entities.get("Email", []):
                relations.append((person, "email", email))
            for phone in entities.get("Telefono", []):
                relations.append((person, "telefono", phone))
        # Example: Azienda has IBAN
        for org in entities.get("Azienda", []):
            for iban in entities.get("IBAN", []):
                relations.append((org, "iban", iban))
        # Example: Persona has Codice Fiscale
        for person in entities.get("Persona", []):
            for cf in entities.get("Codice Fiscale", []):
                relations.append((person, "codice fiscale", cf))
        # Example: Data is related to Persona or Azienda
        for date in entities.get("Data", []):
            for person in entities.get("Persona", []):
                relations.append((date, "data rilevante per", person))
            for org in entities.get("Azienda", []):
                relations.append((date, "data rilevante per", org))
        return relations

    def build_graph(self, entities: dict, relations: list) -> nx.DiGraph:
        """
        Build a directed graph from entities and relations.
        """
        G = nx.DiGraph()
        # Add nodes with type as attribute
        for ent_type, ent_list in entities.items():
            for ent in ent_list:
                G.add_node(ent, type=ent_type)
        # Add edges
        for src, rel, tgt in relations:
            G.add_edge(src, tgt, relation=rel)
        return G

    def abbreviate_label(self, label: str, ent_type: str) -> str:
        """
        Abbreviate node labels for better graph visualization.
        """
        if ent_type == "Persona":
            parts = label.split()
            if len(parts) > 1:
                return f"{parts[0]} {parts[1][0]}."
            return label
        elif ent_type == "Azienda":
            return label.split()[0].capitalize()
        elif ent_type == "Email":
            return label.split("@")[0] + "@..."
        elif ent_type == "Telefono":
            return f"+{label[-4:]}"
        elif ent_type == "IBAN":
            return label[:6] + "..." + label[-4:]
        elif ent_type == "Codice Fiscale":
            return label[:4] + "..." + label[-3:]
        elif ent_type == "Data":
            return label
        else:
            return label[:8] + ("..." if len(label) > 8 else "")

    def visualize_graph(self, G: nx.DiGraph):
        """
        Visualize the graph using matplotlib with maximum spacing and full labels.
        """
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            # Maximize spacing and add even more jitter
            pos = nx.spring_layout(G, k=20.0, iterations=1500, seed=42)
            for key in pos:
                jitter_x = random.uniform(-0.7, 0.7)
                jitter_y = random.uniform(-0.7, 0.7)
                pos[key] = (pos[key][0] + jitter_x, pos[key][1] + jitter_y)

        type_colors = {
            "Persona": "#8dd3c7",
            "Azienda": "#ffffb3",
            "Luogo": "#bebada",
            "Data": "#fb8072",
            "IBAN": "#80b1d3",
            "Codice Fiscale": "#fdb462",
            "Email": "#b3de69",
            "Telefono": "#fccde5",
            "Altro": "#d9d9d9"
        }
        node_colors = []
        for n in G.nodes:
            t = G.nodes[n].get("type", "Altro")
            node_colors.append(type_colors.get(t, "#d9d9d9"))

        plt.figure(figsize=(28, 20))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3400, alpha=0.95, linewidths=2, edgecolors="#333")
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", font_color="#222")
        nx.draw_networkx_edges(G, pos, edge_color="#888", arrows=True, arrowsize=30, width=2, connectionstyle='arc3,rad=0.22')
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='#d62728', font_size=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        plt.title('Entity Graph', fontsize=24)
        plt.axis('off')
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=typ, markerfacecolor=col, markersize=15) for typ, col in type_colors.items()]
        plt.legend(handles=handles, loc='lower left', bbox_to_anchor=(1, 0.1), title='Tipi Entità')
        plt.tight_layout()
        plt.show()
        # Print mapping for clarity
        print("\nLegenda nodi (etichetta -> valore completo):")
        for n in G.nodes:
            t = G.nodes[n].get("type", "Altro")
            print(f"[{n}] ({t}): {n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NER entities from a text file, normalize, build graph, and output JSON.")
    parser.add_argument("document_path", nargs="?", default="document.txt", help="Path to the input document.txt file")
    parser.add_argument("--show-graph", action="store_true", help="Show the entity graph visually")
    args = parser.parse_args()
    extractor = NERExtractor()
    entities = extractor.extract_from_file(args.document_path)
    cleaned = extractor.clean_and_normalize_entities(entities)
    typed = extractor.assign_entity_types(cleaned)
    relations = extractor.infer_relations(typed)
    G = extractor.build_graph(typed, relations)
    print(json.dumps({"entities": typed, "relations": relations}, ensure_ascii=False, indent=2))
    extractor.visualize_graph(G)
