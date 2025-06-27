from neo4j import GraphDatabase
import pprint
import types
import pandas as pd
import ast  # Safer than eval

class Neo4jConnector:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="123456789"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def sanitize_value(self, val):
        if isinstance(val, (str, int, float, type(None))):
            return val
        elif isinstance(val, types.ModuleType):
            return f"[Module:{val.__name__}]"
        else:
            return str(val)

    def insert_ifc_element(self, tx, element_type, name, rt60=None, spl=None):
        tx.run("""
            MERGE (e:IFCElement {name: $name})
            SET e.type = $element_type, e.rt60 = $rt60, e.spl = $spl
        """, name=name, element_type=element_type, rt60=rt60, spl=spl)

    def insert_all(self, elements):
        with self.driver.session() as session:
            for elem in elements:
                sanitized_elem = {
                    "element_type": self.sanitize_value(elem.get("element_type")),
                    "name": self.sanitize_value(elem.get("name")),
                    "RT60": self.sanitize_value(elem.get("RT60")),
                    "SPL": self.sanitize_value(elem.get("SPL"))
                }
                pprint.pprint(sanitized_elem)
                session.write_transaction(
                    self.insert_ifc_element,
                    sanitized_elem["element_type"],
                    sanitized_elem["name"],
                    sanitized_elem["RT60"],
                    sanitized_elem["SPL"]
                )

    def upload_from_csv(self, node_path, edge_path):
        node_df = pd.read_csv(node_path)
        edge_df = pd.read_csv(edge_path)

        with self.driver.session() as session:
            for _, row in node_df.iterrows():
                session.run(
                    """
                    MERGE (n:IfcSpace {id: $id})
                    SET n += $props
                    """,
                    id=row["GlobalId"],
                    props=row.to_dict()
                )

            for _, row in edge_df.iterrows():
                relation_type = row['relation_type']
                attributes = ast.literal_eval(row['attributes'])
                cypher = f"""
                    MATCH (a:IFCElement {{id: $source}})
                    MATCH (b:IFCElement {{id: $target}})
                    MERGE (a)-[r:{relation_type}]->(b)
                    SET r += $props
                """
                session.run(
                    cypher,
                    source=row['source'],
                    target=row['target'],
                    props=attributes
                )

    def insert_from_csv(self, nodes_csv, edges_csv):
        nodes_df = pd.read_csv(nodes_csv)
        edges_df = pd.read_csv(edges_csv)

        with self.driver.session() as session:
            # Insert nodes
            for _, row in nodes_df.iterrows():
                session.run(
                    """
                    MERGE (n:IFCElement {id: $id})
                    SET n += $props
                    """,
                    id=row['GlobalId'],
                    props=row.drop('GlobalId').to_dict()
                )

            # Insert edges
            for _, row in edges_df.iterrows():
                relation_type = row['relation_type']
                attributes = ast.literal_eval(row['attributes'])
                cypher = f"""
                    MATCH (a:IFCElement {{id: $source}})
                    MATCH (b:IFCElement {{id: $target}})
                    MERGE (a)-[r:{relation_type}]->(b)
                    SET r += $props
                """
                session.run(
                    cypher,
                    source=row['source'],
                    target=row['target'],
                    props=attributes
                )