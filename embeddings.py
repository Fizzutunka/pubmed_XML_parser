from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

#neo4j credentials
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), 
                              auth=(os.getenv("NEO4J_USERNAME"), 
                                    os.getenv("NEO4J_PASSWORD"))) 

# test connection to the database
def test_connection(tx):
    result = tx.run("RETURN 'Connection successful' AS message")
    for record in result:
        print(record["message"])

with driver.session() as session:
    session.execute_write(test_connection)

# # retrieving the abstracts from the database. NOT IN USE. neo4j_vector does this code itself within-node
# def get_abstracts(tx):
#     query = """
#     MATCH (a:Article)
#     WHERE a.abstract IS NOT NULL
#     RETURN a.pmid AS pmid, a.abstract AS abstract
#     """
#     result = tx.run(query)
#     return [
#         Document(page_content=record["abstract"], metadata={"pmid": record["pmid"]})
#         for record in result
#     ]

# with driver.session() as session:
#     documents = session.execute_read(get_abstracts)
# print(f"Retrieved {len(documents)} abstracts for embedding.")

# No Chunking method as abstracts are typically small enough to be processed in one go.
# Although, abstracts may be too small for meaningfull embedding model and comparisons

# creating vector embeddings for every pmid: abstract pair  
 # Embed and index into Neo4j vector store
embedder = OpenAIEmbeddings(model="text-embedding-ada-002") 

neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embedder,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    node_label="Article",  # Target the existing nodes called 'article'
    text_node_properties=["abstract"], # Where the text to embedd is stored
    embedding_node_property = "abstract_vectors"    # new property for embeddings
)

print("Abstracts embedded and saved within Article Neo4j nodes.")    


from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from neo4j_graphrag.retrievers import VectorRetriever

with driver.session() as session:
    session.execute_write(test_connection)

# Run similarity search
results = neo4j_vector.similarity_search_with_relevance_scores(query = "antibiotic resistance in hospital-acquired infections",
                                                               k=5)

# Print results
for doc, score in results:
    print(f"\nScore: {score:.4f}")
    print(doc.page_content[:300])  # print first 500 chars
    print("-" * 50)
