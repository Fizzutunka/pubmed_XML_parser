import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from langchain_neo4j import Neo4jGraph


# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# used to embedd the query text
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Define retrieval query 
retrieval_query = """
MATCH (a:Article)
OPTIONAL MATCH (auth:Author)-[:WROTE]->(a)
RETURN 
  a.title AS title,
  a.pmid AS pmid,
  a.pub_date AS published_date
  LIMIT 25
"""
# Create retriever that automatically creates cypher quiries. Requires neo4j_graphrag.retrievers import Text2CypherRetriever
# retriever = Text2CypherRetriever(
#     driver=driver,
#     llm=llm,
# )
# Create retriever

from neo4j_graphrag.retrievers import VectorCypherRetriever
retriever = VectorCypherRetriever(
    driver,
    index_name="articleAbstracts",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

#  Create the LLM
llm = OpenAILLM(model_name="gpt-4o-mini")

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# example search
query_text = "Find a paper discussing DNA."

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 3},
    return_context=True
)

print(response.answer)
# # Optionally print raw retrieved items
# print("\n--- Context Items ---")
# for item in response.retriever_result.items:
#     print(item.content)

# Close the database connection
driver.close()