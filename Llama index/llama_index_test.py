import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core import load_index_from_storage,StorageContext
load_dotenv()


PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    documents = SimpleDirectoryReader(".\data")

    loader = documents.load_data()

    print(loader)

    vector_store = VectorStoreIndex.from_documents(loader,show_progress=True)

    vector_store.storage_context.persist(persist_dir=PERSIST_DIR)

else:

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)

    vector_store = load_index_from_storage(storage_context)

    print(vector_store)



# FIRST KIND OF RETRIEVER AND QUERY ENGINE

# query_engine = vector_store.as_query_engine()

# SECOND KIND OF RETRIEVER
retriever = VectorIndexRetriever(index=vector_store,similarity_top_k=4)

# FOR RESULTS CUT-OFF
post_processor = SimilarityPostprocessor(similarity_cutoff=0.80)

# QUERY ENGINE FOR SECOND KIND OF RETRIEVER
query_engine = RetrieverQueryEngine(retriever=retriever,node_postprocessors=[post_processor])

response = query_engine.query("what is the Health Insurance Coverage Status?")

pprint_response(response,show_source=True)


