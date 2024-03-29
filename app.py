#import libraries
import os
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader, download_loader, set_global_service_context
from llama_index.core import Settings
from flask import Flask,request,app,jsonify,url_for,render_template 


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/ragchatbot_api',methods=['POST'])

def ragchatbot_api():
    data = request.json['data']
    print(data)
    new_data = data["Prompt"]
    print(new_data)
    #new_data = base_new_data.values()


    # Set API keys and set Gemini as llm
    GOOGLE_API_KEY = "AIzaSyDEvLkqFWDGcNEdfej5nGtGk_gqELwini4"
    PINECONE_API_KEY = "953c33e9-4c8e-4c61-868a-64b246640ef4"

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    # set llm as Gemini Pro
    llm = Gemini()


    #Create a Pinecone client
    #To send data back and forth between the app and Pinecone, we'll need to instantiate a Pinecone client.

    #pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


    #Select the Pinecone index
    #Using our Pinecone client, we can select the Index that we previously created and assign it to the variable pinecone_index:
    pinecone_index = pinecone_client.Index("testindex")


    #Call the documents
    documents = SimpleDirectoryReader("data").load_data()

    #Generate embeddings using GeminiEmbedding
    embed_model = GeminiEmbedding(model_name="models/embedding-001")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    #Generate and store embeddings in the Pinecone index
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Create a StorageContext using the created PineconeVectorStore
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    #Using the VectorStoreIndex class, LlamaIndex takes care of sending the data chunks to the embedding model 
    #and then handles storing the vectorized data into the Pinecone index.
    #store embeddings in pinecone index

    # Use the chunks of documents and the storage_context to create the index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )


    #Query Pinecone vector store
    #Now the contents of the pdf are converted to embeddings and stored in the Pinecone index.
    #Let's perform a similarity search by querying the index
    # query pinecone index for similar embeddings
    query_engine = index.as_query_engine()

    gemini_response = query_engine.query(new_data)

    # print response
    return gemini_response

if __name__ == "__main__":
    app.run(debug=True)