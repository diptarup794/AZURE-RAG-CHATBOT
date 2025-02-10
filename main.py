import os
import json
import openai
import tiktoken
from datetime import datetime
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    IndexingSchedule,
    SearchIndexerSkillset,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    WebApiSkill,
    SearchIndexer
)
from azure.search.documents.models import Vector
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document URL Mappings (pre-uploaded documents)
DOCUMENT_URLS = {
    "employee_handbook.docx": "https://company.com/docs/employee-handbook",
    "product_manual.docx": "https://company.com/docs/product-manual",
    "technical_specs.docx": "https://company.com/docs/technical-specs",
}

class DocumentUploader:
    def __init__(self, connection_string, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        self.container_client = self.blob_service_client.get_container_client(container_name)
        
    def upload_documents(self):
        """Upload documents to Azure Blob Storage."""
        uploaded_files = []
        try:
            # Ensure the container exists
            try:
                self.container_client.create_container()
                print(f"Container '{self.container_name}' created.")
            except Exception as e:
                print(f"Container '{self.container_name}' already exists or could not be created: {str(e)}")
            
            # Upload files to blob storage
            for filename, filepath in DOCUMENT_URLS.items():
                try:
                    blob_client = self.container_client.get_blob_client(filename)
                    with open(filepath, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    uploaded_files.append(filename)
                    print(f"Uploaded {filename} to blob storage.")
                except Exception as e:
                    print(f"Failed to upload {filename}: {str(e)}")
            return uploaded_files
        
        except Exception as e:
            print(f"Error during document upload: {str(e)}")
            return None


class SearchIndexerSetup:
    def __init__(self, service_name, admin_key, index_name):
        self.endpoint = f"https://{service_name}.search.windows.net/"
        self.credential = AzureKeyCredential(admin_key)
        self.index_name = index_name
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        
    def create_index(self):
        """Create search index with vector search capabilities if it does not already exist."""
        try:
            # Check if the index already exists
            try:
                existing_index = self.index_client.get_index(self.index_name)
                print(f"Index '{self.index_name}' already exists.")
                return True
            except Exception as e:
                print(f"Index '{self.index_name}' does not exist. Proceeding to create it.")
            
            # Define vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    VectorSearchAlgorithmConfiguration(
                        name="vector-config",
                        kind="hnsw",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ]
            )
            
            # Define index fields based on the structure in the provided table
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, retrievable=True),
                SimpleField(name="content", type=SearchFieldDataType.String, retrievable=True),
                SimpleField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=1536,
                    vector_search_configuration="vector-config"
                ),
                SimpleField(name="source_url", type=SearchFieldDataType.String, retrievable=True),
                SimpleField(name="source_file", type=SearchFieldDataType.String, retrievable=True),
                SimpleField(name="metadata_storage_name", type=SearchFieldDataType.String, retrievable=True)
            ]
            
            # Create the index
            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            self.index_client.create_or_update_index(index)
            print(f"Index '{self.index_name}' created successfully.")
            return True
        
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False
    
    def setup_indexing_pipeline(self):
        """Setup complete indexing pipeline with data source and indexer"""
        try:
            data_source = SearchIndexerDataSourceConnection(
                name=f"{self.index_name}-datasource",
                type="azureblob",
                connection_string=AZURE_STORAGE_CONNECTION_STRING,
                container=SearchIndexerDataContainer(name=BLOB_CONTAINER_NAME)
            )
            self.index_client.create_or_update_data_source_connection(data_source)

            skillset = SearchIndexerSkillset(
                name=f"{self.index_name}-skillset",
                skills=[]
            )
            self.index_client.create_or_update_skillset(skillset)

            indexer = SearchIndexer(
                name=f"{self.index_name}-indexer",
                data_source_name=data_source.name,
                target_index_name=self.index_name,
                skillset_name=skillset.name,
                schedule=IndexingSchedule(interval="PT5M"),
                field_mappings=[
                    {"sourceFieldName": "metadata_storage_path", "targetFieldName": "source_url"},
                    {"sourceFieldName": "metadata_storage_name", "targetFieldName": "source_file"}
                ]
            )
            self.index_client.create_or_update_indexer(indexer)
            
            return True
        
        except Exception as e:
            print(f"Error setting up indexing pipeline: {str(e)}")
            return False

class ChatBot:
    def __init__(self, search_client):
        self.search_client = search_client
    
    def get_embeddings(self, text):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response["data"][0]["embedding"]
    
    def generate_response(self, query):
        try:
            query_embedding = self.get_embeddings(query)
            
            results = self.search_client.search(
                search_text=query,
                select=["content", "source_url", "source_file"],
                top=3,
                vector=Vector(value=query_embedding, k=3, fields="content_vector")
            )
            
            context = "\n".join([result["content"] for result in results])
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                "answer": response.choices[0].message["content"],
                "sources": [{"source_url": result["source_url"], "source_file": result["source_file"]} for result in results]
            }
        
        except Exception as e:
            return {"error": str(e)}

def setup_search_service():
    try:
        # Initialize search setup
        search_setup = SearchIndexerSetup(AZURE_SEARCH_SERVICE, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX)
        
        # Create or validate index
        if not search_setup.create_index():
            raise Exception("Failed to create or validate index")
        
        # Setup indexing pipeline
        if not search_setup.setup_indexing_pipeline():
            raise Exception("Failed to setup indexing pipeline")
        
        search_client = SearchClient(
            endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/",
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        
        return search_client
    
    except Exception as e:
        print(f"Setup error: {str(e)}")
        return None

app = Flask(__name__)
openai.api_key = OPENAI_API_KEY

search_client = setup_search_service()
if search_client:
    chatbot = ChatBot(search_client)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        response = chatbot.generate_response(query)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
