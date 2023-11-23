# -*- coding: utf-8 -*-

import os
import openai
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
# from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import TokenTextSplitter

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Helper function for printing docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

class MerlinBot:
    def __init__(self, config=None):
        # Initialize Pinecone
        self.config = config
        if self.config is None:
            self.config["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
            self.config["PINECONE_API_KEY"] = os.environ["PINECONE_API_KEY"]
            self.config["PINECONE_API_ENV"] = os.environ["PINECONE_API_ENV"]
            self.config["PINECONE_INDEX_NAME"] = self.config["PINECONE_INDEX_NAME"]
        pinecone.init(
            api_key=self.config["PINECONE_API_KEY"],
            environment=self.config["PINECONE_API_ENV"]
        )
        
    def perform_query(self, query):

        index=pinecone.Index('gen-ai-hack') 
        # get api key from platform.openai.com
        openai.api_key = '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'

        embed_model = "text-embedding-ada-002"


        self.query = query

        res = openai.Embedding.create(
            input=[query],
            engine=embed_model
            )
        
        # retrieve from Pinecone
        xq = res['data'][0]['embedding']


        # get relevant contexts (including the questions)
        res = index.query(xq, top_k=4, include_metadata=True)
        
        contexts = [str(item['metadata']) for item in res['matches']]
        
        augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+"1."+query+ " along with confidence score in percentage. Confidence score is how much probable that the answer you have given is correct."
        #"\n\n-----\n\n"+"2."+"Give confidence score in percentage. Confidence score is how much probable that the answer you have given is correct."
        
        augmented_query2 = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+"3. What are the distinct file names from where the final answer is fetched? Do not mention this sentence in output but just give the file names."

        # system message to 'prime' the model
        primer = f"""You are Q&A bot. A highly intelligent system that answers
        user questions based on the information provided by the user above each question in requested format. 
        If the information can not be found in the information provided by the user you truthfully say "I don't know".
        """
        
        '''
        primer = f""" You are Q&A bot. A highly intelligent system that answers user questions based on the 
        information provided by the user above each question in requested format.Please thoroughly understand the entire question, after understanding the question,
        answer it as truthfully as possible using the provided context. If the answer is not contained within
        the text below, say 'I don't know.' Additionally, include the following metadata with your response: Source,
        Page number and a confidence score from 1 to 100, with 100 indicating the highest confidence in your answer"""
        '''
        res1 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query}
                            ]
        )
        res2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query2}
            ]
        )
        
        a1=res1['choices'][0]['message']['content']
        a2=res2['choices'][0]['message']['content']
        ans = a1+"\n\n"+a2
    
        return ans
