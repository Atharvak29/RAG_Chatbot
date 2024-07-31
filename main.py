# import google.generativeai as genai
# #import absl.logging
# import streamlit as st 


# genai.configure(api_key='AIzaSyAwX3BIGRHzjUdNy5oTX3egWG97H42MKvw')

# def main():
#     # st.image(r"C:\Users\athud\Downloads\data.png", use_column_width = False, width = 100)
#     st.header("Chat with PaLM")
#     st.write(" ")

#     prompt = st.text_input("Prompt please....", placeholder = 'Prompt', label_visibility='visible')
#     temp = 0.0 #st.slider("Temperature", 0.0 , 1.0 , step = 0.05 )

#     if st.button("SEND", use_container_width = True ):
#         model = 'models/text-bison-001'

#         response = genai.generate_text(
#             model = model,
#             prompt = prompt,
#             temperature = temp ,
#             max_output_tokens= 1024
#         )

#         st.write(" ")
#         st.header(':blue[Response]')
#         st.write(" ")

#         st.markdown(response.result, unsafe_allow_html= False, help = None)

# if __name__ == '__main__':
#     main()


#---------------------------

# def get_vector():
#     embeddings=GooglePalmEmbeddings()
#     pc = Pinecone(api_key='3308a05e-e4ea-4453-affe-04d4b98d6bcb')

#     index_name = "rag-chatbot"
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name, 
#             dimension=768, 
#             metric='cosine',
#             spec=ServerlessSpec(
#                 cloud='aws',
#                 region='us-west-1'
#             )
#         )

#     # Connect to the Pinecone index
#     index = pc.Index(index_name)


# try:
#     # Create a new conversation
#     response = genai.chat(messages='Hello')

#     # Get the model's response
#     print("Model's response:", response.last)

#     response = response.reply(message = "Have a good morning")
#     print("Model's response:", response.last)
          
# except Exception as e:
#     print("An error occurred:", e)

    

# import openai

# # Set your OpenAI API key
# openai.api_key = 'sk-proj-IrkXpQsxStmtyHRnEg0hT3BlbkFJ72aZFfb0MurohvVTncRB'

# def chat_with_gpt(prompt):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]
#         )

#         return response.choices[0].message['content']
#     except Exception as e:
#         return f"An error occurred: {str(e)}"

# # Main loop
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['quit', 'exit', 'bye']:
#         print("Goodbye!")
#         break
    
#     response = chat_with_gpt(user_input)
#     print("Assistant:", response)



# ------------------------------------------------------------------------------------------------------------------

import google.generativeai as genai
import os
import streamlit as st
from langchain.vectorstores import Pinecone as pc
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import GooglePalmEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone

# Configuration for Google PaLM API
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAwX3BIGRHzjUdNy5oTX3egWG97H42MKvw'
genai.configure(api_key='AIzaSyAwX3BIGRHzjUdNy5oTX3egWG97H42MKvw')

# Initialize Pinecone
os.environ['PINECONE_API_KEY'] = '3308a05e-e4ea-4453-affe-04d4b98d6bcb'
# PINECONE_API_ENV = os.environ.get('3308a05e-e4ea-4453-affe-04d4b98d6bcb')
# pc = Pinecone(api_key='3308a05e-e4ea-4453-affe-04d4b98d6bcb')  # Add your Pinecone API key and environment

# Check if the index exists, if not, create it

index_name = "rag-chatbot"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name, 
#         dimension=768, 
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-west-1'
#         )
#     )

# Connect to the Pinecone index
# index = pc.Index(index_name)

embeddings = GooglePalmEmbeddings()  # Using PaLM for embeddings
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Function to query Pinecone and generate a response
def generate_response(prompt):
    # Query the Pinecone index
    docs = docsearch.similarity_search(prompt, k=3)
    
    # Set up Google Palm for response generation
    llm = GooglePalm(temperature=0.1)
    
    # Use RetrievalQA to combine the LLM with the retrieved documents
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    
    # Run the query and return the response
    response = qa.run(prompt)
    return response

def main():
    st.header("Chat with PaLM + Pinecone")
    st.write(" ")

    prompt = st.text_input("Prompt please....", placeholder='Enter your question here', label_visibility='visible')

    if st.button("SEND", use_container_width=True):
        # Generate the response using the integrated Pinecone + PaLM system
        response = generate_response(prompt)

        st.write(" ")
        st.header(':blue[Response]')
        st.write(" ")

        st.markdown(response, unsafe_allow_html=False)

if __name__ == '__main__':
    main()