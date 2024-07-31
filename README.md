# RAG_Chatbot
Making of a Chatbot for indexing internal documents as a internship project

# Project Title: RAG Chatbot Using Pinecone

## Project Overview
Developed a Retrieval-Augmented Generation (RAG) chatbot application using Pinecone for efficient text retrieval and a language model for generating answers based on predefined PDFs.

## Objectives
- **Primary:** Create a chatbot that answers questions using content from predefined PDFs.
- **Secondary:** Ensure efficient storage and retrieval of text data using Pinecone.

## Methodology
- **Approach:** Combined Pinecone for data storage and retrieval with PaLM for generating responses.
- **Tools and Technologies:** Python, Streamlit, PyPDF, Pinecone, PaLM.
- **Algorithms/Models:** PaLM for natural language understanding and generation.

## Implementation
- **Development Process:** Extracted text, stored in Pinecone, created Streamlit UI, integrated with PaLM.
- **Code and Frameworks:** Used Pinecone's API, PyPDF for text extraction, Streamlit for UI.
- **Integration:** Ensured smooth interaction between UI, Pinecone, and PaLM.

## Results and Evaluation
- **Results:** Successfully answered user queries based on PDF content.
- **Evaluation Metrics:** User satisfaction, accuracy of answers.
- **Analysis:** High accuracy in retrieving relevant text chunks and generating appropriate responses.

## Challenges and Solutions
- **Challenges:** Handling large text volumes, ensuring fast retrieval.
- **Solutions:** Optimized text chunking, used Pinecone's efficient retrieval mechanisms.

## Conclusion
- **Summary:** Achieved goal of creating an efficient RAG chatbot.
- **Impact:** Improved user experience in querying large text datasets.

## Future Work
- **Improvements:** Enhance UI, add more PDFs, improve text chunking strategies.
- **Future Directions:** Explore other language models, extend to more domains.

## Documentation and References
- **Documentation:** Created a detailed README, code comments, and usage guide.
- **References:** Cited sources for PDFs, Pinecone, and PaLM documentation.


## How to run the code 
1. install the requirements.txt 
2. ```streamlit run main.py```
3. For answers from the PDF navigate to RAG_Chatbot > Documents > "Setting up Pinecone.ipynb"         
4. Open using Jupyter notebook, edit query varible to your question and run ```qa.run(query)```