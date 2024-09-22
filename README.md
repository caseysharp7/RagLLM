Large Language Models (LLMs) are a well known example of AI that produce text based on data they have previously encountered; they still suffer when answering questions that 
require knowledge of local conditions, specific varieties, and up-to-date data. One way to address the domain knowledge issue is through Retrieval-Augmented Generation (RAG). 
RAG integrates an external retrieval system with a customized database into LLMs, enabling them to access and utilize relevant information on the fly.

This project is focused on creating a quality cuda database of linked optimized and unoptimized examples using JSON. The user will input cuda code to the LLM in hope that it 
will be optimized. Both the input code and the unoptimized examples will be embedded into vectors so they can then be compared. The most similar unoptimized examples will be 
selected, and their optimized counterparts will be retrieved (with FAISS) and fed to the LLM. Having optimized examples that perform a similar task to the user input code will 
allow the LLM to introduce optimizations, whether it be memory management or data transfer, into the user input code. 

Much of this project has been experimental, so there is additional code in this repository, like my own transformer model, the code to run the LLM parallel between the two 
4090s I work on, and the code used to embed the examples.
