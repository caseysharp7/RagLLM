Large Language Models (LLMs) are a well known example of AI that produce text based on data they have previously encountered; they still suffer when answering questions that 
require knowledge of local conditions, specific varieties, and up-to-date data. One way to address the domain knowledge issue is through Retrieval-Augmented Generation (RAG). 
RAG integrates an external retrieval system with a customized database into LLMs, enabling them to access and utilize relevant information on the fly.

This project is focused on creating a quality cuda database of linked optimized and unoptimized examples using JSON. The user will input cuda code to the LLM in hope that it 
will be optimized. Both the input code and the unoptimized examples will be embedded into vectors so they can then be compared. The most similar unoptimized examples will be 
selected, and their optimized counterparts will be retrieved (with FAISS) and fed to the LLM. Having optimized examples that perform a similar task to the user input code will 
allow the LLM to introduce optimizations, whether it be memory management or data transfer, into the user input code. The unoptimized code examples are given generic variable names when hand coded (i.e. var1, var2 ...) so when they are embedded the variable names have less impact on the comparison.

Much of this project has been experimental, so there is additional code in this repository, like my own transformer model, the code to run the LLM parallel between the two 
4090s I work on, and the code used to embed the examples.

This repository is a work in progress, so some of the cuda examples won't be present as they are still in development because I have to find quality optimized cuda and then
hand code it back to an unoptimized and inefficient state. This is with the assumption that the user input code will be in similar unoptimized states.

As the optimized examples are generally not of my own creation, I will explicitly state that I did not write that code and tell where it did come from, most of which come from
NVIDIA themselves. The unoptimized examples are for the most part my own code, but they are designed to do the same thing as the optimized examples, just less efficiently. This being said, for my own benefit I have spent likely a majority of the time on this project working to understand the optimized examples and how they map to the GPU hardware.
