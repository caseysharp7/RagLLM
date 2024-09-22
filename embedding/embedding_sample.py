from langchain_openai import OpenAIEmbeddings
model = OpenAIEmbeddings(api_key = "secret ;)")

embeddings = model.embed_documents(
    [
        "Sentence to be embedded",
        "numba 2",
        "the third sentence to be embedded",
        "Hello World?^%&$, test , test...",
        "Indeed"
    ]
)

print(len(embeddings), len(embeddings[0]))
