import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
import openai
import dotenv
dotenv.load_dotenv()
# openai.api_key = dotenv.get_key("OPENAI_API_KEY")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    response = openai.chat.completions.create(
        model="gpt-4-turbo",  # Chat model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    response_text = response.choices[0].message.content


    # Extract the actual chunks along with document IDs
    sources = [
        f"**Document ID:** {doc.metadata.get('id', 'Unknown').replace('data\\', '')}  \n**Chunk:** {doc.page_content}"
        for doc, _score in results
    ]

    # Format the response using Markdown
    formatted_sources = "\n\n".join(sources)
    formatted_response = f"{response_text}\n\n\n**Sources:**\n\n{formatted_sources}"
    print(formatted_response)

    # Return the response text and the actual chunks with document IDs
    return response_text, sources, formatted_response



if __name__ == "__main__":
    main()
