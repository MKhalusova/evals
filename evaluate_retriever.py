from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import utils as chromautils
from langchain.embeddings import HuggingFaceEmbeddings
from unstructured.staging.base import dict_to_elements
import pandas as pd
import os
from unstructured.staging.base import elements_from_json
from unstructured.staging.base import elements_to_dicts
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Retrieve the results from a retriever')
    parser.add_argument('--n_documents_to_retrieve', type=int, default=10, help='Number of documents to retrieve')
    parser.add_argument('--model_name', type=str, default="BAAI/bge-large-en-v1.5",
                        help='Name of the embedding model to use')
    parser.add_argument('--documents', type=str, default="local-ingest-output",
                        help='Location of the preprocessed documents to build a retriever with')
    parser.add_argument('--qa_dataset', type=str, default="qa_pairs_dataset.csv", help='Location of the eval dataset')

    return parser.parse_args()


def load_processed_files(directory_path):
    """
    Reads all JSON files in the given directory and returns elements as a list

    Args:
        directory_path (str): The path to the directory containing JSON files.
    """
    elements = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                elements.extend(elements_to_dicts(elements_from_json(filename=file_path)))
            except IOError:
                print(f"Error: Could not read file {filename}.")

    return elements


def prepare_questions(path_to_eval_dataset):
    df = pd.read_csv(path_to_eval_dataset)
    df = df[df['question'].notna()]
    return df["question"].to_list()


def prepare_docs(path_to_processed_files):
    elements = load_processed_files(path_to_processed_files)
    staged_elements = dict_to_elements(elements)

    documents = []

    for element in staged_elements:
        metadata = element.metadata.to_dict()
        metadata['element_id'] = element._element_id
        del metadata['orig_elements']
        documents.append(Document(page_content=element.text, metadata=metadata))

    return chromautils.filter_complex_metadata(documents)


def setup_retriever(docs, embedding_model_name, k):
    db = Chroma.from_documents(docs, HuggingFaceEmbeddings(model_name=embedding_model_name))
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})


def collect_retrieval_results(questions, retriever, output_path):
    results = []
    for question in questions:
        try:
            retrieved_documents = retriever.invoke(question)
            retrieved_ids = [doc.metadata['element_id'] for doc in retrieved_documents]
            results.append({"question": question, "retrieved_ids": retrieved_ids})
        except:
            print(f"Skipped question: {question}")

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()

    docs = prepare_docs(args.documents)
    retriever = setup_retriever(docs, args.model_name, args.n_documents_to_retrieve)

    questions = prepare_questions(args.qa_dataset)

    collect_retrieval_results(questions, retriever,
                              f"retriever_results/{args.model_name.replace('/', '@')}-{args.n_documents_to_retrieve}.csv")
