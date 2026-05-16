def evaluate_rag_system(retriever, generator, test_dataset):
    """
    Evaluates a RAG system using a given retriever and generator against a test dataset.

    Args:
        retriever: A function or object that takes a query and returns a list of relevant documents.
        generator: A function or object that takes a query and retrieved documents to generate an answer.
        test_dataset: A list of dictionaries, where each dictionary contains 'query', 'ground_truth_answer',
                      and 'ground_truth_docs' (optional) keys.

    Returns:
        A dictionary containing evaluation metrics.
    """
    retrieval_accuracies = []
    generation_accuracies = []
    relevances = [] # Placeholder for more sophisticated relevance metrics

    for item in test_dataset:
        query = item['query']
        ground_truth_answer = item['ground_truth_answer']
        ground_truth_docs = set(item.get('ground_truth_docs', [])) # Use set for efficient comparison

        # Step 1: Retrieval Evaluation
        retrieved_docs = retriever(query)
        retrieved_doc_texts = {doc for doc in retrieved_docs} # Assuming retrieved_docs are strings or have a text attribute

        # Simple check for any overlap with ground truth documents
        if ground_truth_docs and retrieved_doc_texts.intersection(ground_truth_docs):
             retrieval_accuracies.append(1)
        elif ground_truth_docs: # If ground truth docs exist but no overlap
             retrieval_accuracies.append(0)
        # If no ground_truth_docs provided, this metric for the item isn't applicable

        # Step 2: Generation Evaluation
        generated_answer = generator(query, retrieved_docs)

        # Simple keyword matching for generation accuracy
        if all(keyword.lower() in generated_answer.lower() for keyword in ground_truth_answer.lower().split()):
            generation_accuracies.append(1)
        else:
            generation_accuracies.append(0)

        # In a real scenario, you'd use more advanced methods like ROUGE, BLEU, BERTScore or human evaluation
        # For relevance, you could assess if retrieved_docs contain information relevant to ground_truth_answer

    avg_retrieval_accuracy = sum(retrieval_accuracies) / len(retrieval_accuracies) if retrieval_accuracies else 0
    avg_generation_accuracy = sum(generation_accuracies) / len(generation_accuracies) if generation_accuracies else 0

    return {
        "avg_retrieval_accuracy": avg_retrieval_accuracy,
        "avg_generation_accuracy": avg_generation_accuracy,
        # "avg_relevance": sum(relevances) / len(relevances) if relevances else 0,
    }

# Mock Retriever and Generator for demonstration
def mock_retriever(query):
    if "Python" in query:
        return ["Python is a high-level programming language.", "Python is widely used in AI."]
    elif "Generative AI" in query:
        return ["Generative AI creates new content.", "Large Language Models are a type of Generative AI."]
    return ["General programming concepts."]

def mock_generator(query, docs):
    if "Python" in query and "programming" in "".join(docs):
        return "Python is a popular programming language, especially for AI applications."
    elif "Generative AI" in query and "new content" in "".join(docs):
        return "Generative AI systems like LLMs can produce novel text, images, and other data."
    return "I cannot provide a specific answer based on the provided information."

# Sample Test Dataset
test_data = [
    {
        "query": "What is Python?",
        "ground_truth_answer": "Python is a programming language.",
        "ground_truth_docs": ["Python is a high-level programming language."]
    },
    {
        "query": "Explain Generative AI.",
        "ground_truth_answer": "Generative AI creates new content.",
        "ground_truth_docs": ["Generative AI creates new content."]
    },
    {
        "query": "Tell me about cars.",
        "ground_truth_answer": "Cars are a form of transport.",
        "ground_truth_docs": [] # No specific ground truth doc given for this query
    }
]

# Run the evaluation
if __name__ == "__main__":
    evaluation_results = evaluate_rag_system(mock_retriever, mock_generator, test_data)
    print("RAG System Evaluation Results:")
    print(f"  Average Retrieval Accuracy: {evaluation_results['avg_retrieval_accuracy']:.2f}")
    print(f"  Average Generation Accuracy: {evaluation_results['avg_generation_accuracy']:.2f}")
