# RAG (Retrieval-Augmented Generation) System for Lifecycle Assistance

This GitHub repository contains a Retrieval-Augmented Generation (RAG) system designed for lifecycle assistance. The project aims to combine the power of large language models (LLMs) with specific and contextual information to generate more accurate and relevant responses.

## Project Structure

The repository is organized into two main directories:

- `Preprocessing and model creation`: Contains scripts and notebooks for data preprocessing and model creation.
- `Sketch Generation`: Contains code related to generating sketches or Arduino code, potentially based on retrieved information.

## Preprocessing and Model Creation

The `Preprocessing and model creation` folder includes:

- `Preprocessing_notebook.ipynb`: A Jupyter notebook that details the data preprocessing steps. It uses `ContextBuilder` to load and prepare data, and `DataPreprocessor` to suggest tasks and generate code.
- `context_builder.py`: A Python script that manages context building from CSV and JSON files. It is responsible for reading data, defining task and feature descriptions, and generating a structured context for the RAG model.
- `data_preprocessor.py`: This file contains the logic for data preprocessing and code generation based on the provided context.
- `llm_helper.py`: Assists with interaction with language models, including initializing and using models like Llama3-70b-8192.
- `retriever_instance.py`: Manages the retrieval of relevant information from a Chroma DB database using cosine similarity.

## Sketch Generation

The `Sketch Generation` folder contains:

- `main_code.ipynb`: A Jupyter notebook that appears to be the main entry point for sketch generation. It uses `langchain_groq` to interact with LLMs and generate Arduino code. It also includes functions to compile and upload code to an Arduino board.
- `prompt_template.py`: Contains prompt templates used to guide the LLM in generating code or text.

## Key Features

- **Retrieval-Augmented Generation (RAG)**: Combines LLMs with a knowledge base for more accurate responses.
- **Data Preprocessing**: Tools to prepare contextual data and task descriptions.
- **Code Generation**: Ability to generate code (especially Arduino) based on queries and contexts.
- **LLM Integration**: Use of models like Llama3-70b-8192 via `langchain_groq`.
- **Context Retrieval**: Use of Chroma DB to retrieve relevant documents.




## Contribution

To contribute to this project, please follow standard GitHub guidelines: fork the repository, create a branch, make your changes, and submit a pull request.

## License

Mohamed Ali Msadek & Mohamed Amine Khediri
EURECOM - France
