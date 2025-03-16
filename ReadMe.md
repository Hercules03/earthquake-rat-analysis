# Japan Earthquake Analysis with RAT (Retrieval Augmented Thinking)

This project implements an advanced AI chatbot system for analyzing Japan's earthquake data using RAGLight's Retrieval Augmented Thinking (RAT) capabilities. The system ingests research papers, datasets, and code repositories to provide expert-level responses on earthquake prediction, analysis, and related phenomena.

## Project Overview

The Japan Earthquake Analysis project aims to:

1. Create an expert chatbot system for earthquake analysis
2. Process and analyze data related to Japan's earthquakes
3. Leverage machine learning techniques for earthquake prediction
4. Provide comprehensive analysis of factors affecting seismic activity

This implementation uses RAGLight's RAT approach, which enhances traditional Retrieval Augmented Generation (RAG) with iterative reasoning loops for more accurate and thoughtful responses.

## Components

- **Earthquake RAT Chatbot**: The core implementation that uses both local documents and GitHub repositories as knowledge sources
- **Text-Only Version**: A simplified implementation that uses text files to avoid PDF processing issues
- **Testing Scripts**: Tools to verify RAGLight functionality with minimal setup
- **Example Scripts**: Demonstrations of how to use the chatbot for earthquake analysis

## Installation

### Prerequisites

- Python 3.11 or higher
- Ollama (for running local LLMs)
- Required models in Ollama:
  - llama3 or llama3.2:3b-instruct-q4_K_M
  - deepseek-r1:7b

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Hercules03/earthquake-rat-analysis.git
cd earthquake-rat-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional system dependencies for PDF processing (if needed):
```bash
# On macOS with Homebrew
brew install poppler tesseract

# On Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr
```

## Usage

### Using the Text-Only Version (Recommended for Initial Testing)

The text-only version works without PDF processing dependencies and uses sample text files:

```bash
python run_text_only_chatbot.py
```

This script:
- Creates sample text files about earthquakes
- Builds a RAT pipeline
- Runs sample questions
- Starts an interactive session

### Using the Full Implementation with Research Papers

1. Place your research papers and documents in the `knowledge_base` folder
2. Run the example script:
```bash
python earthquake_analysis_examples.py
```

### Testing RAGLight Setup

To verify RAGLight is working correctly with your setup:

```bash
python test_rat_api.py
```

## Implementation Details

### Knowledge Sources

The system uses multiple knowledge sources:

1. **Local Documents**: Research papers, articles, and datasets in the `knowledge_base` folder
2. **GitHub Repositories**: Code repositories related to earthquake analysis and prediction:
   - Earthquake seismology repositories
   - Japan earthquake locators
   - QuakeMigrate
   - Stanford Earthquake Dataset (STEAD)
   - Machine learning models for earthquake prediction

### RAT Pipeline

The RAT implementation uses:

1. **Knowledge Retrieval**: Fetching relevant documents based on query
2. **Reasoning Iterations**: Using a reasoning model (DeepSeek) to think through the query
3. **Multiple Reflections**: Iteratively refining the reasoning
4. **Final Response Generation**: Creating a comprehensive answer based on the refined reasoning

## Project Structure

```
earthquake-rat-analysis/
├── knowledge_base/            # Folder for research papers and documents
├── text_knowledge_base/       # Folder for text files (text-only version)
├── earthquake_rat_chatbot.py  # Main RAT chatbot implementation
├── earthquake_analysis_examples.py # Examples of chatbot usage
├── text_only_earthquake_chatbot.py # Text-only implementation
├── run_text_only_chatbot.py   # Script to run text-only version
├── test_rat_api.py            # Test script for RAGLight
└── requirements.txt           # Project dependencies
```

## Future Enhancements

Planned improvements for this project:

1. Integration with real-time earthquake data sources
2. Visualization capabilities for seismic data
3. Fine-tuning reasoning models for earthquake-specific analysis
4. Deployment options for wider accessibility
5. Integration with GIS systems for spatial analysis

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**:
   - Ensure you've installed `unstructured[pdf]` and related dependencies

2. **Model Availability**:
   - Check that required models are available in Ollama with `ollama list`

3. **Memory Issues**:
   - For large knowledge bases, ensure sufficient RAM is available
   - Reduce the number of GitHub repositories or documents if needed

4. **Slow Response Times**:
   - Reduce the `reflection` parameter (e.g., from 3 to 1)
   - Use smaller or faster models if available

## License

[MIT License](LICENSE)

## Acknowledgments

- RAGLight library for providing the RAT implementation
- Authors of referenced research papers on earthquake prediction
- Maintainers of earthquake data repositories used as knowledge sources
