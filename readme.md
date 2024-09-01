# Generative AI Based Data Analyzer

Welcome to the **Generative AI Based Data Analyzer** application! This tool leverages advanced language models to help you analyze and visualize your data in a user-friendly environment. With support for different large language models (LLMs) and the ability to process both CSV and Excel files, this application is designed to make data exploration and analysis easier and more insightful.

## Features

- **Interactive Data Analysis**: Upload your CSV or Excel files and ask questions about your data in natural language.
- **Multiple LLMs Supported**: Choose from various models such as Gemma-7b-IT, Llama3–70b-8192, Llama3–8b-8192, and Mixtral-8x7b-32768.
- **Dynamic Graph Generation**: Automatically generate graphs based on the analysis results when needed.
- **Session History**: Keep track of your queries and responses for reference.

## Getting Started

Follow these steps to get the application up and running:

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/Surajrs812/Generative-AI-Based-Data-Analyzer.git
cd Generative-AI-Based-Data-Analyzer
```

### 2. Add Groq API Key

Create a `.env` file in the root directory of the project and add your Groq API key:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install Dependencies

Make sure you have all necessary dependencies installed. Run the following command:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

To start the application, use the following command:

```bash
streamlit run app.py
```

### 5. Using the Application

1. **Select LLM Model**: Choose your preferred LLM model from the sidebar.
2. **Upload Data**: Upload your CSV or Excel file.
3. **Ask Questions**: Type your queries in the input box, and the model will analyze your data and provide responses. If the response requires a graph, it will be generated and displayed automatically.
4. **View History**: Previous queries and responses are stored in the session history for your reference.

## Example Queries

Here are some example queries you can try:

- "Compare the sales data for the first and second quarters."
- "What is the average price of products in the dataset?"
- "Generate a bar chart showing the distribution of product categories."

## Troubleshooting

- Ensure that the correct Groq API key is added in the `.env` file.
- Make sure all dependencies are installed correctly.
- If the application crashes, check the terminal for error messages and ensure that your data files are correctly formatted.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. We welcome all contributions, including bug fixes, feature enhancements, and documentation improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or suggestions, please open an issue in the GitHub repository or contact the maintainer directly.

