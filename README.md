# Superhero Help Desk Application
This project provides an interactive chat-based help desk application for a fictional superhero company. It leverages the powerful language model, Cohere, and utilizes ChromaDB for query-based example storage and retrieval to provide accurate department classification and mood assessment of user messages.

## Features
- Real-time message classification to appropriate departments.
- Sentiment or mood analysis of user messages.
- Dynamic training using ChromaDB to improve classification accuracy.
- Fun and interactive user interface to keep the user engaged.

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/septiannugraha/chroma-cohere.git
```

2. Navigate into the project directory:

```bash
cd chroma-cohere
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory of the project and populate it with your Cohere API key:

```env
COHERE_API_KEY=<your-cohere-api-key>
COHERE_MODEL_NAME=<your-cohere-model-name>
```

## Usage

To start the application, run:

```bash
python main.py
```

Follow the on-screen instructions to interact with the help desk.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).