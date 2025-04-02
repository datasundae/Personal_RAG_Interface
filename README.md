# Personal RAG Interface

A modular RAG (Retrieval-Augmented Generation) system designed for personal document management, with a focus on maintaining clean separation of concerns and modularity.

## Features

- **Modular Architecture**: Clean separation of database, processing, and web interface components
- **Vector Database**: PostgreSQL with pgvector for efficient similarity search
- **Document Processing**: Support for multiple document types (PDF, images, text)
- **Security**: Encrypted storage for sensitive data
- **Web Interface**: Flask-based interface for document management

## Project Structure

```
Personal_RAG_Interface/
├── src/
│   ├── database/      # Database operations
│   ├── processing/    # Document processing
│   ├── config/        # Configuration
│   └── web/          # Web interface
├── docs/             # Documentation
├── tests/            # Test files
└── setup.py          # Package setup
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/datasundae/Personal_RAG_Interface.git
cd Personal_RAG_Interface
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up the database:
```bash
# Configure PostgreSQL with pgvector extension
# Update config.py with your database credentials
```

## Usage

1. Start the web interface:
```bash
python -m src.web.app
```

2. Access the interface at `http://localhost:5000`

## Documentation

- [Project Context](docs/project_context.md)
- [Project Rules](docs/project_rules.md)
- [Changelog](docs/CHANGELOG.md)

## Development

- Follow the project rules in `docs/project_rules.md`
- Write tests for new features
- Update documentation as needed
- Use type hints and docstrings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- Report security issues to [security contact]
- Follow security guidelines in project rules
- Keep dependencies updated 