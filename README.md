# Document Verification System

A blockchain-based document verification system that uses AI and computer vision to verify the authenticity of documents.

## Features

- Document upload and registration on blockchain
- AI-powered document analysis
- QR code verification
- Security feature detection
- Real-time verification status
- Modern and responsive UI

## Tech Stack

- Backend: Python Flask
- Frontend: HTML, TailwindCSS
- Blockchain: Ethereum (Sepolia Testnet)
- AI: Google Gemini AI
- Computer Vision: OpenCV

## Prerequisites

- Python 3.12+
- Node.js and npm (for TailwindCSS)
- Ethereum wallet and Sepolia testnet ETH
- Google Cloud API key for Gemini AI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RohitKesavJ/Document-Verification-System.git
cd Document-Verification-System
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add:
```
GOOGLE_API_KEY=your_gemini_api_key
```

4. Initialize the database:
```bash
python app.py
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Use the application:
   - Click "Register Document" to upload and register a new document
   - Click "Verify Document" to verify an existing document

## Project Structure

```
Document-Verification-System/
├── app.py                 # Main Flask application
├── static/               # Static files
│   ├── uploads/         # Uploaded documents
│   └── tailwind.min.css # TailwindCSS styles
├── templates/           # HTML templates
│   ├── index.html      # Home page
│   ├── verify.html     # Verification page
│   └── result.html     # Results page
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Rohit Kesav J - [GitHub](https://github.com/RohitKesavJ)
