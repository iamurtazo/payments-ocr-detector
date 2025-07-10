# Payment Screenshot OCR Detector

A Python application that uses OCR (Optical Character Recognition) to detect and analyze payment/transfer completion screenshots. The app can identify key phrases like "Transfer Complete", currency symbols, and determine whether a screenshot should be marked for deletion.

## Features

- **OCR Text Detection**: Extracts text from screenshots using Tesseract OCR
- **Payment Status Detection**: Identifies successful/failed payments and transfers
- **Currency Recognition**: Detects various currency symbols and codes (USD, EUR, KRW, etc.)
- **Smart Analysis**: Uses pattern matching to identify payment-related content
- **Batch Processing**: Can process multiple images in a directory
- **Deletion Recommendations**: Suggests which screenshots can be safely deleted

## Installation

### Prerequisites

1. **Install Tesseract OCR**:
   - **Windows**: Download from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt