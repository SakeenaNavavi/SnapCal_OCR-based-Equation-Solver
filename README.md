# Equation Solver with Image Recognition

This project is a web application that solves mathematical equations from images using optical character recognition (OCR) and symbolic mathematics. It's built with Flask, OpenCV, and SymPy.

## Features

- Upload images containing mathematical equations
- Process images to extract equation text using OCR
- Parse and solve equations with step-by-step explanations
- Handle various mathematical operations including basic arithmetic, algebraic expressions, and logarithms
- Display results in a user-friendly web interface

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- pip (Python package manager)
- Tesseract OCR engine installed on your system

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/equation-solver.git
   cd equation-solver
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload an image containing a mathematical equation

4. The application will process the image, extract the equation, solve it, and display the results with step-by-step explanations

## Project Structure

- `app.py`: Main Flask application file
- `templates/`: Contains HTML templates for the web interface
- `static/`: Contains static files (CSS, JavaScript, images)
- `uploads/`: Temporary storage for uploaded images

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a pull request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [SymPy](https://www.sympy.org/)
