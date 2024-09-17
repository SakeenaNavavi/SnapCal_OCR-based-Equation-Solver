# Import necessary libraries
from flask import Flask, request, render_template, url_for
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from werkzeug.utils import secure_filename
from sympy import sympify, solve, Symbol, Eq, simplify, expand
import re
import logging

# Initialize Flask application
app = Flask(__name__, static_folder='static')

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    logging.warning('image received')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold

# Function to extract text from the preprocessed image
def extract_text(preprocessed_image):
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/()=^√'
    text = pytesseract.image_to_string(Image.fromarray(preprocessed_image), config=custom_config)
    return text

# Function to parse and solve the equation with step-by-step explanation
def parse_and_solve_equation_with_steps(equation_text):
    # Clean up the equation text
    equation_text = equation_text.replace(' ', '').replace('\n', '')
    
    # Function to add multiplication symbols where implied
    def add_mult_symbols(eq):
        return re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', eq)
    
    # Try to parse as a standard equation (with equals sign)
    match = re.search(r'(.+)=(.+)', equation_text)
    if match:
        left_side = add_mult_symbols(match.group(1))
        right_side = add_mult_symbols(match.group(2))
    else:
        # If no equals sign, assume it's set to zero
        left_side = add_mult_symbols(equation_text)
        right_side = '0'
    
    steps = []
    
    try:
        # Find the main variable in the equation
        variables = list(set(re.findall(r'[a-zA-Z]', equation_text)))
        main_var = variables[0] if variables else 'x'  # Default to 'x' if no variable found
        
        # Create symbols for all variables in the equation
        symbol_dict = {sym: Symbol(sym) for sym in variables}
        
        # Parse the equation
        left_expr = sympify(left_side, locals=symbol_dict)
        right_expr = sympify(right_side, locals=symbol_dict)
        equation = Eq(left_expr, right_expr)
        steps.append(f"Step 1: Understand the original equation\nWe start with: {left_side} = {right_side}")
        
        # Move all terms to the left side
        equation = Eq(left_expr - right_expr, 0)
        steps.append(f"Step 2: Move all terms to the left side of the equation\nWe get: {left_side} - ({right_side}) = 0")
        
        # Expand the equation
        expanded_eq = expand(equation.lhs)
        equation = Eq(expanded_eq, 0)
        steps.append(f"Step 3: Expand the equation\nAfter expanding, we have: {expanded_eq} = 0")
        
        # Simplify the equation
        simplified_eq = simplify(equation.lhs)
        equation = Eq(simplified_eq, 0)
        steps.append(f"Step 4: Simplify the equation\nSimplifying gives us: {simplified_eq} = 0")
        
        # Solve the equation
        solution = solve(equation, symbol_dict[main_var])
        
        # Format the solution
        if isinstance(solution, list):
            if len(solution) == 1:
                formatted_solution = f"{main_var} = {solution[0]}"
                steps.append(f"Step 5: Solve the equation\nThe solution is: {formatted_solution}")
            else:
                formatted_solution = ', '.join(f"{main_var} = {sol}" for sol in solution)
                steps.append(f"Step 5: Solve the equation\nThe equation has multiple solutions:\n{formatted_solution}")
        elif isinstance(solution, dict):
            formatted_solution = ', '.join(f"{k} = {v}" for k, v in solution.items())
            steps.append(f"Step 5: Solve the equation\nThe solution is: {formatted_solution}")
        else:
            formatted_solution = f"{main_var} = {solution}"
            steps.append(f"Step 5: Solve the equation\nThe solution is: {formatted_solution}")
        
        # Rearrange the solution to match the original equation format
        if len(solution) == 1 and not isinstance(solution[0], (dict, tuple)):
            final_solution = f"{main_var} = {solution[0]}"
        else:
            final_solution = formatted_solution
        
        return {
            'steps': steps,
            'solution': final_solution
        }
    except Exception as e:
        return {
            'steps': steps,
            'error': f"Error solving equation: {str(e)}"
        }

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and solve the equation
            preprocessed = preprocess_image(filepath)
            equation_text = extract_text(preprocessed)
            result = parse_and_solve_equation_with_steps(equation_text)
            
            # Render the result
            return render_template('index.html', 
                                   equation=equation_text, 
                                   steps=result.get('steps', []),
                                   solution=result.get('solution', ''),
                                   error=result.get('error', ''))
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)