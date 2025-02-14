# <strong>[Project Name]</strong> - Plants Disease Detection & Solution Recommendation

### <em>Overview</em>
The <strong>[Project Name]</strong> is an automated web application designed to detect plant diseases from images and recommend appropriate solutions. This multi-class classification model identifies specific diseases from a set of twelve categories. Once the disease is detected, the system provides a solution from a pre-defined database. The project is highly beneficial for early crop disease detection and for offering timely solutions to ensure the health of crops.

### <em>Features</em>
<ul>
    <li><strong>Disease Detection</strong>: Detects plant diseases from images using deep learning models.</li>
    <li><strong>Solution Recommendation</strong>: Provides specific solutions for the detected disease from a database.</li>
    <li><strong>Flask Web Application</strong>: Real-time interaction via a Flask web interface.</li>
    <li><strong>Model</strong>: Utilizes pre-trained AlexNet and ResNet50 models for accurate disease classification.</li>
</ul>

---

### <strong>Installation Instructions</strong>

1. <strong>Clone the Repository</strong><br>
   First, clone this repository to your local system using the following command:
   <pre><code>git clone https://github.com/[YOUR_USERNAME]/[YOUR_REPOSITORY_NAME].git</code></pre>
   Replace <em>[YOUR_USERNAME]</em> and <em>[YOUR_REPOSITORY_NAME]</em> with your actual GitHub username and repository name.

2. <strong>Install Dependencies</strong><br>
   After cloning the repository, navigate to the project directory and install the necessary dependencies by running the following command:
   <pre><code>pip install -r requirements.txt</code></pre>

---

### <strong>Running the Project Locally</strong>

1. <strong>Training the Model</strong><br>
   - Download the dataset and run the provided Jupyter Notebook to train the AlexNet and ResNet50 models. The models will be saved after training.
   - You can modify the number of epochs in the notebook as needed.

2. <strong>Run the Flask App</strong><br>
   After the models are trained, navigate to the project directory in the terminal and run the following command:
   <pre><code>python app.py</code></pre>
   This will start the Flask server, and the web application will be available in your browser.

3. <strong>Access the Web Application</strong><br>
   Open the provided Flask interface link in your browser. You can upload an image of a plant, and the system will return a prediction of the disease along with a recommended solution.

---

### <strong>Requirements</strong>

The Python dependencies required to run this project are listed in requirements.txt file 

You can install these dependencies via:
<pre><code>pip install -r requirements.txt</code></pre>

---

### <strong>Usage</strong>

1. <strong>Upload an image</strong>: The web application allows users to upload images of plant leaves to detect diseases.
2. <strong>View the prediction</strong>: The system will output the disease detected and recommend the most effective solution based on the provided database.

---

### <strong>License</strong>

Include information about your project’s license here, e.g., MIT, GPL, etc.

---

### <strong>Author</strong>

- <strong>[Your Name]</strong> - <a href="[Your GitHub Profile URL]">[Your GitHub Profile URL]</a>

---

### <strong>Notes</strong>

<ul>
    <li>Make sure you have a working internet connection when training the models or running the Flask app for the first time, as some dependencies need to be downloaded.</li>
    <li>This project is designed for <strong>educational</strong> and <strong>research purposes</strong> only.</li>
</ul>

---

