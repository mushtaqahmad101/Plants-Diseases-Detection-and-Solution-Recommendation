import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models 
from flask import Flask, request, render_template
from PIL import Image

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ResNet50 model architecture
def get_resnet50_model(num_classes):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4[-1].parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes))
    return model

#  AlexNet model architecture
def get_alexnet_model(num_classes):
    model = models.alexnet(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.classifier.add_module('dropout', nn.Dropout(0.5))
    return model

# Load ResNet50 model
num_classes = 12
model1 = get_resnet50_model(num_classes)
model1.load_state_dict(torch.load("model/improved_resnet50_model.pth"))
model1.eval()

# Load AlexNet model
model2 = get_alexnet_model(num_classes)
model2.load_state_dict(torch.load("model/alexnet_model.pth"))
model2.eval()

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Labels
disease_labels = [
    "Corn Common Rust", "Corn Healthy", "Corn Northern Leaf Blight", 
    "Potato Early Blight", "Potato Healthy", "Potato Late Blight", 
    "Rice Healthy", "Rice Leaf Blast", "Rice Neck Blast", 
    "Wheat Brown Rust", "Wheat Healthy", "Wheat Yellow Rust"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        file = request.files.get('image')
        selected_model = request.form.get('model')

        if not file or file.filename == '':
            return "No selected file"
        if not selected_model:
            return "No model selected"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Run inference
        prediction = predict(filepath, selected_model)
        return render_template(
            'index.html',
            image_path=filepath,
            prediction=prediction,
            selected_model=selected_model,
            disease=prediction  # Pass the predicted disease to the template
        )
    return render_template('index.html', image_path=None, prediction=None, selected_model=None, disease=None)

def predict(image_path, model_name):
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Predict with the selected model
    model = model1 if model_name == "ResNet50" else model2
    with torch.no_grad():
        output = model(image)
    _, pred = torch.max(output, 1)
    return disease_labels[pred.item()]

@app.route('/get_solution', methods=['POST'])
def get_solution():
    disease = request.form.get('disease')
    
    # Solutions for different diseases with bold tags included
    solutions = {
    "Corn Common Rust": "Preventions for Corn Common Rust Disease:\nPlant Resistant Varieties: Choose rust-resistant corn hybrids to reduce susceptibility to infection.\nTimely Fungicide Application: Apply fungicides such as propiconazole or tebuconazole at early stages of rust development to control the spread.\nProper Crop Rotation: Rotate corn with non-host crops like soybeans to break the rust's life cycle and reduce infection risk.\nField Sanitation: Remove and destroy infected plant debris after harvest to reduce sources of infection in the next planting season.\n\nFertilizer Solution for Disease:\nUrea (46-0-0): A nitrogen-based fertilizer that promotes healthy growth and boosts the plant’s overall vigor, making it more resistant to diseases.\nTriple Superphosphate (0-46-0): A phosphorus-based fertilizer that encourages strong root development and improves plant health, helping corn better resist environmental stresses and infections.",
    "Corn Healthy": "No Worry your crop is healthy, Welldone keep it up",
    "Corn Northern Leaf Blight": "Preventions for Corn Northern Leaf Blight:\nPlant Resistant Varieties: Use hybrids that are resistant to Northern Leaf Blight to minimize infection risk.\nTimely Fungicide Application: Apply fungicides like propiconazole or azoxystrobin during high-risk periods (e.g., wet conditions).\n\nFertilizer Solutions for Corn:\nBalanced Nitrogen Fertilization (e.g., Urea): Ensure proper nitrogen levels to promote healthy growth and disease resistance.\nPotassium Fertilizer (e.g., Potash): Use potassium to strengthen the plant’s defenses and improve its overall stress resistance",
    "Potato Early Blight": "Preventions for Potato Early Blight (Alternaria solani):\nUse Resistant Varieties: Plant potato varieties that are resistant to early blight to reduce disease severity.\nFungicide Application:\nApply fungicides like chlorothalonil or maneb at regular intervals to prevent early blight infection, especially in humid conditions.\n\nSolution for Potato Early Blight:\nProper Crop Rotation:\nRotate potatoes with non-solanaceous crops (e.g., beans or corn) to break the disease cycle.\nGood Field Sanitation: Remove and destroy infected plant debris to reduce the spread of the pathogen to new crops",
    "Potato Healthy": "No Worry your crop is healthy, Welldone keep it up",
    "Potato Late Blight": "Preventions for Potato Late Blight (Phytophthora infestans):\nUse Resistant Varieties:\nPlant late-blight-resistant potato varieties to minimize susceptibility to the disease.\nFungicide Application: Apply fungicides like mancozeb or chlorothalonil as a preventive measure, especially during wet and humid conditions.\n\nSolutions for Potato Late Blight:\nCrop Rotation:\nRotate potatoes with non-solanaceous crops (e.g., wheat or legumes) to disrupt the pathogen's life cycle.\nRemove Infected Plants:\nPromptly remove and destroy infected plants and tubers to prevent the spread of late blight spores",
    "Rice Healthy": "No Worry your crop is healthy, Welldone keep it up",
    "Rice Leaf Blast": "Prevention for Rice Leaf Blast (Magnaporthe oryzae):\nPlant Resistant Varieties: Use rice varieties resistant to leaf blast to minimize infection risk.\nMaintain Proper Spacing: Avoid dense planting to improve air circulation and reduce humidity, which inhibits fungal growth.\n\nSolutions for Rice Leaf Blast:\nFungicide Application: Apply fungicides like tricyclazole or carbendazim at the early stages of infection for effective control.\nBalanced Fertilization: Avoid excessive nitrogen application, as it promotes lush growth susceptible to blast; ensure balanced use of potassium and phosphorus fertilizers to enhance plant resistance",
    "Rice Neck Blast": "Prevention for Rice Neck Blast (Magnaporthe oryzae):\nPlant Resistant Varieties: Use rice varieties bred for resistance to neck blast to reduce susceptibility.\nOptimize Planting Practices: Maintain proper plant spacing and water management to reduce humidity, which is conducive to fungal growth.\n\nSolutions for Rice Neck Blast:\nFungicide Application: Apply systemic fungicides such as tricyclazole or isoprothiolane at the panicle initiation stage for effective control.\nBalanced Fertilization: Avoid excessive nitrogen fertilization and ensure adequate potassium to strengthen plant defenses against neck blast",
    "Wheat Brown Rust": "Prevention for Wheat Brown Rust (Puccinia triticina):\nUse Resistant Varieties: Plant wheat varieties resistant to brown rust to minimize infection risks.\nEarly Sowing: Adjust sowing dates to avoid environmental conditions favorable to rust development, such as cool and moist weather during crop maturity.\n\nSolutions for Wheat Brown Rust:\nFungicide Application: Apply fungicides like propiconazole or tebuconazole at the early stages of rust appearance for effective control.\nCrop Rotation: Rotate wheat with non-host crops to disrupt the life cycle of the rust pathogen.",
    "Wheat Healthy": "No worry your crop is healthy, Welldone keep it up",
    "Wheat Yellow Rust": "Prevention Solutions for Wheat Brown Rust (Puccinia triticina):\nUse Resistant Varieties: Plant wheat varieties resistant to brown rust to minimize infection risks.\nEarly Sowing: Adjust sowing dates to avoid environmental conditions favorable to rust development, such as cool and moist weather during crop maturity.\n\nSolutions for Wheat Brown Rust:\nFungicide Application: Apply fungicides like propiconazole or tebuconazole at the early stages of rust appearance for effective control.\nCrop Rotation: Rotate wheat with non-host crops to disrupt the life cycle of the rust pathogen."
}


    # Fetch the solution from the dictionary
    solution = solutions.get(disease, "Solution not found")
    
    # Return the solution as a JSON response
    return {'solution': solution}


if __name__ == "__main__":
    app.run(debug=True)
