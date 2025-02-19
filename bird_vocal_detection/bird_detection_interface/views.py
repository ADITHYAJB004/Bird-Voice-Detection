from django.shortcuts import render
from .forms import UploadFileForm
import os
from .prediction_models.deep_learning.predict import deep_learning_prediction
from .prediction_models.naive_bayes.predict import process_naive_bayes
from .prediction_models.svm.predict import process_svm_bayes

# Create your views here.
def process_bird_voice(request):
    result = None  # Initialize result to None to handle any errors
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)

        # Process the form only if it is valid
        if form.is_valid():
            # Get the model selected and the uploaded file
            selected_model = request.POST.get('algo')  # 'algo' corresponds to the dropdown selection
            uploaded_file = request.FILES['file']
            
            # Save the file to a specified location
            file_path = os.path.join('./media/', uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Call the appropriate model based on selection
            if selected_model == 'Deep Learning Model' and uploaded_file.name.endswith('.mp3'):
                # Call the deep learning model prediction
                result = predict_bird_species(selected_model, file_path)
            elif selected_model in ['SVM', 'Naive Bayes'] and uploaded_file.name.endswith('.flac'):
                # Call the SVM or Naive Bayes model prediction
                result = predict_bird_species(selected_model, file_path)
            else:
                # If file extension does not match the model selected, return an error
                result = {"predicted_class": "Invalid file type", "species_info": "Please upload a .mp3 file for Deep Learning Model or .flac for SVM/Naive Bayes."}

        else:
            result = {"predicted_class": "Form Submission Error", "species_info": "There was an issue with your form submission."}

        # After processing, pass the result to the result page
        return render(request, 'result.html', {'result': result})

    else:
        form = UploadFileForm()

    # Render the input page on GET request
    return render(request, 'bird_detection.html', {'form': form})

def predict_bird_species(model_type, filename):
    """
    Function to simulate prediction based on the selected model.
    This should be replaced with actual model processing logic.
    """
    if model_type == 'Deep Learning Model':
        # Replace with your actual deep learning model logic
        result = deep_learning_prediction(filename)  # this should return a dictionary with 'predicted_class' and 'species_info'
    elif model_type == 'SVM':
        # Replace with your actual SVM model logic
        result = process_svm_bayes(filename)  # this should return a dictionary with 'predicted_class' and 'species_info'
    elif model_type == 'Naive Bayes':
        # Replace with your actual Naive Bayes model logic
        result = process_naive_bayes(filename)  # this should return a dictionary with 'predicted_class' and 'species_info'
    
    return result
