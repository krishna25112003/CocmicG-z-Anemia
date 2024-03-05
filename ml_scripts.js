// Load your TensorFlow.js anemia detection model
const anemiaModel = await tf.loadLayersModel('my_cnn_model.json');

// Function to preprocess the eye image
function preprocessEyeImage(image) {
    // Perform any necessary preprocessing (resize, normalization, etc.)
    // Return a tensor suitable for the model
    // Example: resize the image to 224x224 pixels
    const tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).toFloat();
    const mean = tf.tensor1d([0.485, 0.456, 0.406]);
    const std = tf.tensor1d([0.229, 0.224, 0.225]);
    const normalized = tensor.div(tf.scalar(255)).sub(mean).div(std).expandDims();
    return normalized;
}

// Function to make anemia predictions
async function predictAnemia(eyeImage) {
    const preprocessedImage = preprocessEyeImage(eyeImage);
    const prediction = anemiaModel.predict(preprocessedImage);
    const result = await prediction.data();
    return result;
}

// Function to handle eye image upload and anemia detection
async function detectAnemia() {
    const eyeImageInput = document.getElementById('eyeImageInput');
    const uploadedEyeImage = document.getElementById('uploadedEyeImage');
    const anemiaResultElement = document.getElementById('anemiaResult');

    // Check if an image is selected
    if (eyeImageInput.files.length > 0) {
        const eyeImageFile = eyeImageInput.files[0];

        // Display the uploaded eye image
        const eyeImageUrl = URL.createObjectURL(eyeImageFile);
        uploadedEyeImage.src = eyeImageUrl;

        // Detect anemia
        const eyeImage = new Image();
        eyeImage.src = eyeImageUrl;
        eyeImage.onload = async () => {
            const result = await predictAnemia(eyeImage);

            // Display the anemia detection result
            const isAnemic = result[0] > 0.5; // Assuming a binary classification
            const resultText = isAnemic ? 'Anemia Detected' : 'No Anemia Detected';
            anemiaResultElement.innerText = `Anemia Detection Result: ${resultText}`;
        };
    } else {
        alert('Please select an eye image.');
    }
}
