const imageForm = document.getElementById('image-form');
const imageInput = document.getElementById('image-input');
const resultDiv = document.getElementById('result');

// Load the pre-trained MobileNet model
const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');

// Utility function to convert image to tensor
async function imageToTensor(image) {
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224]) // Change the size to match your model's input size
        .toFloat()
        .expandDims();

    return tensor;
}

// Function to classify the image
async function classifyImage(image) {
    const tensor = await imageToTensor(image);
    const input = tensor.reshape([1, 224, 224, 3]);

    // Make prediction through the model on our image.
    const predictions = await model.predict(input).data();

    // Get the index of the highest score.
    const maxPrediction = Math.max(...predictions);
    const maxIndex = predictions.indexOf(maxPrediction);

    // Load the labels for the top 3 predictions
    const labels = await (await fetch('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/labels.json')).json();
    const top3Predictions = Array.from(predictions)
        .slice(0, 3)
        .map((prediction, index) => {
            return {
                label: labels[index],
                probability: prediction
            };
        })
        .sort((a, b) => b.probability - a.probability);

    // Display the result
    resultDiv.innerHTML = `
        <p>Prediction: ${top3Predictions[0].label} (${top3Predictions[0].probability.toFixed(3)})</p>
        <p>Prediction: ${top3Predictions[1].label} (${top3Predictions[1].probability.toFixed(3)})</p>
        <p>Prediction: ${top3Predictions[2].label} (${top3Predictions[2].probability.toFixed(3)})</p>
    `;
}

// Handle form submission
imageForm.addEventListener('submit', async (event) => {
    event.preventDefault