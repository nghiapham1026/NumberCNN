document.getElementById('upload-form').onsubmit = async function(e) {
    e.preventDefault();
    
    // Initialize FormData object
    let formData = new FormData();
    
    // Check if the user has uploaded a file
    if (e.target.image_file.files.length > 0) {
        formData.append('image', e.target.image_file.files[0]);
    } else {
        // If no file is uploaded, send the image URL
        formData.append('image_url', e.target.image_url.value);
    }

    let response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });

    let prediction = await response.text();
    document.getElementById('prediction').innerText = 'Prediction: ' + prediction;
};
