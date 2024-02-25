document.getElementById('upload-form').onsubmit = async function(e) {
    e.preventDefault();
    let formData = new FormData();
    formData.append('image', e.target.image.files[0]);

    let response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });

    let prediction = await response.text();
    document.getElementById('prediction').innerText = 'Prediction: ' + prediction;
};
