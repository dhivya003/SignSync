function uploadVideo() {
    let file = document.getElementById("uploadVideo").files[0];
    let formData = new FormData();
    formData.append("video", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("prediction").innerText = data.prediction;
    })
    .catch(error => console.error("Error:", error));
}
