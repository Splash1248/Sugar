document.getElementById("uploadForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("videoInput");
  const resultDiv = document.getElementById("result");

  if (!fileInput.files[0]) {
    resultDiv.textContent = "❌ Please upload a video file.";
    return;
  }

  const formData = new FormData();
  formData.append("video", fileInput.files[0]);

  resultDiv.textContent = "⏳ Processing...";

  fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.glucose_level) {
        resultDiv.textContent = `🩸 Predicted Blood Glucose Level: ${data.glucose_level.toFixed(2)} mg/dL`;
      } else {
        resultDiv.textContent = "❌ Prediction failed.";
      }
    })
    .catch((err) => {
      resultDiv.textContent = "❌ Error during prediction.";
      console.error(err);
    });
});
