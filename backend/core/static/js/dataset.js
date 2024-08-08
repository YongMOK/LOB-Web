function preprocessDataset(datasetId) {
    const preprocessBtn = document.getElementById(`preprocessBtn-${datasetId}`);
    preprocessBtn.innerHTML = 'Processing...';
    preprocessBtn.disabled = true;
    if (confirm("Are you sure you want to preprocess this dataset?")) {
        fetch(`/datasets/${datasetId}/preprocess/`, {
            method: "POST",
            headers: {
                "X-CSRFToken": "{{ csrf_token }}"
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert(data.message);
                location.reload();
            } else {
                alert("Preprocessing failed.");
                preprocessBtn.innerHTML = 'Preprocess';
                preprocessBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred while preprocessing.");
            preprocessBtn.innerHTML = 'Preprocess';
            preprocessBtn.disabled = false;
        });
    } else {
        preprocessBtn.innerHTML = 'Preprocess';
        preprocessBtn.disabled = false;
    }
}