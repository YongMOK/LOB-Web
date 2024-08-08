function setDatasetId(datasetId, marketName) {
    document.getElementById("dataset-id").value = datasetId;
    document.getElementById("market-name").value = marketName;
    fetchModelNames(marketName, 'existing_model');
}

function setDatasetIdForPrediction(datasetId, marketName) {
    document.getElementById("predict-dataset-id").value = datasetId;
    document.getElementById("predict-market-name").value = marketName;
    fetchModelNames(marketName, 'predict_model_name');
}

function fetchModelNames(marketName, elementId) {
    fetch(`/model_recommendations/${marketName}/`)
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById(elementId);
            select.innerHTML = '<option value="">Select Model</option>';
            data.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.text = name;
                select.appendChild(option);
            });
        })
        .catch(error => {
            console.error("Error fetching model names:", error);
        });
}

document.getElementById('newModelCheck').addEventListener('change', function () {
    const newModelGroup = document.getElementById('newModelGroup');
    if (this.checked) {
        newModelGroup.style.display = 'block';
    } else {
        newModelGroup.style.display = 'none';
    }
});

document.addEventListener('DOMContentLoaded', function () {
    const trainModelForm = document.getElementById('train-model-form');
    if (trainModelForm) {
        trainModelForm.addEventListener('submit', function (event) {
            event.preventDefault();
            const form = event.target;
            const formData = new FormData(trainModelForm);
            const trainButton = form.querySelector('button[type="submit"]');
            trainButton.innerHTML = 'Training...';
            trainButton.disabled = true;
            
            function trainModel(forceRetrain = false) {
                if (forceRetrain) {
                    formData.append('force_retrain', true);
                }

                fetch('{% url "modelparameter-train" %}', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': formData.get('csrfmiddlewaretoken')
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.result) {
                        alert('Success: ' + data.result.status);
                        location.reload();
                    } else if (data.already_trained) {
                        if (confirm(data.message + " Do you want to retrain it?")) {
                            trainModel(true);
                        } else {
                            trainButton.innerHTML = 'Train Model';
                            trainButton.disabled = false;
                        }
                    } else {
                        alert('Unexpected response: ' + JSON.stringify(data));
                        trainButton.innerHTML = 'Train Model';
                        trainButton.disabled = false;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while training the model.');
                    trainButton.innerHTML = 'Train Model';
                    trainButton.disabled = false;
                });
            }
            
            trainModel();
        });
    }
});


document.getElementById("predict-model-form").addEventListener("submit", function(event){
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const predictBtn = form.querySelector('button[type="submit"]');
    predictBtn.innerHTML = 'Predicting...';
    predictBtn.disabled = true;

    fetch("{% url 'prediction-predict' %}",{
        method: "POST",
        headers: {
            "X-CSRFToken": formData.get('csrfmiddlewaretoken')
        },
        body: formData,
    }).then(response => response.json())
    .then(data =>{
        if(data.result){
            const report = data.result.classification_report;
            const matrix = data.result.confusion_matrix;
            document.getElementById("classificationReportTable").innerHTML = generateReportTable(report);
            document.getElementById("confusionMatrixTable").innerHTML = generateMatrixTable(matrix);

            const resultModal = new boostrap.Modal(document.getElementById("resultModal"));
            resultModal.show();

        }else{
            alert(data.message || "Failed to predict.");
            predictBtn.innerHTML = 'Predict';
            predictBtn.disabled = false;
        }
    }).catch(error =>{
        console.error("Error: ", error);
        alert("An error occurred while predicting.");
        predictBtn.innerHTML = 'Predict';
        predictBtn.disabled = false;
    });

    // Function to generate classification report table
    function generateReportTable(report){
        console.log("processing ...............1");
        let table = "<table class='table table-bordered'>";
        table += "<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>";
        table += "<tbody>";
        console.log("processing ...............2");
        for(let i in report){
            console.log("processing ...............3");
            table += "<tr>";
            table += "<td>" + i + "</td>";
            table += "<td>" + report[i].precision.toFixed(2) + "</td>";
            table += "<td>" + report[i].recall.toFixed(2) + "</td>";
            table += "<td>" + report[i].f1_score.toFixed(2) + "</td>";
            table += "</tr>";
        }
        console.log("processing ...............4");
        table += "</tbody></table>";
        return table;
    }
    
    // Function to generate confusion matrix table
    function generateMatrixTable(matrix){
        let table = '<table class="table table-bordered"><thead><tr>';
        for (let i = 0; i < matrix.length; i++) {
            table += `<th>Class ${i + 1}</th>`;
        }
        table += '</tr></thead><tbody>';
        matrix.forEach(row => {
            table += '<tr>';
            row.forEach(cell => {
                table += `<td>${cell}</td>`;
            });
            table += '</tr>';
        });
        table += '</tbody></table>';
        return table;
    }
    // Close the modal if the user clicks outside of it
    window.onclick = function(event) {
        const modal = document.getElementById('myModal');
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }
});