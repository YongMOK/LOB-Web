document.addEventListener('DOMContentLoaded', function() {
    const marketSelect = document.getElementById('market');
    const selectAllCheckbox = document.getElementById('select-all');
    const checkboxes = document.querySelectorAll('.model-checkbox');
    const trainButton = document.querySelector('#train-model-form button[type="submit"]'); // Reference to the Train button
    
    let selectedMarket = null;
    let selectedModels = [];
    let datasets = [];
    let bestK = null;
    let finalResults = {}; // Dictionary to store the final results for each model

    // Weights for each metric
    const weights = {
        accuracy: 0.4,
        precision: 0.2,
        recall: 0.2,
        f1Score: 0.2
    };

    // Event listener for market selection
    if (marketSelect) {
        marketSelect.addEventListener('change', function() {
            selectedMarket = marketSelect.value;
            fetchDatasets(selectedMarket)
                .then(fetchedDatasets => {
                    datasets = fetchedDatasets;
                });
        });
    }

    // Event listener for select all checkbox
    if (selectAllCheckbox) {
        selectAllCheckbox.onclick = function() {
            selectedModels = [];
            checkboxes.forEach(checkbox => {
                checkbox.checked = selectAllCheckbox.checked;
                if (checkbox.checked) {
                    selectedModels.push(checkbox.value);
                }
            });
        };
    }

    // Event listeners for individual model checkboxes
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (checkbox.checked) {
                selectedModels.push(checkbox.value);
            } else {
                selectedModels = selectedModels.filter(model => model !== checkbox.value);
            }
        });
    });

    // Function to fetch datasets for the selected market
    const fetchDatasets = async (market) => {
        const url = `/get_datasets/${market}/`;
        try {
            const response = await fetch(url);
            const data = await response.json();
            if (data.error) {
                alert(`Error: ${data.error}`);
                return [];
            }
            return data.datasets;
        } catch (error) {
            console.error("Error fetching datasets: ", error);
            return [];
        }
    };

    // Event listener for form submission
    const trainForm = document.getElementById('train-model-form');
    if (trainForm) {
        trainForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const formData = new FormData(this);

            selectedMarket = document.getElementById('market').value;
            const kMin = document.getElementById('k-min').value;
            const kMax = document.getElementById('k-max').value;

            formData.append('market', selectedMarket);
            selectedModels.forEach(model => formData.append('models', model));
            formData.append('k-min', kMin);
            formData.append('k-max', kMax);

            datasets = await fetchDatasets(selectedMarket);

            document.getElementById('training-steps').style.display = 'block';
            document.getElementById('step1').style.display = 'block';
            trainButton.style.display = 'none';
            findBestK(formData);
        });
    }

    // Function to find the best K
    const findBestK = (formData) => {
        const url = `/find_best_k/${selectedMarket}/`;
        fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }
            bestK = data.k;
            document.getElementById('choosing-k').style.display = 'none';
            document.getElementById('best-k').innerText = bestK;
            document.getElementById('k-value').style.display = 'block';
            var histogramContent = document.getElementById('histogram');
            histogramContent.innerHTML = data.histogram_html;

            var scriptTags = histogramContent.getElementsByTagName('script');
            for (var i = 0; i < scriptTags.length; i++) {
                eval(scriptTags[i].innerText);
            }

            document.getElementById('step2').style.display = 'block';
            processAllDatasets(datasets);
        })
        .catch(error => {
            alert(`Unexpected error: ${error}`);
            console.error('There was an error with the fetch operation:', error);
        });
    };

    // Function to preprocess a single dataset
    const preprocessDataset = (dataset) => {
        const url = `/preprocess_train/${dataset.id}/`;
        return fetch(url, {
            method: 'POST',
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        })
        .then(response => response.json().then(data => {
            if (response.status === 200) {
                console.log(`Dataset ${dataset.id} already processed.`);
            } else if (response.status === 201) {
                console.log(`Dataset ${dataset.id} processed successfully.`);
            } else if (response.status === 400) {
                console.error(`Failed to read dataset ${dataset.id}: ${data.error}`);
            } else {
                console.error(`Unexpected status code ${response.status} for dataset ${dataset.id}.`);
            }
        }))
        .catch(error => {
            console.error(`There was an error processing dataset ${dataset.id}:`, error);
        });
    };

    // Function to preprocess all datasets
    const processAllDatasets = async (datasets) => {
        for (let i = 0; i < datasets.length; i++) {
            const dataset = datasets[i];
            const processingMessage = document.createElement('p');
            processingMessage.innerText = `Processing dataset of ${dataset.date}.....`;
            document.getElementById('step2').appendChild(processingMessage);

            await preprocessDataset(dataset);
        }
        const processingFinish = document.createElement('p');
        processingFinish.innerText = `.............Finished processing dataset............`;
        document.getElementById('step2').appendChild(processingFinish);

        document.getElementById('step3').style.display = 'block';
        for (let model of selectedModels) {
            await trainModel(model);
        }
    };

    // Function to train a single model
    const trainModel = async (model) => {
        const modelTrainingMessage = document.createElement('p');
        modelTrainingMessage.innerText = `Training model with ${model}.....`;
        document.getElementById('step3').appendChild(modelTrainingMessage);

        const url = `/incremental_train_model/${selectedMarket}/${model}/`;
        const formData = new FormData();
        formData.append('k', bestK);

        await fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error(`Error training model ${model}: ${data.error}`);
            } else {
                console.log(`Model ${model} trained.`);
                data.results.forEach(result => {
                    const datasetTrainingMessage = document.createElement('p');
                    datasetTrainingMessage.innerText = `  - Tested dataset on  ${result.dataset_date}, we get the accuracy ${result.score.toFixed(2)} `;
                    document.getElementById('training-results').appendChild(datasetTrainingMessage);
                });

                if (data.complete) {
                    const modelResultMessage = document.createElement('p');
                    modelResultMessage.innerText = `.......Finished training model ${model}.......`;
                    document.getElementById('training-results').appendChild(modelResultMessage);

                    // Store the final result for this model
                    finalResults[model] = data.results[data.results.length - 1];
                }
            }
        })
        .catch(error => {
            console.error(`There was an error training model ${model}:`, error);
        });

        showResultsAndBestModel();
    };

    // Function to normalize a metric
    const normalize = (value, minValue, maxValue) => {
        return (value - minValue) / (maxValue - minValue);
    };

    // Function to calculate weighted score
    const calculateWeightedScore = (metrics, weights) => {
        const { accuracy, precision, recall, f1Score } = metrics;
        return (
            normalize(accuracy, 0, 1) * weights.accuracy +
            normalize(precision, 0, 1) * weights.precision +
            normalize(recall, 0, 1) * weights.recall +
            normalize(f1Score, 0, 1) * weights.f1Score
        );
    };

   // Function to show the results and the best model
   const showResultsAndBestModel = () => {
        document.getElementById('step4').style.display = 'block';
        document.getElementById('step4').innerHTML = '<h3>Step 4: Results:</h3>';

        const table = document.createElement('table');
        table.setAttribute('border', '1'); // Optional: Add border to table for better visibility
        table.setAttribute('cellpadding', '7'); // Optional: Add padding to cells for better readability
        const headerRow = document.createElement('tr');
        ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Score'].forEach(text => {
            let th = document.createElement('th');
            th.innerText = text;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);
        best_model = null;
        best_model_score = 0;
        best_model_evaluation_id = 0;
        Object.keys(finalResults).forEach(model => {
            const result = finalResults[model];
            const metrics = {
                accuracy: result.score, // Assuming the score is accuracy
                precision: result.classification_report['macro avg'].precision,
                recall: result.classification_report['macro avg'].recall,
                f1Score: result.classification_report['macro avg']['f1-score']
            };

            const score = calculateWeightedScore(metrics, weights);
            if (score > best_model_score) {
                best_model_score = score;
                best_model = { model, score};
                best_model_evaluation_id = result.evaluation_id;
            }

            const row = document.createElement('tr');
            row.appendChild(createTableCell(model));
            row.appendChild(createTableCell(metrics.accuracy.toFixed(2))); // Use toFixed(2) for 2 decimal places
            row.appendChild(createTableCell(metrics.precision.toFixed(2))); // Use toFixed(2) for 2 decimal places
            row.appendChild(createTableCell(metrics.recall.toFixed(2))); // Use toFixed(2) for 2 decimal places
            row.appendChild(createTableCell(metrics.f1Score.toFixed(2))); // Use toFixed(2) for 2 decimal places
            row.appendChild(createTableCell(score.toFixed(2))); // Use toFixed(2) for 2 decimal places

            table.appendChild(row);
        });
        console.log("best model: ", best_model);
        document.getElementById('step4').appendChild(table);


        saveBestModel(best_model, best_model_evaluation_id);


        // Show the Train button again
        trainButton.style.display = 'block';
        document.getElementById('best-model').innerHTML = `<h3>Best model: </h3><p>Model: ${best_model.model}, Score: ${best_model_score.score}</p>`;
    };

    // Function to create a table cell
    const createTableCell = (text) => {
        let td = document.createElement('td');
        td.innerText = text;
        return td;
    };

    // Function to save the best model
    const saveBestModel = (bestModel, best_model_evaluation_id) => {
        const url = `/save_best_model/`;
        const formData = new FormData();
        formData.append('market', selectedMarket);
        formData.append('model', bestModel.model);
        formData.append('evaluation', best_model_evaluation_id);
        formData.append("best_k", bestK)
        console.log("best_k", bestK);

        fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error(`Error saving best model: ${data.error}`);
            } else {
                console.log(`Best model saved.`);
            }
        })
        .catch(error => {
            console.error(`There was an error saving the best model:`, error);
        });
    };
});
