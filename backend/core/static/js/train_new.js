document.addEventListener('DOMContentLoaded', function () {
    const marketSelect = document.getElementById('market');
    const selectAllCheckbox = document.getElementById('select-all');
    const checkboxes = document.querySelectorAll('.model-checkbox');
    const trainButton = document.querySelector('#train-model-form button[type="submit"]');
    const cancelledButton = document.getElementById('canceledButton');

    let selectedMarket = null;
    let selectedModels = [];
    let datasets = [];
    let bestK = null;
    let finalResults = {};
    let cancelRequest = false;

    const weights = {
        accuracy: 0.4,
        precision: 0.2,
        recall: 0.2,
        f1Score: 0.2
    };

    marketSelect?.addEventListener('change', handleMarketChange);
    selectAllCheckbox?.addEventListener('click', handleSelectAll);
    checkboxes.forEach(checkbox => checkbox.addEventListener('change', handleModelSelection));

    const trainForm = document.getElementById('train-model-form');
    trainForm?.addEventListener('submit', handleFormSubmit);

    // Step 1: Handle market selection
    function handleMarketChange() {
        selectedMarket = marketSelect.value;
        fetchDatasets(selectedMarket).then(fetchedDatasets => {
            datasets = fetchedDatasets;
        });
    }
    cancelledButton.addEventListener('click',()=>{
        cancelRequest = true;
        trainButton.style.display = 'block';
        console.log('cancelling...');
        alert('Please wait for cancellation because some processes have already sent to the server and it is running. Alternatively, you can refresh the page!')
        cancelledButton.style.display = 'none';
        
    });

    // Step 2: Handle "Select All" checkbox
    function handleSelectAll() {
        selectedModels = selectAllCheckbox.checked ? Array.from(checkboxes).map(cb => cb.value) : [];
        checkboxes.forEach(checkbox => (checkbox.checked = selectAllCheckbox.checked));
    }

    // Step 3: Handle individual model selection
    function handleModelSelection(event) {
        const model = event.target.value;
        if (event.target.checked) {
            selectedModels.push(model);
        } else {
            selectedModels = selectedModels.filter(selectedModel => selectedModel !== model);
        }
    }

    // Step 4: Fetch datasets for the selected market
    async function fetchDatasets(market) {
        const url = `/get_datasets/${market}/`;
        try {
            const response = await fetch(url);
            const data = await response.json();
            return data.error ? (alert(`Error: ${data.error}`), []) : data.datasets;
        } catch (error) {
            console.error("Error fetching datasets:", error);
            return [];
        }
    }

    // Step 5: Handle form submission and initiate the training process
    async function handleFormSubmit(event) {
        event.preventDefault();
        cancelRequest = false;
        const formData = new FormData(trainForm);

        selectedMarket = document.getElementById('market').value;
        const kMin = document.getElementById('k-min').value;
        const kMax = document.getElementById('k-max').value;

        formData.append('market', selectedMarket);
        selectedModels.forEach(model => formData.append('models', model));
        formData.append('k-min', kMin);
        formData.append('k-max', kMax);

        datasets = await fetchDatasets(selectedMarket);
        document.getElementById('training-steps').style.display = 'block';
        trainButton.style.display = 'none';
        cancelledButton.style.display = 'block';

        await findBestK(formData);
        await processAllDatasets(datasets);
        await trainAllModels();
        showResultsAndBestModel();
    }

    // Step 6: Find the best K value
    async function findBestK(formData) {
        document.getElementById('step1').style.display = 'block';
        const url = `/find_best_k/${selectedMarket}/`;
        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                }
            });
            const data = await response.json();
            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }
            bestK = data.k;
            displayBestK(data);
        } catch (error) {
            console.error('Error finding best k:', error);
        }
    }

    function displayBestK(data) {
        document.getElementById('choosing-k').style.display = 'none';
        document.getElementById('best-k').innerText = bestK;
        document.getElementById('k-value').style.display = 'block';

        const histogramContent = document.getElementById('histogram');
        histogramContent.innerHTML = data.histogram_html;
        Array.from(histogramContent.getElementsByTagName('script')).forEach(script => eval(script.innerText));
    }

    // Step 7: Preprocess all datasets
    async function processAllDatasets(datasets) {
        document.getElementById('step2').style.display = 'block';
        for (let dataset of datasets) {
            if (cancelRequest){
                console.log('Process canceled.');
                return;
            }
            await preprocessDataset(dataset);
        }
        const processingFinish = createMessage(`.............Finished processing dataset............`, 'step2');
    }

    async function preprocessDataset(dataset) {
        const url = `/preprocess_train/${dataset.id}/`;
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            const data = await response.json();
            logPreprocessStatus(response.status, dataset.id, data);
        } catch (error) {
            console.error(`Error processing dataset ${dataset.id}:`, error);
        }
    }

    function logPreprocessStatus(status, datasetId, data) {
        const statusMessages = {
            200: `Dataset ${datasetId} already processed.`,
            201: `Dataset ${datasetId} processed successfully.`,
            400: `Failed to read dataset ${datasetId}: ${data.error}`
        };
        console.log(statusMessages[status] || `Unexpected status code ${status} for dataset ${datasetId}.`);
    }

    // Step 8: Train all selected models
    async function trainAllModels() {
        document.getElementById('step3').style.display = 'block';
        for (let model of selectedModels) {
            if (cancelRequest){
                console.log('Training canceled.');
                return;
            }
            await trainModel(model);
        }
    }

    async function trainModel(model) {
        const url = `/incremental_train_model/${selectedMarket}/${model}/`;
        const modelTrainingMessage = document.createElement('div');
        const trainingMessage = document.createElement('h4');
        trainingMessage.textContent = `Training model with ${model} :`;
        modelTrainingMessage.appendChild(trainingMessage);
        document.getElementById('step3').appendChild(modelTrainingMessage);
      
        const formData = new FormData();
        formData.append('k', bestK);
      
        try {
          const response = await fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
              'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            }
          });
          const data = await response.json();
          if (data.error) {
            console.error(`Error training model ${model}: ${data.error}`);
          } else {
            logTrainingResults(model, data, modelTrainingMessage);
          }
        } catch (error) {
          console.error(`Error training model ${model}:`, error);
        }
      }
      
      function logTrainingResults(model, data, modelTrainingMessage) {
        const resultsList = document.createElement('ul');
        data.results.forEach(result => {
          const resultItem = document.createElement('li');
          resultItem.textContent = ` Tested dataset on  ${result.dataset_date}, we get the accuracy ${result.score.toFixed(2)} `;
          resultsList.appendChild(resultItem);
        });
        modelTrainingMessage.appendChild(resultsList);
      
        if (data.complete) {
          const finishedMessage = document.createElement('p');
          finishedMessage.textContent = `   .......Finished training model ${model}.......`;
          modelTrainingMessage.appendChild(finishedMessage);
          finalResults[model] = data.results[data.results.length - 1];
        }
      }
      
      

    // Step 9: Show results and identify the best model
    function showResultsAndBestModel() {
        if (cancelRequest){
            console.log('Results shown canceled.');
            return;
        }
        document.getElementById('step4').style.display = 'block';
        document.getElementById('step4').innerHTML = '<h3>Step 4: Results:</h3>';
        
        const table = createResultsTable();
        const bestModel = findBestModel();
        document.getElementById('step4').appendChild(table);

        saveBestModel(bestModel);
        trainButton.style.display = 'block';
        document.getElementById('best-model').style.display = 'block';
        document.getElementById('best-model').innerHTML = `<h3>Best model:</h3><p>Model: ${bestModel.model}, Score: ${bestModel.score.toFixed(2)}</p>`;
    }

    function createResultsTable() {
        const table = document.createElement('table');
        table.setAttribute('border', '1');
        table.setAttribute('cellpadding', '7');

        const headerRow = document.createElement('tr');
        ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Score'].forEach(text => {
            const th = document.createElement('th');
            th.innerText = text;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        Object.entries(finalResults).forEach(([model, result]) => {
            const metrics = {
                accuracy: result.score,
                precision: result.classification_report['macro avg'].precision,
                recall: result.classification_report['macro avg'].recall,
                f1Score: result.classification_report['macro avg']['f1-score']
            };

            const score = calculateWeightedScore(metrics, weights);
            const row = document.createElement('tr');

            // Create the 'Model' cell
            const modelCell = document.createElement('td');
            modelCell.innerText = model;
            row.appendChild(modelCell);
            //create cells for other metrics
            ['accuracy', 'precision', 'recall', 'f1Score', 'score'].forEach(key => {
                row.appendChild(createTableCell(metrics[key]?.toFixed(2) || '0.00'));
            });
            // Create the 'Score' cell
            row.appendChild(createTableCell(score.toFixed(2)));
            table.appendChild(row);
        });

        return table;
    }

    function createTableCell(text) {
        const td = document.createElement('td');
        td.innerText = text;
        return td;
    }

    function findBestModel() {
        let bestModel = null;
        let bestScore = 0;
        Object.entries(finalResults).forEach(([model, result]) => {
            const metrics = {
                accuracy: result.score,
                precision: result.classification_report['macro avg'].precision,
                recall: result.classification_report['macro avg'].recall,
                f1Score: result.classification_report['macro avg']['f1-score']
            };

            const score = calculateWeightedScore(metrics, weights);
            if (score > bestScore) {
                bestScore = score;
                bestModel = { model, score, evaluationId: result.evaluation_id };
            }
        });
        console.log("Best model:", bestModel);
        return bestModel;
    }

    // Step 10: Save the best model
    function saveBestModel(bestModel) {
        const url = `/save_best_model/`;
        const formData = new FormData();
        formData.append('market', selectedMarket);
        formData.append('model', bestModel.model);
        formData.append('evaluation', bestModel.evaluationId);
        formData.append('best_k', bestK);

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
            .catch(error => console.error(`Error saving the best model:`, error));
    }

    function calculateWeightedScore(metrics, weights) {
        const { accuracy, precision, recall, f1Score } = metrics;
        return (
            normalize(accuracy, 0, 1) * weights.accuracy +
            normalize(precision, 0, 1) * weights.precision +
            normalize(recall, 0, 1) * weights.recall +
            normalize(f1Score, 0, 1) * weights.f1Score
        );
    }

    function normalize(value, minValue, maxValue) {
        return (value - minValue) / (maxValue - minValue);
    }

    function createMessage(text, stepId) {
        const message = document.createElement('p');
        message.innerText = text;
        document.getElementById(stepId).appendChild(message);
        return message;
    }
});
