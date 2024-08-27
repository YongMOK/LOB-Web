document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("uploadForm");
    const uploadButton = document.getElementById("uploadButton");
    const viewResults = document.getElementById("view_results");
    const loadingSpinner = document.getElementById("loadingSpinner");

    let results = [];
    let predictions_success = false;

    if (uploadForm) {
        uploadForm.addEventListener("submit", function (event) {
            event.preventDefault(); // Prevent the default form submission
            uploadButton.style.display = "none";
            loadingSpinner.style.display = "block";
            submitForm();
        });
    }

    function submitForm() {
        const formData = new FormData(uploadForm);

        console.log("fetching form data...");
        fetch("/upload_predictions/", {
            method: "POST",
            headers: {
                "X-CSRFToken": getCSRFToken(),
            },
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    console.error("Error:", data.error);
                    alert(data.error);
                    uploadButton.style.display = "block";
                } else {
                    results = JSON.parse(data.result);
                    alert(data.message);
                    predictions_success = true;
                    handle_results(results);
                    uploadButton.style.display = "block";
                }
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("An error occurred during processing.");
                uploadButton.style.display = "block";
            })
            .finally(() => {
                // Hide loading spinner
                loadingSpinner.style.display = "none";
            });
    }

    function handle_results(results) {
        console.log('Results:', results);
        if (predictions_success) {
            viewResults.style.display = "block";
            plot_results(results);
        }
    }

    function plot_results(results) {
        if (!Array.isArray(results)) {
            console.error("Results is not an array:", results);
            return;
        }
        const chartData = results.map((item) => [
            new Date(item.timestamp).getTime(),
            item.prediction,
        ]);

        if (chartData.length == 0) {
            console.error("No results to plot.");
            return;
        }

        Highcharts.stockChart('resultsChart', {
            chart: {
                type: 'line'
            },
            title: {
                text: 'Prediction Results of Mid-Price Movements'
            },
            rangeSelector: {
                buttons: [
                    { type: 'hour', count: 1, text: '1h' },
                    { type: 'minute', count: 30, text: '30m' },
                    { type: 'minute', count: 15, text: '15m' },
                    { type: 'minute', count: 10, text: '10m' },
                    { type: 'minute', count: 7, text: '7m' },
                    { type: 'minute', count: 5, text: '5m' },
                    { type: 'all', text: 'All' },
                ],
                selected: 3,
                inputEnabled: false
            },
            series: [{
                name: 'Mid-price movement',
                data: chartData,
                tooltip: {
                    valueDecimals: 2
                }
            }],
            xAxis: {
                type: 'datetime',
                min: Math.min(new Date(results[0].timestamp).getTime(), new Date().getTime() - 3600 * 1000), // 1 hour before current or dataset start
                max: Math.max(new Date(results[results.length - 1].timestamp).getTime(), new Date(results[results.length - 1].timestamp).getTime() + 180 * 1000), 
                plotLines: [{
                    color: 'red', // Color of the current time line
                    width: 2,
                    value: new Date().getTime(), // Current time
                    dashStyle: 'Dash',
                    label: {
                        text: 'Current Time',
                        align: 'left',
                        style: {
                            color: 'red'
                        }
                    }
                }, {
                    color: 'blue', // Color of the line for the second-to-last timestamp
                    width: 2,
                    value: results[results.length - 2].timestamp, // Second-to-last result timestamp
                    dashStyle: 'Dash',
                    label: {
                        text: 'Actual Time',
                        align: 'left',
                        style: {
                            color: 'blue'
                        }
                    }
                }]
            },
            yAxis: {
                title: {
                    text: 'Accumulated Movement'
                },
                min: Math.min(...results.map(item => item.prediction)) - 1, // Adding buffer for aesthetics
                max: Math.max(...results.map(item => item.prediction)) + 1, // Adding buffer for aesthetics
            },
            navigator: {
                enabled: true
            },
            scrollbar: {
                enabled: true
            },
            accessibility: {
                enabled: false
            }
        });
    }

    function getCSRFToken() {
        const cookies = document.cookie.split(";");
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.startsWith("csrftoken=")) {
                return cookie.substring("csrftoken=".length, cookie.length);
            }
        }
        return "";
    }
});
