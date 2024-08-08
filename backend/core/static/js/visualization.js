document.addEventListener("DOMContentLoaded", function(){
    console.log("Step..........1")
    var slider = document.getElementById("k_value");
    var output = document.getElementById("rangeValue");

    output.textContent = slider.value; // Display the default slider value

    // Update the current slider value (each time you drag the slider handle)
    slider.oninput = function() {
        console.log("Step..........2");
        output.textContent = this.value;
    };

    // Handle the form submission
    document.getElementById("Visualization-model-form").addEventListener('submit', function(event){
        event.preventDefault();
        // Perform your form submission logic here
        console.log("Step..........3");
        var datasetId = document.getElementById('Visualization-dataset-id').value;
        var marketName = document.getElementById('Visualization-market-name').value;
        var kValue = document.getElementById('k_value').value;
        console.log("Step..........4");
        fetch(`data_visualization/${datasetId}/${marketName}/${kValue}/`,{
            method: "GET",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            }
        })
        .then(response => response.text())
        .then(data => {
            // Load the response data (histogram) into the histogram content div
            console.log("Step..........5");
            document.getElementById('histogram-content').innerHTML = data;
        })
        .catch(error => console.error('Error:', error));
    });
});
function setDatasetIdVisualization(datasetId, marketName) {
    document.getElementById('Visualization-dataset-id').value = datasetId;
    document.getElementById('Visualization-market-name').value = marketName;
    document.getElementById('visualization-form-container').style.display = 'block';
}