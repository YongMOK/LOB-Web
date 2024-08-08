document.addEventListener('DOMContentLoaded', function() {
    // Get the results JSON data from the script tag
    const resultsScript = document.getElementById('resultsData');
    const results = JSON.parse(resultsScript.textContent || '{}');
    console.log(results);

    if (results.length > 0) {
        plot_results(results);
    }
});

function plot_results(results) {
    if (!Array.isArray(results)) {
        console.error("Results is not an array:", results);
        return;
    }
    
    const chartData = results.map((item) => [
        new Date(item.timestamp).getTime(),
        item.prediction,
    ]);

    if (chartData.length === 0) {
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

// Function to get the CSRF token from the cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
