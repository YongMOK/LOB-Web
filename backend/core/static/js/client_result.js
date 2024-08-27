document.addEventListener('DOMContentLoaded', function() {
    // Get the results JSON data from the script tag
    const resultsScript = document.getElementById('resultsData');
    const results = JSON.parse(resultsScript.textContent || '{}');
    console.log("Parsed results:", results);

    if (Array.isArray(results) && results.length > 0) {
        plot_results(results);
    } else {
        console.error("No valid results to plot or results are not in an array format.");
    }
});

function plot_results(results) {
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
            plotLines: chartData.length > 1 ? [{
                color: 'red',
                width: 2,
                value: new Date().getTime(),
                dashStyle: 'Dash',
                label: {
                    text: 'Current Time',
                    align: 'left',
                    style: {
                        color: 'red'
                    }
                }
            }, {
                color: 'blue',
                width: 2,
                value: results[results.length - 2].timestamp,
                dashStyle: 'Dash',
                label: {
                    text: 'Actual Time',
                    align: 'left',
                    style: {
                        color: 'blue'
                    }
                }
            }] : [],
        },
        yAxis: {
            title: {
                text: 'Accumulated Movement'
            },
            min: Math.min(...results.map(item => item.prediction)) - 1,
            max: Math.max(...results.map(item => item.prediction)) + 1,
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
