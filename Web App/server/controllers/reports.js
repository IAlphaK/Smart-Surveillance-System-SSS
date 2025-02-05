import PDFDocument from 'pdfkit';
import Incident from "../models/Incident.js";
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import moment from 'moment'; // You can use moment.js to format dates

// Moderate width and height for a balance between quality and size
const width = 800; // Keep width high for quality
const height = 600; // Keep height high for quality
const chartJSNodeCanvas = new ChartJSNodeCanvas({
    width,
    height,
    chartCallback: (ChartJS) => {
        // Set pixel ratio to enhance resolution
        ChartJS.defaults.devicePixelRatio = 1.5; 
        
    }
});

// Function to generate a bar chart image
// Function to generate a bar chart image
// Function to generate a bar chart image
const generateBarChartImage = async (labels, data) => {
    const configuration = {
        type: 'bar',
        data: {
            labels: labels, // "Smoking" and "Fighting"
            datasets: [{
                label: '',
                data: data, // [smokingCount, fightingCount]
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)', // Color for smoking
                    'rgba(54, 162, 235, 0.5)', // Color for fighting
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Incidents'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Behavior Type'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 14 // Adjust font size for better readability
                        }
                    }
                }
            }
        }
    };
    return await chartJSNodeCanvas.renderToBuffer(configuration);
};


// Function to generate a doughnut chart image
const generateDoughnutChartImage = async (labels, data) => {
    const configuration = {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)', // Color for smoking
                    'rgba(54, 162, 235, 0.5)', // Color for fighting
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                ],
                borderWidth: 1
            }]
        },
        options: {
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 14 // Adjust font size for better readability
                        }
                    }
                },
                // Use the datalabels plugin to show counts inside the segments
                datalabels: {
                    color: '#000', // Label color
                    font: {
                        size: 16, // Label font size
                        weight: 'bold',
                    },
                    formatter: (value) => value // Show the count directly
                }
            }
        }
    };
    return await chartJSNodeCanvas.renderToBuffer(configuration);
};

// Function to generate a line chart image for incidents by camera
const generateLineChartImage = async (cameras, smokingData, fightingData) => {
    const configuration = {
        type: 'line',
        data: {
            labels: cameras, // Camera names as labels on the x-axis
            datasets: [
                {
                    label: 'Smoking',
                    data: smokingData, // Counts of smoking incidents by camera
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    fill: false,
                    tension: 0.1 // Smoothness of the line
                },
                {
                    label: 'Fighting',
                    data: fightingData, // Counts of fighting incidents by camera
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    fill: false,
                    tension: 0.1 // Smoothness of the line
                }
            ]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Incidents'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Camera'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 14 // Adjust font size for better readability
                        }
                    }
                }
            }
        }
    };
    return await chartJSNodeCanvas.renderToBuffer(configuration);
};

// Function to generate a bubble chart image for incidents by camera
const generateBubbleChartImage = async (cameras, smokingData, fightingData) => {
    // Prepare data for bubble chart
    const smokingBubbles = cameras.map((camera, index) => ({
        x: index, // Camera index as x-coordinate
        y: smokingData[index], // Number of smoking incidents as y-coordinate
        r: Math.sqrt(smokingData[index]) * 3 // Bubble size scaled by the number of incidents
    }));

    const fightingBubbles = cameras.map((camera, index) => ({
        x: index, // Camera index as x-coordinate
        y: fightingData[index], // Number of fighting incidents as y-coordinate
        r: Math.sqrt(fightingData[index]) * 3 // Bubble size scaled by the number of incidents
    }));

    const configuration = {
        type: 'bubble',
        data: {
            labels: cameras, // Camera names as labels on the x-axis
            datasets: [
                {
                    label: 'Smoking',
                    data: smokingBubbles, // Bubble data for smoking incidents
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                },
                {
                    label: 'Fighting',
                    data: fightingBubbles, // Bubble data for fighting incidents
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                }
            ]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Incidents'
                    }
                },
                x: {
                    type: 'category',
                    labels: cameras, // Camera names as x-axis labels
                    title: {
                        display: true,
                        text: 'Camera'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 14 // Adjust font size for better readability
                        }
                    }
                }
            }
        }
    };
    return await chartJSNodeCanvas.renderToBuffer(configuration);
};

// Function to generate PDF
export const generateReport = async (req, res) => {
    try {
        // Fetch data from MongoDB
        const incidents = await Incident.find();
        const smokingIncidents = incidents.filter(incident => incident.behaviourDetected === "Smoking");
        const fightingIncidents = incidents.filter(incident => incident.behaviourDetected === "Fighting");

        // Generate chart data for bar and doughnut charts
        const labels = ["Smoking", "Fighting"];
        const data = [smokingIncidents.length, fightingIncidents.length];

        // Aggregate data by camera
        const cameras = [...new Set(incidents.map(incident => incident.cameraName))]; // Unique list of camera names
        const smokingDataByCamera = cameras.map(camera => smokingIncidents.filter(incident => incident.cameraName === camera).length);
        const fightingDataByCamera = cameras.map(camera => fightingIncidents.filter(incident => incident.cameraName === camera).length);

        // Generate bar chart image
        const barChartImage = await generateBarChartImage(labels, data, 'Incident Counts');

        // Generate doughnut chart image
        const doughnutChartImage = await generateDoughnutChartImage(labels, data);

        // Generate line chart image by camera
        const lineChartImage = await generateLineChartImage(cameras, smokingDataByCamera, fightingDataByCamera);

        // Generate bubble chart image by camera
        const bubbleChartImage = await generateBubbleChartImage(cameras, smokingDataByCamera, fightingDataByCamera);

        // Create a new PDF document
        const doc = new PDFDocument();
        
        // Set headers for downloading PDF
        res.setHeader('Content-disposition', 'attachment; filename="incident_report.pdf"');
        res.setHeader('Content-type', 'application/pdf');

        // Pipe the PDF document to the response
        doc.pipe(res);

        // Add title and findings
        doc.fontSize(20).text('SSMS Incident Report', { align: 'center' });
        doc.moveDown();

        // Add description of findings for bar chart
        doc.fontSize(16).text('Findings from the Bar Chart:', { underline: true });
        doc.moveDown();
        doc.fontSize(12).text(`The bar chart shows the total number of incidents categorized as either "Smoking" or "Fighting". In this dataset, "Smoking" incidents are represented in red and account for ${smokingIncidents.length} occurrences. "Fighting" incidents are represented in blue and account for ${fightingIncidents.length} occurrences. The chart visually compares these behaviors to easily identify the prevalent behavior.`);
        doc.moveDown();

        // Add bar chart to PDF
        doc.fontSize(16).text('Incident Bar Chart:', { underline: true, align: 'center' });
        doc.moveDown();
        // Center the image and slightly reduce size
        doc.image(barChartImage, {
            fit: [500, 350], // Reduced size while maintaining quality
            align: 'center',
            valign: 'center'
        });

        // Ensure there's enough space before adding the doughnut chart
        doc.moveDown(27);

        // Add description of findings for doughnut chart
        doc.fontSize(16).text('Findings from the Doughnut Chart:', { underline: true });
        doc.moveDown();
        doc.fontSize(12).text(`The doughnut chart visualizes the proportion of incidents categorized as either "Smoking" or "Fighting". In this dataset, "Smoking" incidents are represented in red, accounting for ${smokingIncidents.length} occurrences. "Fighting" incidents are represented in blue and account for ${fightingIncidents.length} occurrences. The doughnut chart makes it easy to see the relative distribution of each behavior.`);
        
        // Ensure there's enough space before adding the doughnut chart
        doc.moveDown(2);

        // Add doughnut chart to PDF
        doc.fontSize(16).text('Incident Doughnut Chart:', { underline: true, align: 'center' });
        doc.moveDown();
        doc.image(doughnutChartImage, {
            fit: [400, 400], // Reduced size while maintaining quality
            align: 'center',
            valign: 'center',
            x: 115
        });
        
        doc.moveDown(27);

        // Add description of findings for line chart
        doc.fontSize(16).text('Findings from the Line Chart:', { underline: true });
        doc.moveDown();
        doc.fontSize(12).text(`The line chart provides a detailed overview of the distribution of incidents categorized as either 'Smoking' or 'Fighting' across various cameras. Each camera is represented on the x-axis, and the y-axis displays the number of incidents detected. In this dataset, the 'Smoking' incidents are depicted by a red line, while the 'Fighting' incidents are represented by a blue line. This visualization allows us to quickly identify which cameras are detecting more occurrences of each behavior. By analyzing the trends, we can determine if specific cameras are more prone to capturing particular types of incidents, highlighting potential hotspots for either behavior.`);
        doc.moveDown(5);

        // Add line chart to PDF
        doc.fontSize(16).text('Incident Line Chart by Camera:', { underline: true, align: 'center' });
        doc.moveDown();
        doc.image(lineChartImage, { fit: [500, 350], align: 'center', valign: 'center' });

        doc.moveDown(27);

        // Add description of findings for bubble chart
        doc.fontSize(16).text('Findings from the Bubble Chart:', { underline: true });
        doc.moveDown();
        doc.fontSize(12).text(`The bubble chart visualizes the number of incidents categorized as either 'Smoking' or 'Fighting' across different cameras. Each camera is represented on the x-axis, with the y-axis showing the number of incidents detected. The size of each bubble corresponds to the number of incidents detected, making it easy to identify which cameras capture the most incidents and which behavior is more prevalent.`);
        
        doc.moveDown(5);
        
        // Add bubble chart to PDF
        doc.fontSize(16).text('Incident Bubble Chart by Camera:', { underline: true, align: 'center' });
        doc.moveDown();
        doc.image(bubbleChartImage, { fit: [500, 350], align: 'center', valign: 'center' });

        // Finalize the PDF and end the stream
        doc.end();
    } catch (error) {
        console.error("Error generating report:", error);
        res.status(500).send("Failed to generate report");
    }
};