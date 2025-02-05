import React from 'react';
import { ResponsivePie } from '@nivo/pie';
import { useTheme } from '@mui/material';

const PieChart = ({ data }) => {
    const theme = useTheme();

    // Constructing the data for the pie chart
    const chartData = [
        {
            id: "Smokers",
            label: "Smokers",
            value: data.smokingIncidentsCount,
            color: theme.palette.error.main,
        },
        {
            id: "Fighters",
            label: "Fighters",
            value: data.fightingIncidentsCount,
            color: theme.palette.info.main,
        },
    ];

    return (
        <div style={{ height: "400px" }}>
            <ResponsivePie
                data={chartData}
                margin={{ top: 40, right: 40, bottom: 80, left: 40 }}
                innerRadius={0.5}
                padAngle={0.7}
                cornerRadius={3}
                colors={{ datum: 'data.color' }}
                borderWidth={1}
                borderColor={{ from: 'color', modifiers: [['darker', 0.2]] }}
                
                // Disable radial labels (outside labels around the pie)
                radialLabelsSkipAngle={0}
                radialLabelsTextColor="#00000000" // Transparent color
                radialLabelsLinkColor="#00000000" // Transparent color
                
                // Slice Labels (inside the slices)
                sliceLabelsSkipAngle={10}
                sliceLabelsTextColor={theme.palette.text.primary}
                
                // Tooltip Styling
                tooltip={({ datum }) => (
                    <div
                        style={{
                            background: theme.palette.background.default,
                            padding: '5px 10px',
                            borderRadius: '5px',
                            color: theme.palette.text.primary,
                            border: `1px solid ${theme.palette.secondary.main}`,
                        }}
                    >
                        <strong>{datum.label}:</strong> {datum.value}
                    </div>
                )}
                
                legends={[{
                    anchor: 'bottom',
                    direction: 'row',
                    justify: false,
                    translateX: 0,
                    translateY: 56,
                    itemsSpacing: 0,
                    itemWidth: 100,
                    itemHeight: 18,
                    itemTextColor: theme.palette.text.primary,
                    itemDirection: 'left-to-right',
                    itemOpacity: 1,
                    symbolSize: 18,
                    symbolShape: 'circle',
                    effects: [{
                        on: 'hover',
                        style: {
                            itemTextColor: theme.palette.text.primary
                        }
                    }]
                }]}
            />
        </div>
    );
};

export default PieChart;
