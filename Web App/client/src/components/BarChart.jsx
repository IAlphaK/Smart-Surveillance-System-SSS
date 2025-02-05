import React from 'react';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Typography, useTheme } from "@mui/material";

const BarChart = ({ title ,data }) => {
    const theme = useTheme();

    const chartData = [
        {
            name: 'Smoking',
            count: data.smokingIncidentsCount,
        },
        {
            name: 'Fighting',
            count: data.fightingIncidentsCount,
        },
    ];

    return (
        <ResponsiveContainer width="100%" height={300}>
            
            <RechartsBarChart
                data={chartData}
                margin={{
                    top: 5, right: 30, left: 20, bottom: 5,
                }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" stroke={theme.palette.text.primary} />
                <YAxis stroke={theme.palette.text.primary} 
                       tickFormatter={(tick) => Number.isInteger(tick) ? tick : ''}
                       allowDecimals={false}
                       domain={[0, 'dataMax']}
                />
                <Tooltip 
                    contentStyle={{
                        backgroundColor: theme.palette.background.default,
                        borderColor: theme.palette.secondary.main,
                    }}
                    itemStyle={{
                        color: theme.palette.text.primary,
                    }}
                />
                <Legend />
                <Bar dataKey="count" fill={theme.palette.secondary.main} barSize={80} />
            </RechartsBarChart>
        </ResponsiveContainer>
    );
}

export default BarChart;
