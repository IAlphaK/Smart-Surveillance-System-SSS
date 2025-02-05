import React from 'react';
import { Box, Typography, useTheme } from "@mui/material";
import FlexBetween from "./FlexBetween";

const StatBox = ({ title, value, increase, icon, description }) => {
  const theme = useTheme();
  return (
    <Box
      gridColumn="span 2"
      gridRow="span 1"
      display="flex"
      flexDirection="column"
      justifyContent="space-between"
      p="0.75rem 0.75rem"
      flex="1 1 100%"
      backgroundColor={theme.palette.background.alt}
      borderRadius="0.55rem"
    >
      <FlexBetween>
        <Typography variant="h6" sx={{ color: theme.palette.secondary[100] }}>
          {title}
        </Typography>
        {icon}
      </FlexBetween>

      <Typography
        variant="h3"
        fontWeight="600"
        sx={{ color: theme.palette.secondary[200] }}
      >
        {value}
      </Typography>

      <FlexBetween gap="1rem" alignItems="flex-start">
        <Typography
          variant="h5"
          fontStyle="italic"
          sx={{ color: theme.palette.secondary[200] }}
        >
          {increase}
        </Typography>
        {/* Align description text to the left */}
        <Typography sx={{ textAlign: 'left', flex: 1 }}>
          {description}
        </Typography>
      </FlexBetween>
    </Box>
  );
}

export default StatBox;
