// styled component is way to reuse the styles, or css in a componenet-like manner
const{Box} = require("@mui/material");
const {styled} = require("@mui/system");

const FlexBetween = styled(Box)({
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center"  
});

export default FlexBetween;