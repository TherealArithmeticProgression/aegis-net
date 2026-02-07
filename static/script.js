// Static script.js for interactive features

// Function to display current date and time
function displayDateTime() {
    const now = new Date();
    const formattedDate = now.toISOString().substring(0, 19).replace('T', ' ');
    console.log('Current Date and Time (UTC):', formattedDate);
}

displayDateTime();