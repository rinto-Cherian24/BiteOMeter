document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("nailForm");
    const nameInput = document.getElementById("name");
    const fileInput = document.getElementById("image");
    const submitBtn = document.getElementById("submitBtn");
    const resetBtn = document.getElementById("resetBtn");
    const simpsonAnimation = document.getElementById("simpson-animation");

    // Function to convert file to base64
    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    // Function to show loading state
    function showLoading() {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    }

    // Function to hide loading state
    function hideLoading() {
        submitBtn.disabled = false;
        submitBtn.innerHTML = 'Submit';
    }

    // Function to show result
    function showResult(data) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-display';
        
        // Determine message based on health logic
        let message, emoji;
        if (data.label === 'Bite') {
            message = 'Your nails look healthy! They\'re safe to bite if you want to. 🍩💅';
            emoji = '🍩';
        } else {
            message = 'Your nails need some TLC! Better not bite them for now. 💪✨';
            emoji = '💪';
        }
        
        resultDiv.innerHTML = `
            <div class="result-card ${data.label === 'Bite' ? 'bite-result' : 'nobite-result'}">
                <h3>${data.name}, your nail health analysis is ready! ${emoji}</h3>
                <div class="prediction">
                    <span class="label">${data.label}</span>
                    <span class="confidence">${data.confidence}% confidence</span>
                </div>
                <p class="message">${message}</p>
                <p class="health-info"><small>Health Score: ${(data.health_score * 100).toFixed(1)}%</small></p>
            </div>
        `;

        // Remove any existing result
        const existingResult = document.querySelector('.result-display');
        if (existingResult) {
            existingResult.remove();
        }

        // Add result after form
        form.parentNode.insertBefore(resultDiv, form.nextSibling);

        // Show Homer animation
        simpsonAnimation.style.display = "block";
        setTimeout(() => {
            simpsonAnimation.style.display = "none";
        }, 3000);
    }

    // Function to show error message
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-display';
        errorDiv.innerHTML = `
            <div class="error-card">
                <h3>❌ Error</h3>
                <p>${message}</p>
                <p><small>Please check if the backend is running and try again.</small></p>
            </div>
        `;

        // Remove any existing error or result
        const existingError = document.querySelector('.error-display');
        const existingResult = document.querySelector('.result-display');
        if (existingError) existingError.remove();
        if (existingResult) existingResult.remove();

        // Add error after form
        form.parentNode.insertBefore(errorDiv, form.nextSibling);
    }

    // Submit form
    form.addEventListener("submit", async function (e) {
        e.preventDefault();

        const name = nameInput.value.trim();
        const file = fileInput.files[0];

        if (!name || !file) {
            alert("Please enter your name and select an image!");
            return;
        }

        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            alert("Please select a valid image file (JPEG, PNG, GIF, or WebP)!");
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert("Image file is too large! Please select an image smaller than 10MB.");
            return;
        }

        try {
            showLoading();
            console.log("Starting image upload process...");

            // Convert image to base64
            console.log("Converting image to base64...");
            const base64Image = await fileToBase64(file);
            console.log("Image converted successfully. Length:", base64Image.length);

            // Prepare request data
            const requestData = {
                name: name,
                image: base64Image
            };

            console.log("Sending request to backend...");
            console.log("Backend URL: http://127.0.0.1:5000/predict");

            // Send to Flask backend with better error handling
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(requestData)
            });

            console.log("Response received. Status:", response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Backend error response:", errorText);
                throw new Error(`Server error (${response.status}): ${errorText}`);
            }

            const data = await response.json();
            console.log("Backend response data:", data);
            
            if (data.error) {
                throw new Error(data.error);
            }

            showResult(data);
            
        } catch (error) {
            console.error("Detailed error:", error);
            
            // Handle specific error types
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                showError("Cannot connect to the backend server. Please make sure the Flask server is running on http://127.0.0.1:5000");
            } else if (error.message.includes('CORS')) {
                showError("CORS error. Please check if the backend has CORS enabled.");
            } else {
                showError(`Error: ${error.message}`);
            }
        } finally {
            hideLoading();
        }
    });

    // Reset form
    resetBtn.addEventListener("click", function () {
        form.reset();
        const existingResult = document.querySelector('.result-display');
        const existingError = document.querySelector('.error-display');
        if (existingResult) existingResult.remove();
        if (existingError) existingError.remove();
    });

    // Test backend connection on page load
    async function testBackendConnection() {
        try {
            const response = await fetch("http://127.0.0.1:5000/", {
                method: "GET",
                headers: {
                    "Accept": "text/plain"
                }
            });
            
            if (response.ok) {
                console.log("✅ Backend connection successful");
            } else {
                console.warn("⚠️ Backend responded with status:", response.status);
            }
        } catch (error) {
            console.warn("⚠️ Backend connection test failed:", error.message);
        }
    }

    // Test connection when page loads
    testBackendConnection();
});
