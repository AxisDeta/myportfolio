document.addEventListener('DOMContentLoaded', function () {
    const textElement = document.getElementById('typed-text');
    if (textElement) { // Check if element exists before typing
        const fullText = "Glad you dropped by. I'm passionate about making sense of data and building impactful solutions with it. Feel free to have a look around.";
        let charIndex = 0;

        function typeText() {
            if (charIndex < fullText.length) {
                textElement.textContent += fullText.charAt(charIndex);
                charIndex++;
                setTimeout(typeText, 15); // Adjust typing speed here (milliseconds)
            }
        }
        typeText(); // Start typing only if element exists
    }

    // Intersection Observer for fade-in animation
    const sections = document.querySelectorAll('section');
    const categoryContainers = document.querySelectorAll('.category-container');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
            }
        });
    }, { threshold: 0.1 }); // Adjust threshold as needed

    sections.forEach(section => {
        section.classList.add('fade-in-section');
        observer.observe(section);
    });

    // Observe category containers for scroll animations
    categoryContainers.forEach(container => {
        observer.observe(container);
    });

    // Animated Counter for Impact Metrics
    function animateCounter(element) {
        const targetStr = element.getAttribute('data-target');
        const target = parseFloat(targetStr);
        const isDecimal = targetStr.includes('.');
        const label = element.nextElementSibling.textContent;
        const isPercentage = label.includes('%');
        const duration = 2000; // 2 seconds
        const increment = target / (duration / 16); // 60fps
        let current = 0;

        const updateCounter = () => {
            current += increment;
            if (current < target) {
                element.textContent = isDecimal ? current.toFixed(1) : Math.floor(current);
                requestAnimationFrame(updateCounter);
            } else {
                // Add + sign to all except percentage
                const finalValue = isDecimal ? target.toFixed(1) : target;
                element.textContent = isPercentage ? finalValue : finalValue + '+';
            }
        };

        updateCounter();
    }

    // Trigger counter animation when hero section is visible
    const heroSection = document.querySelector('#hero-section');
    if (heroSection) {
        const heroObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const metricNumbers = document.querySelectorAll('.metric-number');
                    metricNumbers.forEach(num => animateCounter(num));
                    heroObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.3 });

        heroObserver.observe(heroSection);
    }

    // Image enlargement using a lightbox approach
    const magnifiableImages = document.querySelectorAll('.magnifiable');
    const body = document.body;

    magnifiableImages.forEach(image => {
        image.addEventListener('click', function () {
            // Create overlay
            const overlay = document.createElement('div');
            overlay.classList.add('overlay', 'active');

            // Create lightbox container
            const lightbox = document.createElement('div');
            lightbox.classList.add('lightbox');

            // Create image inside lightbox
            const lightboxImage = document.createElement('img');
            lightboxImage.src = image.src;

            // Add image to lightbox
            lightbox.appendChild(lightboxImage);

            // Add overlay and lightbox to body
            body.appendChild(overlay);
            body.appendChild(lightbox);
            body.style.overflow = 'hidden'; // Prevent scrolling

            // Close function
            const closeLightbox = () => {
                body.removeChild(overlay);
                body.removeChild(lightbox);
                body.style.overflow = '';
            };

            overlay.addEventListener('click', closeLightbox);
            lightbox.addEventListener('click', closeLightbox);
        });
    });

    // AI Chat Modal Logic
    const aiChatModalContainer = document.getElementById('aiChatModalContainer');
    let currentItemId = null;
    let currentItemType = null;

    document.querySelectorAll('.ask-ai-btn').forEach(button => {
        button.addEventListener('click', async function () {
            try {
                console.log('Ask AI button clicked');
                currentItemId = this.dataset.itemId;
                currentItemType = this.dataset.itemType;
                console.log('Item ID:', currentItemId, 'Item Type:', currentItemType);

                if (!aiChatModalContainer) {
                    console.error('aiChatModalContainer not found in DOM!');
                    return;
                }

                // Load the modal content dynamically
                console.log('Fetching modal content...');
                const response = await fetch(`/ai_chat_modal/${currentItemId}`);
                console.log('Fetch response status:', response.status);

                if (!response.ok) {
                    console.error('Failed to fetch modal:', response.statusText);
                    return;
                }

                const modalHtml = await response.text();
                console.log('Modal HTML received, length:', modalHtml.length);
                aiChatModalContainer.innerHTML = modalHtml;

                const aiChatModal = document.getElementById('aiChatModal');
                if (!aiChatModal) {
                    console.error('aiChatModal element not found after inserting HTML!');
                    return;
                }

                const closeButton = aiChatModal.querySelector('.close-button');
                const chatInput = document.getElementById('chatInput');
                const sendMessageButton = document.getElementById('sendMessageButton');
                const chatWindow = document.getElementById('chatWindow');

                console.log('Setting modal display to block');
                aiChatModal.style.display = 'block';
                body.style.overflow = 'hidden';

                closeButton.onclick = function () {
                    aiChatModal.style.display = 'none';
                    aiChatModalContainer.innerHTML = ''; // Clear modal content
                    body.style.overflow = '';
                }

                window.onclick = function (event) {
                    if (event.target == aiChatModal) {
                        aiChatModal.style.display = 'none';
                        aiChatModalContainer.innerHTML = ''; // Clear modal content
                        body.style.overflow = '';
                    }
                }

                sendMessageButton.onclick = sendMessage;
                chatInput.addEventListener('keypress', function (e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });

                // Initial AI message or welcome
                appendMessage('ai', `Hello! I'm here to answer questions about ${document.getElementById('chatTitle').textContent}. What would you like to know?`);

                async function sendMessage() {
                    const userMessage = chatInput.value;
                    if (userMessage.trim() === '') return;

                    appendMessage('user', userMessage);
                    chatInput.value = '';

                    // Show typing indicator
                    const typingIndicator = appendMessage('ai', '...', true);

                    try {
                        const aiResponse = await fetch('/ask_ai', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ item_id: currentItemId, query: userMessage })
                        });
                        const data = await aiResponse.json();

                        // Remove typing indicator
                        typingIndicator.remove();
                        appendMessage('ai', data.response);
                    } catch (error) {
                        console.error('Error asking AI:', error);
                        // Remove typing indicator
                        typingIndicator.remove();
                        appendMessage('ai', 'Sorry, I am having trouble connecting to the AI. Please try again later.');
                    }
                }

                function appendMessage(sender, message, isTypingIndicator = false) {
                    const messageElement = document.createElement('div');
                    messageElement.classList.add('chat-message', sender + '-message');
                    if (isTypingIndicator) {
                        messageElement.classList.add('typing-indicator');
                        messageElement.textContent = message; // Set content directly for typing indicator
                    } else {
                        // Convert markdown to HTML for better formatting
                        let formattedMessage = message;

                        // Convert **bold** to <strong>
                        formattedMessage = formattedMessage.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

                        // Convert bullet points (* item) to proper list items
                        formattedMessage = formattedMessage.replace(/^\* (.+)$/gm, '<li>$1</li>');

                        // Wrap consecutive list items in <ul> tags
                        formattedMessage = formattedMessage.replace(/(<li>.*<\/li>\n?)+/g, function (match) {
                            return '<ul>' + match + '</ul>';
                        });

                        // Convert newlines to <br> for proper display
                        formattedMessage = formattedMessage.replace(/\n/g, '<br>');

                        messageElement.innerHTML = formattedMessage;
                    }
                    chatWindow.appendChild(messageElement);
                    chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to bottom
                    return messageElement; // Return the element for potential removal (e.g., typing indicator)
                }
            } catch (error) {
                console.error('Error in Ask AI button handler:', error);
                alert('Sorry, there was an error opening the AI chat. Please check the console for details.');
            }
        });
    });
});
