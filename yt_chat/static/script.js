<script>
    function sendMessage() {
        var userInput = document.getElementById('user-input').value;
        var messagesContainer = document.getElementById('messages-container');
        var userMessage = '<strong>You:</strong> ' + userInput + '<br>';
        messagesContainer.innerHTML += userMessage;

        // Send user message to the server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'video_url=' + encodeURIComponent('{{ video_url }}') + '&user_message=' + encodeURIComponent(userInput),
        })
        .then(response => response.json())
        .then(data => {
            var botResponse = '<strong>Bot:</strong> ' + data.bot_response + '<br>';
            messagesContainer.innerHTML += botResponse;
        })
        .catch(error => console.error('Error:', error));

        // Clear user input
        document.getElementById('user-input').value = '';
    }
</script>
