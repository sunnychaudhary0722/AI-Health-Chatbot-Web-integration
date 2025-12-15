const chatBox = document.getElementById('chat-box');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const typing = document.getElementById('typing');

function appendMessage(text, sender) {
  const msg = document.createElement('div');
  msg.className = `message ${sender}`;
  msg.innerHTML = sender === 'bot' ? text : escapeHTML(text);
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function escapeHTML(str) {
  return str.replace(/[&<>"']/g, m => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[m]));
}

function showTyping(show) {
  typing.style.display = show ? 'block' : 'none';
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  appendMessage(text, 'user');
  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;

  showTyping(true);

  // ⏳ artificial delay
  await new Promise(res => setTimeout(res, 2000));

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ message: text })
    });

    const data = await res.json();
    const reply = (data.reply || '').replace(/\n/g, '<br>');

    showTyping(false);
    appendMessage(reply, 'bot');

  } catch (e) {
    showTyping(false);
    appendMessage('⚠️ Server error. Please try again.', 'bot');
  }

  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
}

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});
