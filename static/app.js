const chatForm = document.querySelector("#chatForm");
const chatViewport = document.querySelector("#chatViewport");
const messagesEl = document.querySelector("#messages");
const medicalModelBadge = document.querySelector("#medicalModelBadge");
const modelSelect = document.querySelector("#modelSelect");
const promptInput = document.querySelector("#promptInput");
const sendButton = document.querySelector("#sendButton");

const messages = [];
let isSending = false;

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  });
}

function autoResizeTextarea() {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 176)}px`;
}

function syncSendButton() {
  sendButton.disabled = isSending || promptInput.value.trim().length === 0;
}

function syncMedicalBadge() {
  const selectedOption = modelSelect.options[modelSelect.selectedIndex];
  const isMedicalModel = selectedOption?.dataset.medical === "true";
  medicalModelBadge.classList.toggle("hidden", !isMedicalModel);
  medicalModelBadge.classList.toggle("inline-flex", isMedicalModel);
}

function appendMessage(role, content, loading = false) {
  const row = document.createElement("div");
  row.className = role === "user" ? "flex justify-end" : "flex justify-start";

  const message = document.createElement(role === "user" ? "div" : "article");
  message.textContent = content;
  message.className =
    role === "user"
      ? "max-w-[82%] whitespace-pre-wrap rounded-[24px] bg-zinc-100 px-4 py-3 text-[15px] leading-6 text-zinc-950"
      : "w-full whitespace-pre-wrap text-[15px] leading-7 text-zinc-800";

  if (loading) {
    message.classList.add("animate-pulse", "text-zinc-400");
  }

  row.appendChild(message);
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function appendFeedbackComposer(meta) {
  if (!meta?.requestId) {
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "flex justify-start";

  const panel = document.createElement("div");
  panel.className =
    "mt-2 flex w-full max-w-2xl items-center gap-2 rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2";

  const label = document.createElement("span");
  label.className = "text-xs text-zinc-600";
  label.textContent = "Feedback:";

  const commentInput = document.createElement("input");
  commentInput.type = "text";
  commentInput.maxLength = 240;
  commentInput.placeholder = "Optional comment";
  commentInput.className =
    "h-8 flex-1 rounded-lg border border-zinc-200 bg-white px-2 text-xs text-zinc-800 outline-none focus:border-teal-600";

  const upButton = document.createElement("button");
  upButton.type = "button";
  upButton.textContent = "Correct";
  upButton.className =
    "h-8 rounded-lg border border-teal-200 bg-teal-50 px-2 text-xs font-medium text-teal-700 hover:bg-teal-100";

  const downButton = document.createElement("button");
  downButton.type = "button";
  downButton.textContent = "Incorrect";
  downButton.className =
    "h-8 rounded-lg border border-rose-200 bg-rose-50 px-2 text-xs font-medium text-rose-700 hover:bg-rose-100";

  const status = document.createElement("span");
  status.className = "text-xs text-zinc-500";
  status.textContent = "";

  async function sendFeedback(rating) {
    upButton.disabled = true;
    downButton.disabled = true;
    commentInput.disabled = true;
    status.textContent = "Saving...";

    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: meta.requestId,
          rating,
          comment: commentInput.value.trim(),
          model: meta.model,
          prompt_version: meta.promptVersion,
        }),
      });

      if (!response.ok) {
        throw new Error("Feedback could not be saved.");
      }
      status.textContent = "Saved";
    } catch (error) {
      status.textContent = error.message || "Save failed";
      upButton.disabled = false;
      downButton.disabled = false;
      commentInput.disabled = false;
    }
  }

  upButton.addEventListener("click", () => sendFeedback("up"));
  downButton.addEventListener("click", () => sendFeedback("down"));

  panel.appendChild(label);
  panel.appendChild(commentInput);
  panel.appendChild(upButton);
  panel.appendChild(downButton);
  panel.appendChild(status);
  wrapper.appendChild(panel);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

async function sendMessage(content) {
  isSending = true;
  syncSendButton();

  messages.push({ role: "user", content });
  appendMessage("user", content);
  const loadingRow = appendMessage("assistant", "...", true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelSelect.value,
        messages,
        temperature: 0.2,
      }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "The assistant is unavailable.");
    }

    const assistantMessage = data.message?.content || "I could not generate a response.";
    loadingRow.remove();
    messages.push({ role: "assistant", content: assistantMessage });
    appendMessage("assistant", assistantMessage);
    appendFeedbackComposer({
      requestId: data.request_id,
      model: data.model || modelSelect.value,
      promptVersion: data.prompt_version || "",
    });
  } catch (error) {
    loadingRow.remove();
    appendMessage("assistant", error.message || "Something went wrong.");
  } finally {
    isSending = false;
    syncSendButton();
    promptInput.focus();
  }
}

promptInput.addEventListener("input", () => {
  autoResizeTextarea();
  syncSendButton();
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

modelSelect.addEventListener("change", syncMedicalBadge);

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = promptInput.value.trim();
  if (!content || isSending) {
    return;
  }

  promptInput.value = "";
  autoResizeTextarea();
  await sendMessage(content);
});

autoResizeTextarea();
syncMedicalBadge();
syncSendButton();
