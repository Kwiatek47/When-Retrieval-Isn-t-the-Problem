const chatForm = document.querySelector("#chatForm");
const chatViewport = document.querySelector("#chatViewport");
const messagesEl = document.querySelector("#messages");
const medicalModelBadge = document.querySelector("#medicalModelBadge");
const modelSelect = document.querySelector("#modelSelect");
const promptInput = document.querySelector("#promptInput");
const sendButton = document.querySelector("#sendButton");

const messages = [];
let isSending = false;

function renderAssistantMarkdown(markdown) {
  const md =
    typeof marked !== "undefined"
      ? marked
      : typeof globalThis !== "undefined" && globalThis.marked
        ? globalThis.marked
        : undefined;
  const purify =
    typeof DOMPurify !== "undefined"
      ? DOMPurify
      : typeof globalThis !== "undefined" && globalThis.DOMPurify
        ? globalThis.DOMPurify
        : undefined;

  if (!md?.parse || !purify?.sanitize) {
    const fallback = document.createElement("div");
    fallback.textContent = markdown;
    return fallback.innerHTML;
  }

  const html = md.parse(markdown, { gfm: true, breaks: true });
  const safe = purify.sanitize(html);
  const tpl = document.createElement("template");
  tpl.innerHTML = safe;
  tpl.content.querySelectorAll('a[href^="http"]').forEach((a) => {
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener noreferrer");
  });
  return tpl.innerHTML;
}

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

function appendMessage(
  role,
  content,
  loading = false,
  citations = [],
  citationValidation = null,
  evidenceConflicts = null,
  answerQuality = null,
  assistantPlainText = false
) {
  const row = document.createElement("div");
  row.className = role === "user" ? "flex justify-end" : "flex justify-start";

  const message = document.createElement(role === "user" ? "div" : "article");
  message.className =
    role === "user"
      ? "max-w-[82%] whitespace-pre-wrap rounded-[24px] bg-zinc-100 px-4 py-3 text-[15px] leading-6 text-zinc-950"
      : "w-full text-[15px] leading-7 text-zinc-800";

  if (role === "user") {
    message.textContent = content;
  } else {
    const bodyEl = document.createElement("div");
    bodyEl.className = "assistant-md w-full";
    if (loading || assistantPlainText) {
      bodyEl.textContent = content;
    } else {
      bodyEl.innerHTML = renderAssistantMarkdown(content);
    }
    message.appendChild(bodyEl);
  }

  if (loading) {
    message.classList.add("animate-pulse", "text-zinc-400");
  }

  if (role === "assistant" && citations.length > 0) {
    const citationsEl = document.createElement("div");
    citationsEl.className = "mt-4 flex flex-wrap gap-2 text-xs text-zinc-500";

    citations.forEach((citation) => {
      const item = document.createElement("span");
      item.className = "rounded-full border border-zinc-200 px-3 py-1";
      item.textContent = `[${citation.id}] ${citation.title}`;
      item.title = citation.source;
      citationsEl.appendChild(item);
    });

    message.appendChild(citationsEl);
  }

  if (role === "assistant" && citationValidation && !citationValidation.passed) {
    const validationEl = document.createElement("div");
    validationEl.className = "mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800";
    validationEl.textContent = "Citation validation failed: the response is missing required inline citations or cites an unknown source.";
    message.appendChild(validationEl);
  }

  if (role === "assistant" && evidenceConflicts?.detected) {
    const conflictEl = document.createElement("div");
    conflictEl.className = "mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800";
    conflictEl.textContent = `Conflicting evidence detected across ${evidenceConflicts.conflict_count} source pair(s).`;
    message.appendChild(conflictEl);
  }

  if (role === "assistant" && answerQuality?.groundedness !== null && answerQuality?.groundedness !== undefined) {
    const qualityEl = document.createElement("div");
    const groundedness = Math.round(answerQuality.groundedness * 100);
    const hallucinationRate = Math.round((answerQuality.hallucination_rate || 0) * 100);
    const averageSimilarity = answerQuality.average_similarity === null || answerQuality.average_similarity === undefined
      ? null
      : Math.round(answerQuality.average_similarity * 100);
    qualityEl.className = "mt-3 flex flex-wrap gap-2 text-xs text-zinc-500";

    const groundednessEl = document.createElement("span");
    groundednessEl.className = "rounded-full border border-zinc-200 px-3 py-1";
    groundednessEl.textContent = `Groundedness ${groundedness}%`;
    qualityEl.appendChild(groundednessEl);

    const hallucinationEl = document.createElement("span");
    hallucinationEl.className = hallucinationRate > 0
      ? "rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-amber-800"
      : "rounded-full border border-zinc-200 px-3 py-1";
    hallucinationEl.textContent = `Unsupported ${hallucinationRate}%`;
    qualityEl.appendChild(hallucinationEl);

    if (averageSimilarity !== null) {
      const similarityEl = document.createElement("span");
      similarityEl.className = "rounded-full border border-zinc-200 px-3 py-1";
      similarityEl.textContent = `Similarity ${averageSimilarity}%`;
      qualityEl.appendChild(similarityEl);
    }

    message.appendChild(qualityEl);
  }

  row.appendChild(message);
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
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
    appendMessage(
      "assistant",
      assistantMessage,
      false,
      data.citations || [],
      data.citation_validation || null,
      data.evidence_conflicts || null,
      data.answer_quality || null
    );
  } catch (error) {
    loadingRow.remove();
    appendMessage("assistant", error.message || "Something went wrong.", false, [], null, null, null, true);
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
