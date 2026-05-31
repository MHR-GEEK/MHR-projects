const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const form = $("#analyze-form");
const imageInput = $("#image-input");
const dropZone = $("#drop-zone");
const previewShell = $("#preview-shell");
const previewImage = $("#preview-image");
const fileChip = $("#file-chip");
const toast = $("#toast");
const analyzeButton = $(".analyze-button");
const chatDrawer = $("#chat-drawer");
const chatLog = $("#chat-log");

let toastTimer;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 3600);
}

function setText(id, value) {
  const node = $(id);
  node.textContent = value || "Unclear";
}

function renderList(container, items, fallback, mapper) {
  container.innerHTML = "";
  if (!items || !items.length) {
    const empty = document.createElement(container.tagName === "OL" ? "li" : "span");
    empty.className = "muted";
    empty.textContent = fallback;
    container.append(empty);
    return;
  }
  items.forEach((item) => container.append(mapper(item)));
}

function renderResult(result) {
  setText("#skin-type", result.skin_type);
  setText("#sensitivity", result.sensitivity);
  setText("#psl-score", result.psl_score);
  setText("#ratio-score", result.image_ratio_score);
  $("#professional-summary").textContent =
    result.professional_summary ||
    "The scan is ready. Review the details below as educational skincare guidance, not a medical diagnosis.";

  renderList($("#skin-info-list"), result.skin_information, "No skin information returned.", (note) => {
    const item = document.createElement("li");
    item.textContent = note;
    return item;
  });

  renderList($("#concerns-list"), result.concerns, "No concerns returned.", (concern) => {
    const tag = document.createElement("span");
    tag.textContent = concern;
    return tag;
  });

  renderList($("#routine-list"), result.routine, "No routine generated.", (step) => {
    const item = document.createElement("li");
    const label = step.step ? `${step.step}: ` : "";
    item.textContent = `${label}${step.recommendation || "No recommendation"}`;
    return item;
  });

  renderList($("#care-plan-list"), result.care_plan, "No care advice returned.", (note) => {
    const item = document.createElement("li");
    item.textContent = note;
    return item;
  });

  renderList($("#proportion-list"), result.proportion_notes, "No photo notes returned.", (note) => {
    const item = document.createElement("li");
    item.textContent = note;
    return item;
  });

  $("#analysis-notes").textContent = result.notes || "";
  showToast(result.ai_available ? "Analysis complete." : result.notes || "AI is not configured.");
}

function loadPreview(file) {
  if (!file) return;
  fileChip.textContent = file.name;
  previewImage.src = URL.createObjectURL(file);
  previewShell.hidden = false;
  dropZone.style.display = "none";
}

imageInput?.addEventListener("change", () => loadPreview(imageInput.files[0]));
$("#change-image")?.addEventListener("click", () => imageInput.click());

$("#support-desk-link")?.addEventListener("click", (event) => {
  event.preventDefault();
  const address = ["ks8257", "proton.me"].join("@");
  window.location.href = `mailto:${address}`;
});

function compressImage(file, maxSize = 960, quality = 0.74) {
  return new Promise((resolve) => {
    if (!file.type.startsWith("image/")) {
      resolve(file);
      return;
    }

    const image = new Image();
    image.onload = () => {
      const scale = Math.min(1, maxSize / Math.max(image.width, image.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.width * scale));
      canvas.height = Math.max(1, Math.round(image.height * scale));
      const context = canvas.getContext("2d", { alpha: false });
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => resolve(blob || file), "image/jpeg", quality);
    };
    image.onerror = () => resolve(file);
    image.src = URL.createObjectURL(file);
  });
}

function blobToBase64(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = () => resolve("");
    reader.readAsDataURL(blob);
  });
}

["dragenter", "dragover"].forEach((eventName) => {
  dropZone?.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone?.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone?.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  loadPreview(file);
});

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files.length) {
    showToast("Choose a face image first.");
    return;
  }

  analyzeButton.classList.add("is-loading");
  analyzeButton.querySelector("span").textContent = "Fast scanning";
  setText("#skin-type", "Scanning");
  setText("#sensitivity", "Scanning");
  setText("#psl-score", "Scanning");
  setText("#ratio-score", "Scanning");
  $("#professional-summary").textContent = "Reading visible skin texture, tone, hydration clues, and photo quality.";
  renderList($("#skin-info-list"), ["Preparing an easy skin information summary."], "", (note) => {
    const item = document.createElement("li");
    item.textContent = note;
    return item;
  });
  renderList($("#care-plan-list"), ["Reviewing image context for follow-up chat."], "", (note) => {
    const item = document.createElement("li");
    item.textContent = note;
    return item;
  });

  try {
    const optimized = await compressImage(imageInput.files[0]);
    const data = new FormData();
    data.append("image", optimized, "fast-scan.jpg");
    const response = await fetch("/analyze", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.message || "Analysis failed.");
    }
    renderResult(payload.result);
    $("#insights").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message);
  } finally {
    analyzeButton.classList.remove("is-loading");
    analyzeButton.querySelector("span").textContent = "Run AI scan";
  }
});

function openChat() {
  chatDrawer.classList.add("is-open");
  chatDrawer.setAttribute("aria-hidden", "false");
  $("#chat-input").focus();
}

function closeChat() {
  chatDrawer.classList.remove("is-open");
  chatDrawer.setAttribute("aria-hidden", "true");
}

$$("[data-open-chat]").forEach((button) => button.addEventListener("click", openChat));
$("#close-chat")?.addEventListener("click", closeChat);
chatDrawer?.addEventListener("click", (event) => {
  if (event.target === chatDrawer) closeChat();
});

function addMessage(text, className) {
  const message = document.createElement("p");
  message.className = className;
  message.textContent = text;
  chatLog.append(message);
  chatLog.scrollTop = chatLog.scrollHeight;
  return message;
}

$("#chat-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMessage(message, "user-message");
  const pending = addMessage("", "bot-message is-pending");

  try {
    let imageBase64 = "";
    if (imageInput.files.length) {
      const optimized = await compressImage(imageInput.files[0], 820, 0.68);
      imageBase64 = await blobToBase64(optimized);
    }
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, image_base64: imageBase64 }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.message || "Chat failed.");
    pending.classList.remove("is-pending");
    pending.textContent = payload.reply;
  } catch (error) {
    pending.classList.remove("is-pending");
    pending.textContent = error.message;
  }
});

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("is-visible");
    });
  },
  { threshold: 0.16 }
);

$$(".reveal").forEach((element, index) => {
  element.style.transitionDelay = `${Math.min(index * 45, 240)}ms`;
  observer.observe(element);
});

function updateScrollProgress() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollable > 0 ? window.scrollY / scrollable : 0;
  $("#scroll-progress").style.width = `${Math.round(progress * 100)}%`;
}

window.addEventListener("scroll", updateScrollProgress, { passive: true });
updateScrollProgress();

const canvas = $("#ambient-canvas");
const context = canvas.getContext("2d");
let points = [];

function resizeCanvas() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * ratio);
  canvas.height = Math.floor(window.innerHeight * ratio);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  points = Array.from({ length: Math.min(76, Math.floor(window.innerWidth / 16)) }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 0.28,
    vy: (Math.random() - 0.5) * 0.28,
  }));
}

function animateCanvas() {
  context.clearRect(0, 0, window.innerWidth, window.innerHeight);
  points.forEach((point, index) => {
    point.x += point.vx;
    point.y += point.vy;
    if (point.x < 0 || point.x > window.innerWidth) point.vx *= -1;
    if (point.y < 0 || point.y > window.innerHeight) point.vy *= -1;

    context.beginPath();
    context.arc(point.x, point.y, 1.4, 0, Math.PI * 2);
    context.fillStyle = "rgba(117, 246, 185, 0.55)";
    context.fill();

    for (let next = index + 1; next < points.length; next += 1) {
      const other = points[next];
      const distance = Math.hypot(point.x - other.x, point.y - other.y);
      if (distance < 126) {
        context.beginPath();
        context.moveTo(point.x, point.y);
        context.lineTo(other.x, other.y);
        context.strokeStyle = `rgba(71, 229, 255, ${0.13 * (1 - distance / 126)})`;
        context.stroke();
      }
    }
  });
  requestAnimationFrame(animateCanvas);
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
animateCanvas();

const welcomeScreen = $("#welcome-screen");
if (welcomeScreen) {
  const hasSeenWelcome = localStorage.getItem("aiDermaWelcomeSeen") === "yes";
  if (hasSeenWelcome) {
    welcomeScreen.remove();
  } else {
    localStorage.setItem("aiDermaWelcomeSeen", "yes");
    setTimeout(() => {
      welcomeScreen.classList.add("is-hidden");
      setTimeout(() => welcomeScreen.remove(), 520);
    }, 2000);
  }
}
