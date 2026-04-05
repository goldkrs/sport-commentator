const videoApiBase = "http://localhost:8000";
const ttsApiBase = "http://localhost:8001";
const COMMENTARY_INSTRUCTION =
  "You are providing short football play-by-play commentary. Stay focused on the action, keep it crisp, and do not describe the scene or frame itself.";
const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 360;

const videoInput = document.getElementById("videoInput");
const startButton = document.getElementById("startStream");
const stopButton = document.getElementById("stopStream");
const resetButton = document.getElementById("resetStream");
const videoPreview = document.getElementById("videoPreview");
const statusMessages = document.getElementById("statusMessages");
const transcriptOutput = document.getElementById("transcriptOutput");
const loadingOverlay = document.getElementById("loadingOverlay");
const overlayMessage = loadingOverlay?.querySelector(".loading-overlay__message");

const canvas = document.createElement("canvas");
canvas.width = FRAME_WIDTH;
canvas.height = FRAME_HEIGHT;
const ctx = canvas.getContext("2d");

const captureVideo = document.createElement("video");
captureVideo.muted = true;
captureVideo.playsInline = true;
captureVideo.crossOrigin = "anonymous";
captureVideo.style.display = "none";
document.body.appendChild(captureVideo);

let captureUrl = null;
let captureAbortHandler = null;
let processingAbortController = null;
let isProcessing = false;
let generatedVideoUrl = null;
let captureLoopId = null;
let sourceFrameRate = 25;

startButton?.addEventListener("click", () => {
  if (isProcessing) {
    return;
  }
  const file = videoInput?.files?.[0];
  if (!file) {
    appendStatus("Choose a silent video before processing.");
    return;
  }
  processVideo(file);
});

stopButton?.addEventListener("click", () => stopProcessing());
resetButton?.addEventListener("click", () => {
  stopProcessing();
  transcriptOutput.textContent = "";
  resetPreview();
  appendStatus("Session reset.");
});

window.addEventListener("beforeunload", () => stopProcessing());

appendStatus("Ready to process a silent video.");
setProcessingState(false);
hideOverlay();

function appendStatus(message) {
  const entry = document.createElement("div");
  entry.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
  statusMessages?.prepend(entry);
}

function showOverlay(message) {
  if (overlayMessage) {
    overlayMessage.textContent = message;
  }
  loadingOverlay?.classList.remove("hidden");
}

function hideOverlay() {
  loadingOverlay?.classList.add("hidden");
}

function setProcessingState(active) {
  isProcessing = active;
  if (startButton) {
    startButton.disabled = active;
  }
  if (stopButton) {
    stopButton.disabled = !active;
  }
}

function resetPreview() {
  videoPreview.pause();
  videoPreview.removeAttribute("src");
  if (generatedVideoUrl) {
    URL.revokeObjectURL(generatedVideoUrl);
    generatedVideoUrl = null;
  }
}

function stopProcessing(silent = false) {
  if (captureAbortHandler) {
    captureAbortHandler(new Error("Processing interrupted by user."));
  }
  captureAbortHandler = null;
  if (captureLoopId) {
    cancelAnimationFrame(captureLoopId);
    captureLoopId = null;
  }
  if (processingAbortController) {
    processingAbortController.abort();
    processingAbortController = null;
  }
  captureVideo.pause();
  captureVideo.removeAttribute("src");
  captureVideo.load();
  hideOverlay();
  setProcessingState(false);
  if (!silent) {
    appendStatus("Processing stopped.");
  }
}

function setProcessingController(controller) {
  if (processingAbortController) {
    processingAbortController.abort();
  }
  processingAbortController = controller;
}

function captureFrames(file) {
  return new Promise((resolve, reject) => {
    const frames = [];
    let settled = false;

    function cleanup() {
      if (captureLoopId) {
        cancelAnimationFrame(captureLoopId);
        captureLoopId = null;
      }
      captureVideo.pause();
      captureVideo.removeAttribute("src");
      captureVideo.load();
      if (captureUrl) {
        URL.revokeObjectURL(captureUrl);
        captureUrl = null;
      }
      captureAbortHandler = null;
    }

    function settle(value, isError) {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      if (isError) {
        reject(value);
      } else {
        resolve(frames.length ? frames : value);
      }
    }

    captureAbortHandler = (reason) => {
      const err = reason instanceof Error ? reason : new Error(reason || "Capture aborted.");
      settle(err, true);
    };

    captureUrl = URL.createObjectURL(file);
    captureVideo.src = captureUrl;
    captureVideo.currentTime = 0;

    const snapshot = () => {
      ctx.drawImage(captureVideo, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
      frames.push(canvas.toDataURL("image/jpeg", 0.8));
    };

    const captureLoop = () => {
      if (captureVideo.paused || captureVideo.ended) {
        settle(frames, false);
        return;
      }
      snapshot();
      captureLoopId = requestAnimationFrame(captureLoop);
    };

    captureVideo.addEventListener(
      "loadeddata",
      () => {
        if (captureVideo.captureStream) {
          const tracks = captureVideo.captureStream().getVideoTracks();
          if (tracks.length) {
            const settings = tracks[0].getSettings();
            sourceFrameRate = settings.frameRate || sourceFrameRate;
            tracks.forEach((track) => track.stop());
          }
        }
        captureLoopId = requestAnimationFrame(captureLoop);
      },
      { once: true }
    );
    captureVideo.addEventListener("ended", () => settle(frames, false));
    captureVideo.addEventListener("error", () => settle(new Error("Unable to load video."), true));

    captureVideo
      .play()
      .catch((err) => settle(err, true));
  });
}

function processVideo(file) {
  stopProcessing(true);
  setProcessingState(true);
  showOverlay("Capturing frames from video...");
  captureFrames(file)
    .then(async (frames) => {
      appendStatus(`Captured ${frames.length} frame(s).`);
      if (!frames.length) {
        throw new Error("No frames captured from the provided video.");
      }
      showOverlay("Generating transcript...");
      const transcript = await requestVideoTranscript(frames);
      const cleaned = (transcript?.text || "").trim();
      transcriptOutput.textContent = cleaned || "No commentary generated.";
      showOverlay("Rendering narrated video...");
      const narratedBlob = await requestNarratedVideo(frames, cleaned, sourceFrameRate);
      resetPreview();
      generatedVideoUrl = URL.createObjectURL(narratedBlob);
      videoPreview.src = generatedVideoUrl;
      videoPreview.muted = false;
      await videoPreview.play().catch(() => {});
      appendStatus("Narrated video ready.");
    })
    .catch((error) => {
      if (error?.name === "AbortError") {
        appendStatus("Processing cancelled.");
      } else {
        appendStatus(`Processing failed: ${error?.message || error}`);
      }
    })
    .finally(() => {
      hideOverlay();
      setProcessingState(false);
    });
}

function requestVideoTranscript(frames) {
  const controller = new AbortController();
  setProcessingController(controller);
  const form = new FormData();
  form.append("prompt", COMMENTARY_INSTRUCTION);
  form.append("max_new_tokens", "30");
  frames.forEach((frame, index) => {
    form.append("frames", dataUrlToFile(frame, `frame_${index + 1}.jpg`));
  });
  return fetch(`${videoApiBase}/generate_paragraph`, {
    method: "POST",
    body: form,
    signal: controller.signal,
  })
    .then((response) => {
      if (!response.ok) {
        return response.text().then((text) => {
          throw new Error(text || `Transcript request failed (${response.status})`);
        });
      }
      return response.json();
    })
    .finally(() => {
      if (processingAbortController === controller) {
        processingAbortController = null;
      }
    });
}

async function requestNarratedVideo(frames, transcriptText, frameRate) {
  if (!transcriptText?.trim()) {
    throw new Error("Transcript text is required to generate the narrated video.");
  }

  const controller = new AbortController();
  setProcessingController(controller);
  const form = new FormData();
  form.append("generated_text", transcriptText.trim());
  form.append("language", "English");
  form.append("frame_rate", `${frameRate}`);
  frames.forEach((frame, index) => {
    form.append("frames", dataUrlToFile(frame, `frame_${index + 1}.jpg`));
  });
  return fetch(`${ttsApiBase}/narrated_video`, {
    method: "POST",
    body: form,
    signal: controller.signal,
  })
    .then((response) => {
      if (!response.ok) {
        return response.text().then((text) => {
          throw new Error(text || `Narrated video request failed (${response.status})`);
        });
      }
      return response.blob();
    })
    .finally(() => {
      if (processingAbortController === controller) {
        processingAbortController = null;
      }
    });
}

function dataUrlToFile(dataUrl, filename) {
  const [meta, base64] = dataUrl.split(",");
  const mime = /data:(.*?);/.exec(meta)?.[1] || "image/jpeg";
  const binary = atob(base64 || "");
  const buffer = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    buffer[i] = binary.charCodeAt(i);
  }
  return new File([buffer], filename, { type: mime });
}
