const vlSocketUrl = 'ws://localhost:8000/ws/stream';



const ttsSocketUrl = 'ws://localhost:8001/ws/tts';



const videoInput = document.getElementById('videoInput');



const startBtn = document.getElementById('startStream');



const stopBtn = document.getElementById('stopStream');



const resetBtn = document.getElementById('resetStream');



const videoPreview = document.getElementById('videoPreview');



const statusMessages = document.getElementById('statusMessages');



const transcriptOutput = document.getElementById('transcriptOutput');



videoPreview.addEventListener('pause', () => stopActiveAudio());



videoPreview.addEventListener('ended', () => stopActiveAudio());



videoPreview.addEventListener('seeked', handleSegmentSeek);



const canvas = document.createElement('canvas');



canvas.width = FRAME_WIDTH;



canvas.height = FRAME_HEIGHT;



const ctx = canvas.getContext('2d');



const captureVideo = document.createElement('video');



captureVideo.muted = true;



captureVideo.playsInline = true;



captureVideo.style.display = 'none';



captureVideo.crossOrigin = 'anonymous';



document.body.appendChild(captureVideo);



const SEGMENT_DURATION_MS = 20000;



const CAPTURE_INTERVAL_MS = 500;



const SEGMENT_LOOKUP_TOLERANCE = 0.6;



const SEGMENT_MERGE_MAX_DURATION_MS = 0;



let videoSocket = null;



let ttsSocket = null;



let captureTimer = null;



let audioCtx = null;



let pendingTexts = [];



let pendingSegmentMetas = [];



let sentSegmentMetas = [];



let ttsReady = false;



let transcriptText = '';



let segmentFrames = [];



let segmentStartTime = 0;



let segmentQueue = [];



let segmentInFlight = null;



let nextSegmentId = 1;



let activeVideoUrl = null;



const segmentCache = new Map();



let currentAudioSource = null;



let currentSegmentMeta = null;



let suppressSegmentSeek = false;



let playbackStarted = false;

const TTS_RATE_MAX = 240

const audioPlaybackQueue = [];



function appendStatus(message) {



  const entry = document.createElement('div');



  entry.textContent = new Date().toLocaleTimeString() + ' - ' + message;



  statusMessages.prepend(entry);



}



startBtn.addEventListener('click', () => {



  if (!videoInput.files?.[0]) {



    appendStatus('Choose a silent video before streaming.');



    return;



  }



  startStreaming();



});



stopBtn.addEventListener('click', () => stopStreaming());



resetBtn.addEventListener('click', () => {



  pendingTexts = [];



  transcriptText = '';



  transcriptOutput.textContent = '';



  if (videoSocket && videoSocket.readyState === WebSocket.OPEN) {



    videoSocket.send(JSON.stringify({ type: 'reset' }));



  }



  if (ttsSocket && ttsSocket.readyState === WebSocket.OPEN) {



    ttsSocket.send(JSON.stringify({ type: 'reset' }));



  }



  appendStatus('Session reset requested.');



});



function startStreaming() {



  stopStreaming();



  transcriptText = '';



  transcriptOutput.textContent = '';



  ttsReady = false;



  pendingTexts = [];



  pendingSegmentMetas = [];



  sentSegmentMetas = [];



  segmentQueue = [];



  segmentFrames = [];



  segmentStartTime = 0;



  segmentInFlight = null;



  nextSegmentId = 1;



  segmentCache.clear();



  const objectUrl = URL.createObjectURL(videoInput.files[0]);



  activeVideoUrl = objectUrl;



  videoPreview.src = objectUrl;



  videoPreview.muted = true;



  videoPreview.currentTime = 0;



  videoPreview.pause();



  captureVideo.src = objectUrl;



  captureVideo.load();



  captureVideo.play().catch(() => {});



  audioCtx = audioCtx || new AudioContext();



  openVideoSocket();



  openTtsSocket();



}



function stopStreaming() {



  if (captureTimer) {



    clearInterval(captureTimer);



    captureTimer = null;



  }



  if (videoSocket) {



    videoSocket.close();



    videoSocket = null;



  }



  if (ttsSocket) {



    ttsSocket.close();



    ttsSocket = null;



  }



  stopActiveAudio();



  playbackStarted = false;



  audioPlaybackQueue.length = 0;



  segmentCache.clear();



  ttsReady = false;



  segmentQueue = [];



  segmentFrames = [];



  segmentStartTime = 0;



  segmentInFlight = null;



  nextSegmentId = 1;



  pendingTexts = [];



  pendingSegmentMetas = [];



  sentSegmentMetas = [];



  if (activeVideoUrl) {



    URL.revokeObjectURL(activeVideoUrl);



    activeVideoUrl = null;



  }



  captureVideo.pause();



  captureVideo.removeAttribute('src');



  captureVideo.load();



  if (videoPreview.src) {



    videoPreview.pause();



    videoPreview.src = '';



  }



  appendStatus('Stopped streaming.');



}



function openVideoSocket() {



  videoSocket = new WebSocket(vlSocketUrl);



  videoSocket.addEventListener('open', () => {



    appendStatus('Connected to /ws/stream.');



    startFrameCapture();



  });



  videoSocket.addEventListener('message', handleVideoMessage);



  videoSocket.addEventListener('close', () => appendStatus('Video socket closed.'));



  videoSocket.addEventListener('error', () => appendStatus('Video socket error.'));



}



function applyVariation(text) {



  return text;



}



function cleanTranscriptText(text) {



  return text;



}

function splitTextForPacing(text, durationMs) {
  const normalized = text ? text.trim() : "";
  if (!normalized) {
    return [];
  }

  const words = normalized.split(/\s+/).filter(Boolean);
  if (!words.length) {
    return [];
  }

  const totalDuration = durationMs != null ? durationMs : SEGMENT_DURATION_MS;
  const maxWordsPerChunk = Math.max(
    1,
    Math.floor((TTS_RATE_MAX * (totalDuration / 1000)) / 60)
  );

  if (maxWordsPerChunk >= words.length) {
    return [{ text: normalized, durationMs: totalDuration }];
  }

  const chunks = [];
  for (let start = 0; start < words.length; start += maxWordsPerChunk) {
    const section = words.slice(start, start + maxWordsPerChunk);
    chunks.push({ text: section.join(" "), wordCount: section.length });
  }

  const totalWords = words.length;
  let accumulated = 0;
  for (const chunk of chunks) {
    const portion = chunk.wordCount / totalWords;
    chunk.durationMs = Math.max(100, Math.round(totalDuration * portion));
    accumulated += chunk.durationMs;
  }

  const remainder = totalDuration - accumulated;
  if (chunks.length && remainder !== 0) {
    chunks[chunks.length - 1].durationMs += remainder;
  }

  return chunks;
}



function handleVideoMessage(event) {



  try {



    const message = JSON.parse(event.data);



    if (message.type === 'chunk') {



      const text = message.text ?? '';



      appendStatus('VL chunk: ' + (text.slice(0, 60) || ''));



      transcriptText += text + ' ';



      transcriptOutput.textContent = transcriptText.trim();



      if (segmentInFlight) {



        segmentInFlight.text += cleanTranscriptText(text) + ' ';



      }



    } else if (message.type === 'final') {



      const finalText = (message.text ?? segmentInFlight?.text ?? '').trim();



      appendStatus('VL final text received' + (finalText ? ': ' + finalText : ''));



      if (segmentInFlight) {


        const textToSend = applyVariation(cleanTranscriptText(finalText) || 'Short football commentary.');
        const durationForSegment = segmentInFlight.durationMs ?? SEGMENT_DURATION_MS;
        const paceChunks = splitTextForPacing(textToSend, durationForSegment);
        const chunksToQueue = paceChunks.length ? paceChunks : [{ text: textToSend, durationMs: durationForSegment }];
        for (const chunk of chunksToQueue) {
          queueTextChunk(chunk.text, {
            startTime: segmentInFlight.startTime,
            segmentId: segmentInFlight.id,
            durationMs: chunk.durationMs,
          });
        }



        appendStatus(`Segment ${segmentInFlight.id} text ready for playback.`);



        segmentInFlight = null;



      }



      processSegmentQueue();



    }



  } catch (error) {



    appendStatus('Failed to parse VL message: ' + error.message);



  }



}



function queueTextChunk(text, meta = {}) {



  if (!text) {



    return;



  }



  pendingTexts.push(text);



  pendingSegmentMetas.push(meta);



  flushPendingTexts();



}



function flushPendingTexts() {

  if (!ttsReady) {

    return;

  }

  if (!ttsSocket) {

    return;

  }

  if (ttsSocket.readyState !== WebSocket.OPEN) {

    return;

  }

  while (pendingTexts.length) {

    const chunk = pendingTexts.shift();

    let meta = pendingSegmentMetas.shift();
    if (!meta) {

      meta = {};

    }

    sentSegmentMetas.push(meta);

    ttsSocket.send(JSON.stringify({ type: 'text_chunk', text: chunk, duration_ms: typeof meta.durationMs === 'number' ? meta.durationMs : SEGMENT_DURATION_MS }));

  }

}

function enqueueSegment(frames, startTime, durationMs = SEGMENT_DURATION_MS) {



  if (!frames.length) {



    return;



  }



  const segment = {



    id: nextSegmentId++,



    frames,



    startTime,



    durationMs,



    text: '',



  };



  segmentQueue.push(segment);



  appendStatus(`Segment ${segment.id} queued (${frames.length} frames).`);



  maybeMergeSegments();



  processSegmentQueue();



}



function maybeMergeSegments() {



  while (segmentQueue.length >= 2) {



    const [first, second] = segmentQueue;



    const totalDuration = (first.durationMs ?? SEGMENT_DURATION_MS) + (second.durationMs ?? SEGMENT_DURATION_MS);



    if (totalDuration > SEGMENT_MERGE_MAX_DURATION_MS) {



      break;



    }



    const merged = {



      id: first.id,



      frames: [...first.frames, ...second.frames],



      startTime: first.startTime,



      durationMs: totalDuration,



      text: '',



    };



    segmentQueue.splice(0, 2, merged);



    appendStatus(`Merged segments ${first.id} and ${second.id} into ${merged.id}.`);



  }



}



function processSegmentQueue() {



  if (segmentInFlight || !segmentQueue.length || !videoSocket || videoSocket.readyState !== WebSocket.OPEN) {



    return;



  }



  const segment = segmentQueue.shift();



  segmentInFlight = segment;



  appendStatus(`Sending segment ${segment.id} for transcription.`);



  videoSocket.send(JSON.stringify({ type: 'reset' }));



  for (const frame of segment.frames) {



    videoSocket.send(JSON.stringify({ type: 'frame', data: frame }));



  }



  videoSocket.send(



    JSON.stringify({



      type: 'generate',



      max_new_tokens: 40,



    })



  );



}



function openTtsSocket() {



  ttsSocket = new WebSocket(ttsSocketUrl);



  ttsSocket.addEventListener('open', () => {



    appendStatus('Connected to /ws/tts.');



    ttsReady = true;



    flushPendingTexts();



  });



  ttsSocket.addEventListener('message', handleTtsMessage);



  ttsSocket.addEventListener('close', () => appendStatus('TTS socket closed.'));



  ttsSocket.addEventListener('error', () => appendStatus('TTS socket error.'));



}



async function handleTtsMessage(event) {



  try {



    const message = JSON.parse(event.data);



    appendStatus(



      'TTS msg type=' + message.type +



        ' readyState=' +



        (ttsSocket?.readyState ?? 'n/a') +



        ' base64_len=' +



        (message.audio?.length ?? 0)



    );



    if (message.type === 'reference_ack') {



      ttsReady = true;



      appendStatus('TTS reference acknowledged.');



      flushPendingTexts();



    } else if (message.type === 'audio_chunk') {



      const duration = message.duration_ms ?? 'unknown';



      appendStatus(



        'Audio chunk received: duration=' + duration + ' ms base64_len=' + (message.audio?.length ?? 0)



      );



      const meta = sentSegmentMetas.shift() || {};



      cacheSegmentAudio(



        meta,



        message.audio,



        typeof message.duration_ms === 'number' ? message.duration_ms : SEGMENT_DURATION_MS



      );



      await playSegmentAudio(message.audio, meta);



    } else if (message.type === 'error') {



      appendStatus('TTS error: ' + (message.error || message.message || 'Unknown error'));



    }



  } catch (error) {



    appendStatus('Failed to parse TTS message: ' + error.message);



  }



}



function cacheSegmentAudio(meta, audio, durationMs) {



  if (!meta.segmentId || !audio) {



    return;



  }



  const startTime = typeof meta.startTime === 'number' ? meta.startTime : 0;



  const durationMsValue = typeof durationMs === 'number' ? durationMs : SEGMENT_DURATION_MS;



  const durationSec = durationMsValue / 1000;



  segmentCache.set(meta.segmentId, {



    segmentId: meta.segmentId,



    startTime,



    durationMs: durationMsValue,



    endTime: startTime + durationSec,



    audio,



  });



}



function findSegmentForTime(time) {



  for (const entry of segmentCache.values()) {



    if (typeof entry.endTime === 'number') {



      if (time >= entry.startTime && time < entry.endTime) {



        return entry;



      }



    } else {



      const durationSec = (entry.durationMs ?? SEGMENT_DURATION_MS) / 1000;



      if (time >= entry.startTime && time < entry.startTime + durationSec) {



        return entry;



      }



    }



  }



  let bestMatch = null;



  let bestDiff = Infinity;



  for (const entry of segmentCache.values()) {



    const diff = Math.abs(time - entry.startTime);



    if (diff < bestDiff) {



      bestDiff = diff;



      bestMatch = entry;



    }



  }



  if (bestMatch && bestDiff <= SEGMENT_LOOKUP_TOLERANCE) {



    return bestMatch;



  }



  return null;



}



function stopActiveAudio(flushOnStop = false) {



  if (currentAudioSource) {



    try {



      currentAudioSource.stop();



    } catch (err) {



      /* ignore */



    }



    currentAudioSource.disconnect();



    currentAudioSource = null;



  }



  currentSegmentMeta = null;



  if (flushOnStop) {



    flushPendingTexts();



  }



}



function handleSegmentSeek(event) {



  if (suppressSegmentSeek) {



    suppressSegmentSeek = false;



    return;



  }



  stopActiveAudio();



  const time = videoPreview.currentTime;



  const cached = findSegmentForTime(time);



  if (cached) {



    appendStatus(`Replaying cached segment ${cached.segmentId} for ${time.toFixed(2)}s`);



    playSegmentAudio(cached.audio, cached, false);



  } else {



    appendStatus(`No cached audio for ${time.toFixed(2)}s`);



  }



}



async function playSegmentAudio(base64Audio, meta = {}, shouldFlushOnEnd = true) {



  if (!base64Audio) {



    return;



  }



  stopActiveAudio();



  if (!audioCtx) {



    audioCtx = new AudioContext();



  }



  const len = Math.max(0, base64Audio?.length ?? 0);



  const startTime = typeof meta.startTime === 'number' ? meta.startTime : videoPreview.currentTime;



  appendStatus('Decoding audio chunk (state=' + audioCtx.state + ') len=' + len);



  const binary = atob(base64Audio ?? '');



  const buffer = new Uint8Array(binary.length);



  for (let i = 0; i < binary.length; i += 1) {



    buffer[i] = binary.charCodeAt(i);



  }



  const audioBuffer = await audioCtx.decodeAudioData(buffer.buffer);



  const source = audioCtx.createBufferSource();



  source.buffer = audioBuffer;



  source.connect(audioCtx.destination);



  source.addEventListener('ended', () => {



    appendStatus('Audio chunk playback finished');



    currentAudioSource = null;



    currentSegmentMeta = null;



    if (shouldFlushOnEnd) {



      flushPendingTexts();



    }



    videoPreview.pause();



  });



  currentAudioSource = source;



  currentSegmentMeta = meta;



  suppressSegmentSeek = true;



  videoPreview.currentTime = startTime;



  videoPreview.muted = true;



  try {



    await videoPreview.play();



  } catch (err) {



    appendStatus('Video playback blocked: ' + err.message);



  }



  source.start();



}



function startFrameCapture() {



  if (captureTimer) {



    clearInterval(captureTimer);



  }



  segmentFrames = [];



  segmentStartTime = 0;



  captureTimer = setInterval(() => {



    if (!captureVideo || captureVideo.readyState < 2 || !videoSocket || videoSocket.readyState !== WebSocket.OPEN) {



      return;



    }



    if (captureVideo.ended) {



      if (segmentFrames.length) {



        const elapsedMsEnd = segmentStartTime



          ? (captureVideo.currentTime - segmentStartTime) * 1000



          : segmentFrames.length * CAPTURE_INTERVAL_MS;



        enqueueSegment(



          segmentFrames.slice(),



          segmentStartTime || captureVideo.currentTime,



          Math.max(elapsedMsEnd, SEGMENT_DURATION_MS)



        );



        segmentFrames = [];



        segmentStartTime = 0;



      }



      clearInterval(captureTimer);



      captureTimer = null;



      return;



    }



    ctx.drawImage(captureVideo, 0, 0, canvas.width, canvas.height);



    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);



    if (!segmentStartTime) {



      segmentStartTime = captureVideo.currentTime;



    }



    segmentFrames.push(dataUrl);



    const elapsedMs = (captureVideo.currentTime - segmentStartTime) * 1000;



    if (elapsedMs >= SEGMENT_DURATION_MS) {



      enqueueSegment(



        segmentFrames.slice(),



        segmentStartTime,



        Math.max(elapsedMs, SEGMENT_DURATION_MS)



      );



      segmentFrames = [];



      segmentStartTime = 0;



    }



  }, CAPTURE_INTERVAL_MS);



}



window.addEventListener('beforeunload', () => {



  stopStreaming();



});



