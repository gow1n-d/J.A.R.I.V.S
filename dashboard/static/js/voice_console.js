/**
 * J.A.R.V.I.S — Voice Console (Iron Man style continuous conversation).
 * Features:
 *   - Continuous conversation mode: auto-listens after JARVIS finishes speaking
 *   - Live conversation transcript
 *   - Wake-word detection ("JARVIS")
 *   - Natural conversation flow with visual arc reactor feedback
 */

const API = '';
const LANG_STORAGE = 'jarvis_voice_lang';
const CONV_MODE_STORAGE = 'jarvis_conv_mode';

let speechRecognition = null;
let micListening = false;
let currentAudio = null;
let conversationMode = false;  // Continuous conversation toggle
let conversationHistory = [];  // Local transcript
let wakeWordEnabled = true;

function announce(msg) {
    const el = document.getElementById('srAnnounce');
    if (el && msg) el.textContent = msg;
}

function getSelectedBcp47() {
    const sel = document.getElementById('langSelect');
    if (sel && sel.value) return sel.value;
    const saved = localStorage.getItem(LANG_STORAGE);
    if (saved) return saved;
    return navigator.language || 'en-GB';
}

function bindLanguageSelect() {
    const langSel = document.getElementById('langSelect');
    if (!langSel) return;
    const saved = localStorage.getItem(LANG_STORAGE);
    const opts = [...langSel.options].map((o) => o.value);
    if (saved && opts.includes(saved)) langSel.value = saved;
    else if (navigator.language) {
        const nav = navigator.language;
        if (opts.includes(nav)) langSel.value = nav;
        else {
            const short = nav.split('-')[0];
            const match = opts.find((v) => v.toLowerCase().startsWith(short.toLowerCase() + '-'));
            if (match) langSel.value = match;
        }
    }
    localStorage.setItem(LANG_STORAGE, langSel.value);
    langSel.addEventListener('change', () => {
        localStorage.setItem(LANG_STORAGE, langSel.value);
        if (speechRecognition) speechRecognition.lang = langSel.value;
    });
}

/* ── Conversation Transcript ── */
function addTranscriptEntry(who, text) {
    const transcript = document.getElementById('conversationTranscript');
    if (!transcript) return;

    const entry = document.createElement('div');
    entry.className = `transcript-entry transcript-${who}`;

    const label = document.createElement('span');
    label.className = 'transcript-label';
    label.textContent = who === 'user' ? 'YOU' : 'J.A.R.V.I.S';

    const msg = document.createElement('span');
    msg.className = 'transcript-text';
    msg.textContent = text;

    const time = document.createElement('span');
    time.className = 'transcript-time';
    const now = new Date();
    time.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    entry.appendChild(label);
    entry.appendChild(msg);
    entry.appendChild(time);
    transcript.appendChild(entry);

    // Auto-scroll
    transcript.scrollTop = transcript.scrollHeight;

    // Animate entry
    entry.style.opacity = '0';
    entry.style.transform = 'translateY(10px)';
    requestAnimationFrame(() => {
        entry.style.transition = 'all 0.3s ease';
        entry.style.opacity = '1';
        entry.style.transform = 'translateY(0)';
    });

    // Store in history
    conversationHistory.push({ who, text, time: now.toISOString() });
}

function setArcState(state) {
    const el = document.getElementById('arcReactor');
    const label = document.getElementById('arcStateLabel');
    if (!el || !label) return;
    el.dataset.state = state;
    const map = {
        idle: 'STANDBY',
        listening: 'LISTENING',
        thinking: 'PROCESSING',
        speaking: 'SPEAKING'
    };
    label.textContent = map[state] || map.idle;
}

function getSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;
    const r = new SR();
    r.lang = getSelectedBcp47();
    r.continuous = false;
    r.interimResults = false;
    r.maxAlternatives = 1;
    return r;
}

function setMicButtonListening(listening) {
    const btn = document.getElementById('micBtn');
    if (!btn) return;
    if (listening) {
        btn.classList.add('active');
        btn.textContent = 'LISTENING…';
    } else {
        btn.classList.remove('active');
        btn.textContent = conversationMode ? 'CONVERSATION MODE' : 'VOICE COMMAND';
    }
}

function updateConvModeUI() {
    const toggle = document.getElementById('convModeToggle');
    const btn = document.getElementById('micBtn');
    if (toggle) toggle.checked = conversationMode;
    if (btn && !micListening) {
        btn.textContent = conversationMode ? 'CONVERSATION MODE' : 'VOICE COMMAND';
    }
    // Show/hide the indicator
    const indicator = document.getElementById('convModeIndicator');
    if (indicator) {
        indicator.style.display = conversationMode ? 'flex' : 'none';
    }
}

function stopMic() {
    micListening = false;
    setMicButtonListening(false);
    if (speechRecognition) {
        try {
            speechRecognition.stop();
        } catch (e) { /* ignore */ }
    }
    const arc = document.getElementById('arcReactor');
    if (arc && arc.dataset.state === 'listening') setArcState('idle');
}

function stripForSpeech(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .replace(/\*(.*?)\*/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/https?:\/\/\S+/g, '')
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 2500);
}

function speakFallback(text) {
    const plain = stripForSpeech(text);
    if (!plain) {
        setArcState('idle');
        autoListenAfterSpeak();
        return;
    }
    if (!window.speechSynthesis) {
        setArcState('idle');
        autoListenAfterSpeak();
        return;
    }
    try {
        window.speechSynthesis.resume();
    } catch (e) { /* ignore */ }

    const run = () => {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(plain);
        const lang = getSelectedBcp47();
        u.lang = lang;
        const voices = window.speechSynthesis.getVoices();
        const langLow = lang.toLowerCase().replace('_', '-');
        const primary = langLow.split('-')[0];
        const v =
            voices.find((vo) => (vo.lang || '').toLowerCase().replace('_', '-') === langLow) ||
            voices.find((vo) => (vo.lang || '').toLowerCase().startsWith(primary + '-')) ||
            voices.find((vo) => (vo.lang || '').toLowerCase().startsWith(primary));
        if (v) u.voice = v;
        setArcState('speaking');
        u.onend = () => {
            setArcState('idle');
            autoListenAfterSpeak();
        };
        u.onerror = () => {
            setArcState('idle');
            autoListenAfterSpeak();
        };
        window.speechSynthesis.speak(u);
    };

    let started = false;
    const runOnce = () => {
        if (started) return;
        started = true;
        run();
    };
    if (window.speechSynthesis.getVoices().length) {
        runOnce();
    } else {
        window.speechSynthesis.addEventListener('voiceschanged', runOnce, { once: true });
        window.speechSynthesis.getVoices();
        setTimeout(runOnce, 800);
    }
}

/* ── Auto-listen after JARVIS finishes (conversation mode) ── */
function autoListenAfterSpeak() {
    if (!conversationMode) return;
    // Small delay before auto-listening (feels natural)
    setTimeout(() => {
        if (conversationMode && !micListening) {
            startListening();
        }
    }, 600);
}

function startListening() {
    const btn = document.getElementById('micBtn');
    if (!btn || btn.disabled) return;
    if (micListening) return;

    try {
        // Create fresh recognition instance for reliability
        speechRecognition = getSpeechRecognition();
        if (!speechRecognition) return;
        wireRecognitionEvents();

        micListening = true;
        setMicButtonListening(true);
        speechRecognition.lang = getSelectedBcp47();
        speechRecognition.start();
    } catch (e) {
        micListening = false;
        setMicButtonListening(false);
        announce('Microphone could not start.');
        setArcState('idle');
    }
}

function toggleListening() {
    const btn = document.getElementById('micBtn');
    if (!btn || btn.disabled) return;
    if (micListening) {
        stopMic();
        // If in conversation mode and user manually stops, disable conv mode
        if (conversationMode) {
            conversationMode = false;
            localStorage.setItem(CONV_MODE_STORAGE, 'false');
            updateConvModeUI();
        }
        return;
    }
    startListening();
}

function wireRecognitionEvents() {
    if (!speechRecognition) return;

    speechRecognition.onstart = () => setArcState('listening');
    speechRecognition.onerror = (ev) => {
        if (ev.error === 'no-speech' || ev.error === 'aborted') {
            stopMic();
            setArcState('idle');
            // In conversation mode, try again after a pause
            if (conversationMode) {
                setTimeout(() => {
                    if (conversationMode && !micListening) startListening();
                }, 1500);
            }
            return;
        }
        announce('Voice error: ' + ev.error);
        stopMic();
        setArcState('idle');
    };
    speechRecognition.onend = () => {
        if (micListening) stopMic();
    };
    speechRecognition.onresult = (ev) => {
        const text = (ev.results[0] && ev.results[0][0] && ev.results[0][0].transcript || '').trim();
        stopMic();
        if (text) {
            addTranscriptEntry('user', text);
            sendVoiceCommand(text);
        } else {
            setArcState('idle');
            autoListenAfterSpeak();
        }
    };
}

function setupVoiceInput() {
    const btn = document.getElementById('micBtn');
    const arc = document.getElementById('arcReactor');
    speechRecognition = getSpeechRecognition();

    if (!speechRecognition) {
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'VOICE NOT SUPPORTED';
        }
        announce('Use Chrome or Edge for speech input.');
        return;
    }

    wireRecognitionEvents();

    if (window.speechSynthesis) {
        window.speechSynthesis.getVoices();
        window.speechSynthesis.addEventListener('voiceschanged', () => {
            window.speechSynthesis.getVoices();
        });
    }

    if (btn) btn.addEventListener('click', () => toggleListening());
    if (arc) {
        arc.addEventListener('click', () => toggleListening());
        arc.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleListening();
            }
        });
    }

    // Conversation mode toggle
    const convToggle = document.getElementById('convModeToggle');
    if (convToggle) {
        const saved = localStorage.getItem(CONV_MODE_STORAGE);
        conversationMode = saved === 'true';
        convToggle.checked = conversationMode;
        updateConvModeUI();

        convToggle.addEventListener('change', () => {
            conversationMode = convToggle.checked;
            localStorage.setItem(CONV_MODE_STORAGE, conversationMode ? 'true' : 'false');
            updateConvModeUI();
            if (conversationMode && !micListening) {
                startListening();
            }
        });
    }
}

async function playResponseAudio(base64Mp3) {
    const raw = typeof base64Mp3 === 'string' ? base64Mp3.trim() : '';
    if (raw.length < 64) return false;
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    setArcState('speaking');
    currentAudio = new Audio('data:audio/mpeg;base64,' + raw);
    return new Promise((resolve) => {
        let settled = false;
        const finish = (ok) => {
            if (settled) return;
            settled = true;
            currentAudio = null;
            setArcState('idle');
            if (ok) autoListenAfterSpeak();
            resolve(ok);
        };
        currentAudio.onended = () => finish(true);
        currentAudio.onerror = () => finish(false);
        currentAudio.play().then(() => {}).catch(() => finish(false));
    });
}

async function fetchLinkStatus() {
    const textEl = document.getElementById('statusText');
    const pulse = document.getElementById('statusPulse');
    try {
        const res = await fetch(`${API}/api/status`);
        if (!res.ok) throw new Error('offline');
        if (textEl) textEl.textContent = 'ONLINE';
        if (pulse) pulse.style.removeProperty('background');
    } catch (e) {
        if (textEl) textEl.textContent = 'NO LINK';
        if (pulse) pulse.style.background = '#ff6b35';
    }
}

async function sendVoiceCommand(cmd) {
    const text = (cmd || '').trim();
    if (!text) return;

    announce('You said: ' + text);
    setArcState('thinking');

    try {
        const res = await fetch(`${API}/api/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                command: text,
                voice: true,
                lang: getSelectedBcp47()
            })
        });
        const rawBody = await res.text();
        let d;
        try {
            d = JSON.parse(rawBody);
        } catch (parseErr) {
            const errMsg = 'I seem to be having trouble connecting to my cognitive systems, sir.';
            addTranscriptEntry('jarvis', errMsg);
            speakFallback(errMsg);
            return;
        }

        if (d.error) {
            const errText = typeof d.error === 'string' ? d.error : 'I encountered an issue, sir.';
            addTranscriptEntry('jarvis', errText);
            announce(d.error);
            speakFallback(errText);
            return;
        }

        const reply = (d.response || '').trim();
        if (!reply) {
            const noReply = "I've processed that, but I don't have anything specific to report back, sir.";
            addTranscriptEntry('jarvis', noReply);
            speakFallback(noReply);
            return;
        }

        // Add JARVIS response to transcript
        addTranscriptEntry('jarvis', reply);

        /* Prefer server TTS (Edge). Always fall back to browser speech if MP3 missing or cannot play. */
        const mp3 = d.audio;
        if (mp3 && String(mp3).length >= 64) {
            const played = await playResponseAudio(mp3);
            if (played) return;
        }
        speakFallback(reply);
    } catch (e) {
        const errMsg = "I can't reach my cognitive core at the moment, sir. Is the server running?";
        addTranscriptEntry('jarvis', errMsg);
        announce('Cannot reach JARVIS.');
        speakFallback(errMsg);
    }
}

function initVoiceConsole() {
    bindLanguageSelect();
    setupVoiceInput();
    fetchLinkStatus();
    setInterval(fetchLinkStatus, 5000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVoiceConsole);
} else {
    initVoiceConsole();
}
