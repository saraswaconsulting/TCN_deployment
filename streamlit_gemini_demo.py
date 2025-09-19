#!/usr/bin/env python3
"""
Real-time Streamlit ISL Demo with Gemini Integration
Live camera streaming with ISL predictions and sentence formation
"""

import streamlit as st
import cv2
import numpy as np
import torch
import json
import os
import time
import collections
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# Suppress MediaPipe verbose logging
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from common import GRUClassifier, features_from_frame, _import_mediapipe


@dataclass
class PredictedWord:
    """Represents a predicted word with metadata"""
    text: str
    confidence: float
    timestamp: datetime
    selected: bool = True


class WordBuffer:
    """Manages accumulated words with confidence filtering and time windows"""
    
    def __init__(self, max_words: int = 15, confidence_threshold: float = 0.6, 
                 time_window: float = 300.0):
        self.max_words = max_words
        self.confidence_threshold = confidence_threshold
        self.time_window = time_window
        self.words: List[PredictedWord] = []
        self.last_word = ""
        self.last_add_time = datetime.now()
        self.lock = threading.Lock()
        
    def add_word(self, word: str, confidence: float, min_time_between: float = 3.0) -> bool:
        """Add word if it meets criteria"""
        with self.lock:
            now = datetime.now()
            
            # Skip if same word added recently
            if (word == self.last_word and 
                (now - self.last_add_time).total_seconds() < min_time_between):
                return False
                
            # Skip if confidence too low
            if confidence < self.confidence_threshold:
                return False
                
            # Skip if word already exists in current buffer
            existing_words = [w.text for w in self.words]
            if word in existing_words:
                return False
                
            # Remove oldest if buffer is full
            if len(self.words) >= self.max_words:
                self.words.pop(0)
            
            # Add new word
            new_word = PredictedWord(
                text=word,
                confidence=confidence,
                timestamp=now
            )
            
            self.words.append(new_word)
            self.last_word = word
            self.last_add_time = now
            
            return True
        
    def get_selected_words(self) -> List[str]:
        """Get list of currently selected word texts"""
        with self.lock:
            return [w.text for w in self.words if w.selected]
        
    def toggle_word_selection(self, index: int) -> bool:
        """Toggle selection state of word at index"""
        with self.lock:
            if 0 <= index < len(self.words):
                self.words[index].selected = not self.words[index].selected
                return True
            return False
        
    def clear(self):
        """Clear all words"""
        with self.lock:
            self.words.clear()
            self.last_word = ""

    def get_words_copy(self):
        """Get a thread-safe copy of words"""
        with self.lock:
            return self.words.copy()

    def remove_deselected_words(self) -> int:
        """Remove all deselected words from buffer"""
        with self.lock:
            before = len(self.words)
            self.words = [w for w in self.words if w.selected]
            return before - len(self.words)


class GeminiSentencePredictor:
    """Handles Gemini API integration for sentence prediction"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(model)
        self.sentence_history: List[str] = []
        
    def predict_sentence(self, words: List[str]) -> str:
        """Predict sentence from list of words using Gemini"""
        if not words:
            return ""
            
        prompt = f"""You are an expert Indian Sign Language (ISL) interpreter helping deaf users communicate naturally.

I will give you words that were signed in ISL. Your job is to create a natural, fluent English sentence that captures what the person was trying to communicate.

ISL CONTEXT:
- ISL users think and express ideas naturally, just like spoken language users
- You can add any English words (articles, prepositions, verbs, etc.) needed for fluent communication
- Focus on natural meaning, not literal word-for-word translation
- Make it sound like something a person would actually say in conversation

SIGNED WORDS: {', '.join(words)}

Create a natural English sentence that expresses what the signer meant to communicate. Be conversational and natural.

English sentence:"""

        try:
            response = self.gen_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=60,
                    temperature=0.7,
                    top_p=0.9
                )
            )
            
            if response and hasattr(response, 'text') and response.text:
                sentence = response.text.strip()
                if sentence.startswith("English sentence:"):
                    sentence = sentence[17:].strip()
                if sentence.startswith("-"):
                    sentence = sentence[1:].strip()
                if sentence.startswith('"') and sentence.endswith('"'):
                    sentence = sentence[1:-1]
                
                if sentence and sentence not in self.sentence_history:
                    self.sentence_history.append(sentence)
                    if len(self.sentence_history) > 5:
                        self.sentence_history.pop(0)
                        
                return sentence
            else:
                return self._create_fallback_sentence(words)
            
        except Exception as e:
            st.error(f"Gemini API error: {e}")
            return self._create_fallback_sentence(words)
            
    def _create_fallback_sentence(self, words: List[str]) -> str:
        """Create a natural sentence as fallback"""
        if not words:
            return ""
        
        if len(words) == 1:
            word = words[0].lower()
            if word in ['happy', 'sad', 'angry', 'tired', 'hungry']:
                return f"I am {word}."
            else:
                return f"I see {word}."
        elif len(words) == 2:
            return f"I see {words[0]} and {words[1]}."
        else:
            return f"I see {', '.join(words[:-1])} and {words[-1]}."


class ISLVideoProcessor(VideoProcessorBase):
    """Real-time video processor for ISL recognition"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.holistic = None
        self.buf = collections.deque(maxlen=32)
        self.frame_idx = 0
        self.current_prediction = "Initializing..."
        self.word_buffer = None
        self.gemini = None
        
    def setup_model(self, checkpoint_path):
        """Initialize the ISL model"""
        if self.model is not None:
            return
            
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.class_to_idx = ckpt["class_to_idx"]
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            if "args" in ckpt:
                model_args = ckpt["args"]
                hidden = model_args.hidden
                layers = model_args.layers 
                dropout = model_args.dropout
            else:
                hidden = 256
                layers = 2
                dropout = 0.3
                
            self.model = GRUClassifier(
                in_dim=150, hid=hidden, num_layers=layers,
                num_classes=len(self.class_to_idx), dropout=dropout, bidir=True
            ).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            
            # Setup MediaPipe
            mp = _import_mediapipe()
            self.holistic = mp.solutions.holistic.Holistic(
                model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            
            # Setup word buffer and Gemini
            self.word_buffer = WordBuffer()
            
            # Get API key from Streamlit secrets or environment
            api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.gemini = GeminiSentencePredictor(api_key)
            
            self.current_prediction = "Ready"
            
        except Exception as e:
            self.current_prediction = f"Setup error: {e}"
    
    def recv(self, frame):
        """Process each video frame (called by streamlit-webrtc)"""
        if self.model is None:
            # Draw loading message
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, "Loading model...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            res = self.holistic.process(rgb)
            feat = features_from_frame(res)
            self.buf.append(feat)
            
            # Predict every 4 frames
            if len(self.buf) == 32 and self.frame_idx % 4 == 0:
                x = torch.from_numpy(np.stack(self.buf)[None, ...]).to(self.device)
                with torch.no_grad():
                    logits = self.model(x)
                    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred_idx = int(prob.argmax())
                    confidence = float(prob[pred_idx])
                    
                predicted_word = self.idx_to_class[pred_idx]
                self.current_prediction = f"{predicted_word} ({confidence:.2f})"
                
                # Add to word buffer
                if self.word_buffer and self.word_buffer.add_word(predicted_word, confidence):
                    st.session_state.word_added = True
            
            # Draw prediction on frame
            cv2.putText(img, f"Current: {self.current_prediction}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # No landmark overlay: keep raw camera view for a clean UI
            
            self.frame_idx += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)[:50]}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(
        page_title="Real-time ISL Recognition with Gemini",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🤟 Real-time ISL Recognition with Gemini AI")
    st.markdown("*Live camera streaming with intelligent sentence formation*")
    
    # Initialize session state
    if 'word_added' not in st.session_state:
        st.session_state.word_added = False
    if 'current_sentence' not in st.session_state:
        st.session_state.current_sentence = ""
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = None
    if 'last_pred_time' not in st.session_state:
        st.session_state.last_pred_time = 0.0
    
    # API Key setup
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ Gemini API key not found!")
        st.info("Please set GEMINI_API_KEY in Streamlit secrets or environment variables")
        st.code("# Add to .streamlit/secrets.toml:\nGEMINI_API_KEY = 'your_api_key_here'")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎮 Controls")
        
        if st.button("🗑️ Clear All Words", type="secondary"):
            if st.session_state.video_processor and st.session_state.video_processor.word_buffer:
                st.session_state.video_processor.word_buffer.clear()
                st.session_state.current_sentence = ""
                st.success("Cleared all words!")

        if st.button("🧹 Remove Deselected Words"):
            if st.session_state.video_processor and st.session_state.video_processor.word_buffer:
                removed = st.session_state.video_processor.word_buffer.remove_deselected_words()
                st.success(f"Removed {removed} deselected word(s)")
        
        st.markdown("---")
        st.markdown("### Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.1)
        max_words = st.slider("Max Words in Buffer", 5, 20, 15)
        auto_predict = st.checkbox("Auto-predict sentence", value=False, help="Automatically generate sentences when new words are added")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera")
        
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Start video stream
        webrtc_ctx = webrtc_streamer(
            key="isl-camera",
            video_processor_factory=ISLVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            # Request higher quality video from the browser camera
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        # Setup model when video starts
        if webrtc_ctx.video_processor:
            st.session_state.video_processor = webrtc_ctx.video_processor
            if hasattr(webrtc_ctx.video_processor, 'setup_model'):
                webrtc_ctx.video_processor.setup_model("checkpoints/best_gru.pt")
    
    with col2:
        st.subheader("📝 Word Buffer")
        
        # Display collected words
        if (st.session_state.video_processor and 
            hasattr(st.session_state.video_processor, 'word_buffer') and 
            st.session_state.video_processor.word_buffer):
            
            # Apply sidebar settings to the buffer on each run
            st.session_state.video_processor.word_buffer.confidence_threshold = confidence_threshold
            st.session_state.video_processor.word_buffer.max_words = max_words

            word_buffer = st.session_state.video_processor.word_buffer
            words = word_buffer.get_words_copy()
            
            if words:
                st.markdown("**Detected Words:**")
                
                # Create word selection interface
                for i, word_obj in enumerate(words):
                    col_word, col_conf, col_select = st.columns([3, 2, 1])
                    
                    with col_word:
                        st.text(word_obj.text)
                    with col_conf:
                        st.text(f"{word_obj.confidence:.2f}")
                    with col_select:
                        if st.button("❌", key=f"toggle_{i}"):
                            word_buffer.toggle_word_selection(i)
                
                # Predict sentence button
                selected_words = word_buffer.get_selected_words()
                # Auto-predict when enabled
                if auto_predict and selected_words:
                    should_predict = st.session_state.word_added or (time.time() - st.session_state.last_pred_time) > 2.5
                    if should_predict:
                        if st.session_state.video_processor.gemini:
                            with st.spinner("Generating sentence..."):
                                sentence = st.session_state.video_processor.gemini.predict_sentence(selected_words)
                                st.session_state.current_sentence = sentence
                                st.session_state.last_pred_time = time.time()
                        else:
                            st.session_state.current_sentence = " ".join(selected_words)
                            st.session_state.last_pred_time = time.time()
                        st.session_state.word_added = False
                if selected_words and st.button("🧠 Predict Sentence", type="primary"):
                    if st.session_state.video_processor.gemini:
                        with st.spinner("Generating sentence..."):
                            sentence = st.session_state.video_processor.gemini.predict_sentence(selected_words)
                            st.session_state.current_sentence = sentence
                    else:
                        st.session_state.current_sentence = " ".join(selected_words)
                
                # Show selected words
                if selected_words:
                    st.markdown("**Selected words:**")
                    st.write(" → ".join(selected_words))
            else:
                st.info("No words detected yet. Start signing!")
        
        # Display current sentence
        st.subheader("💬 Generated Sentence")
        if st.session_state.current_sentence:
            st.success(st.session_state.current_sentence)
        else:
            st.info("Predict a sentence from detected words")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Allow camera access** when prompted
        2. **Sign ISL gestures** in front of the camera
        3. **Watch words appear** in the Word Buffer as they're detected
        4. **Click ❌ to deselect** unwanted words
        5. **Click 'Predict Sentence'** to generate natural English
        6. **Use 'Clear All Words'** to start fresh
        
        **Tips:**
        - Ensure good lighting and clear hand visibility
        - Sign at a moderate pace for better accuracy
        - Deselect hallucinated or wrong words before prediction
        """)


if __name__ == "__main__":
    main()