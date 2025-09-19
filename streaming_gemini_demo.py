#!/usr/bin/env python3
"""
Enhanced Real-time ISL Demo with Gemini Integration
- Webcam -> MediaPipe Holistic -> sliding window -> GRU -> Word accumulation
- Press 'P' to predict sentence from accumulated words using Gemini 2.5 Flash
- Click on words to deselect/remove them
- Press 'C' to clear all words and start fresh
- Press 'Q' to quit
"""

import argparse
import time
import collections
import numpy as np
import cv2
import torch
import json
import yaml
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import google.generativeai as genai

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
    position: Tuple[int, int] = (0, 0)  # For UI positioning
    

class WordBuffer:
    """Manages accumulated words with confidence filtering and time windows"""
    
    def __init__(self, max_words: int = 15, confidence_threshold: float = 0.6, 
                 time_window: float = 300.0):  # 5 minutes - much longer!
        self.max_words = max_words
        self.confidence_threshold = confidence_threshold
        self.time_window = time_window
        self.words: List[PredictedWord] = []
        self.last_word = ""
        self.last_add_time = datetime.now()
        
    def add_word(self, word: str, confidence: float, min_time_between: float = 3.0) -> bool:  # Increased to 3 seconds
        """Add word if it meets criteria"""
        now = datetime.now()
        
        # Skip if same word added recently
        if (word == self.last_word and 
            (now - self.last_add_time).total_seconds() < min_time_between):
            return False
            
        # Skip if confidence too low
        if confidence < self.confidence_threshold:
            return False
            
        # Skip if word already exists in current buffer (prevent duplicates)
        existing_words = [w.text for w in self.words]
        if word in existing_words:
            return False
            
        # Only clean old words when buffer is completely full
        if len(self.words) >= self.max_words:
            print("Buffer full - removing oldest word to make space")
            self.words.pop(0)  # Just remove the oldest, don't clean by time
        
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
        
    def _clean_old_words(self):
        """Remove words older than time window (rarely used now)"""
        cutoff = datetime.now() - timedelta(seconds=self.time_window)
        old_count = len(self.words)
        self.words = [w for w in self.words if w.timestamp > cutoff]
        new_count = len(self.words)
        if old_count > new_count:
            print(f"Cleaned {old_count - new_count} very old words from buffer")
        
    def get_selected_words(self) -> List[str]:
        """Get list of currently selected word texts"""
        return [w.text for w in self.words if w.selected]
        
    def toggle_word_selection(self, index: int) -> bool:
        """Toggle selection state of word at index"""
        if 0 <= index < len(self.words):
            self.words[index].selected = not self.words[index].selected
            print(f"{'Selected' if self.words[index].selected else 'Deselected'} word: {self.words[index].text}")
            return True
        return False
        
    def remove_deselected_words(self):
        """Remove all deselected words from buffer"""
        removed_count = 0
        self.words = [w for w in self.words if w.selected]
        return removed_count
        
    def clear(self):
        """Clear all words"""
        self.words.clear()
        self.last_word = ""


class GeminiSentencePredictor:
    """Handles Gemini API integration for sentence prediction"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(model)
        self.sentence_history: List[str] = []
        
    def test_gemini_connection(self) -> bool:
        """Test if Gemini API is working"""
        try:
            test_response = self.gen_model.generate_content("Say 'Hello, Gemini is working!'")
            if test_response and test_response.text:
                print(f"Gemini test successful: {test_response.text.strip()}")
                return True
            else:
                print("Gemini test failed: Empty response")
                return False
        except Exception as e:
            print(f"Gemini test failed: {e}")
            return False
        
    def predict_sentence(self, words: List[str], context_sentences: List[str] = None) -> str:
        """Predict sentence from list of words using Gemini"""
        if not words:
            return ""
            
        # Natural ISL-to-English prompt that allows full language freedom
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
            print(f"Sending to Gemini: {words}")
            response = self.gen_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=60,
                    temperature=0.7,  # Higher for more natural variation
                    top_p=0.9
                )
            )
            
            if response and hasattr(response, 'text') and response.text:
                sentence = response.text.strip()
                # Clean up response formatting
                if sentence.startswith("English sentence:"):
                    sentence = sentence[17:].strip()
                if sentence.startswith("-"):
                    sentence = sentence[1:].strip()
                if sentence.startswith('"') and sentence.endswith('"'):
                    sentence = sentence[1:-1]
                
                print(f"Gemini response: '{sentence}'")
                
                # Store in history
                if sentence and sentence not in self.sentence_history:
                    self.sentence_history.append(sentence)
                    if len(self.sentence_history) > 5:
                        self.sentence_history.pop(0)
                        
                return sentence
            else:
                print("Gemini API issue - using fallback")
                return self._create_natural_fallback_sentence(words)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._create_natural_fallback_sentence(words)
            
    def _create_natural_fallback_sentence(self, words: List[str]) -> str:
        """Create a natural conversational sentence as fallback"""
        if not words:
            return ""
        
        # Convert to lowercase for processing
        lower_words = [w.lower() for w in words]
        
        # Natural sentence patterns based on common ISL communication
        if len(words) == 1:
            word = lower_words[0]
            if word in ['happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty']:
                return f"I am {word}."
            elif word in ['camera', 'phone', 'computer']:
                return f"I need the {word}."
            elif word in ['food', 'water', 'help']:
                return f"I want {word}."
            elif word in ['home', 'school', 'work']:
                return f"I am going to {word}."
            else:
                return f"I see a {word}."
                
        elif len(words) == 2:
            w1, w2 = lower_words[0], lower_words[1]
            
            # Common two-word combinations
            if w1 == 'happy' and w2 == 'birthday':
                return "Happy birthday!"
            elif w1 in ['me', 'i'] and w2 in ['hungry', 'tired', 'happy']:
                return f"I am {w2}."
            elif w1 in ['go', 'come'] and w2 in ['home', 'school', 'work']:
                return f"I am going {w2}."
            elif w1 == 'camera' and w2 in ['photo', 'picture']:
                return "I want to take a photo with the camera."
            elif w1 in ['poster', 'picture'] and w2 == 'animals':
                return f"I see a {w1} with {w2}."
            else:
                return f"I see the {w1} and {w2}."
                
        elif len(words) == 3:
            w1, w2, w3 = lower_words[0], lower_words[1], lower_words[2]
            
            # Three-word natural patterns
            if 'camera' in lower_words and 'poster' in lower_words and 'animals' in lower_words:
                return "I took a photo of the animal poster."
            elif 'tomorrow' in lower_words and 'school' in lower_words:
                return "Tomorrow I will go to school."
            elif 'me' in lower_words or 'i' in lower_words:
                other_words = [w for w in lower_words if w not in ['me', 'i']]
                if len(other_words) == 2:
                    return f"I {other_words[0]} {other_words[1]}."
            elif 'exercise' in lower_words and 'healthy' in lower_words:
                return "Exercise keeps me healthy."
            else:
                return f"I see {w1}, {w2}, and {w3}."
        else:
            # For more words, create natural groupings
            if len(words) == 4:
                return f"I see {lower_words[0]}, {lower_words[1]}, {lower_words[2]}, and {lower_words[3]}."
            else:
                return f"There are many things: {', '.join(lower_words[:3])} and more."


class EnhancedStreamingDemo:
    """Main demo class with UI and interaction handling"""
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.setup_model()
        self.setup_gemini()
        self.setup_ui()
        
    def setup_model(self):
        """Initialize the GRU model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self.args.checkpoint, map_location=device, weights_only=False)
        self.class_to_idx = ckpt["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Use model configuration from checkpoint if available
        if "args" in ckpt:
            model_args = ckpt["args"]
            hidden = model_args.hidden
            layers = model_args.layers 
            dropout = model_args.dropout
        else:
            hidden = self.args.hidden
            layers = self.args.layers
            dropout = self.args.dropout
            
        self.model = GRUClassifier(
            in_dim=150, hid=hidden, num_layers=layers,
            num_classes=len(self.class_to_idx), dropout=dropout, bidir=True
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device
        
    def setup_gemini(self):
        """Initialize Gemini predictor and word buffer"""
        gemini_config = self.config.get('gemini', {})
        api_key = gemini_config.get('api_key', '')
        model = gemini_config.get('model', 'gemini-2.5-flash')
        
        if not api_key:
            raise ValueError("Gemini API key not found in config!")
            
        print(f"Initializing Gemini with model: {model}")
        self.gemini = GeminiSentencePredictor(api_key, model)
        
        # Test Gemini connection
        print("Testing Gemini connection...")
        if not self.gemini.test_gemini_connection():
            print("WARNING: Gemini connection test failed. Sentence prediction may not work properly.")
        
        word_config = self.config.get('word_buffer', {})
        self.word_buffer = WordBuffer(
            max_words=word_config.get('max_words', 15),
            confidence_threshold=word_config.get('confidence_threshold', 0.6),
            time_window=word_config.get('time_window', 10.0)
        )
        
    def setup_ui(self):
        """Initialize UI components"""
        self.current_sentence = ""
        self.show_help = True
        self.help_fade_time = time.time() + 5.0  # Show help for 5 seconds
        
        # UI layout constants
        self.WORD_BOX_HEIGHT = 120
        self.SENTENCE_BOX_HEIGHT = 80
        self.HELP_BOX_HEIGHT = 100
        
    def handle_mouse_click(self, event, x, y, flags, param):
        """Handle mouse clicks for word deselection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is in word box area
            word_box_start_y = 60
            word_box_end_y = word_box_start_y + self.WORD_BOX_HEIGHT
            
            if word_box_start_y <= y <= word_box_end_y:
                # Calculate which word was clicked based on position
                words_per_row = 4
                word_width = 150
                word_height = 25
                
                row = (y - word_box_start_y) // word_height
                col = x // word_width
                word_index = row * words_per_row + col
                
                if self.word_buffer.toggle_word_selection(word_index):
                    print(f"Toggled word {word_index}")
                    
    def draw_ui(self, frame):
        """Draw the complete UI overlay"""
        height, width = frame.shape[:2]
        
        # Create overlay for better text visibility
        overlay = frame.copy()
        
        # Draw word box
        self.draw_word_box(overlay)
        
        # Draw sentence box
        self.draw_sentence_box(overlay)
        
        # Draw help if needed
        if self.show_help and time.time() < self.help_fade_time:
            self.draw_help_box(overlay)
        elif time.time() >= self.help_fade_time:
            self.show_help = False
            
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
    def draw_word_box(self, frame):
        """Draw the word accumulation box"""
        start_y = 60
        box_height = self.WORD_BOX_HEIGHT
        
        # Background box
        cv2.rectangle(frame, (10, start_y), (640, start_y + box_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, start_y), (640, start_y + box_height), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "Unique Words (Click to select/deselect):", (15, start_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw words in grid
        words_per_row = 4
        word_width = 150
        word_height = 25
        
        for i, word_obj in enumerate(self.word_buffer.words):
            row = i // words_per_row
            col = i % words_per_row
            
            x = 15 + col * word_width
            y = start_y + 40 + row * word_height
            
            # Color based on selection state
            color = (0, 255, 0) if word_obj.selected else (100, 100, 100)
            text = f"{word_obj.text} ({word_obj.confidence:.2f})"
            
            # Background for word
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x-2, y-15), (x + text_size[0] + 4, y + 5), 
                         (30, 30, 30), -1)
            
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    def draw_sentence_box(self, frame):
        """Draw the predicted sentence box"""
        start_y = 200
        box_height = self.SENTENCE_BOX_HEIGHT
        
        # Background box
        cv2.rectangle(frame, (10, start_y), (640, start_y + box_height), (30, 50, 30), -1)
        cv2.rectangle(frame, (10, start_y), (640, start_y + box_height), (0, 150, 0), 2)
        
        # Title
        cv2.putText(frame, "Predicted Sentence:", (15, start_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Sentence text (wrapped if needed)
        if self.current_sentence:
            self.draw_wrapped_text(frame, self.current_sentence, (15, start_y + 45), 
                                 600, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Press 'P' to predict sentence from words above", 
                       (15, start_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
    def draw_help_box(self, frame):
        """Draw help instructions"""
        start_y = 300
        box_height = self.HELP_BOX_HEIGHT
        
        # Background box
        cv2.rectangle(frame, (10, start_y), (640, start_y + box_height), (50, 30, 30), -1)
        cv2.rectangle(frame, (10, start_y), (640, start_y + box_height), (100, 50, 50), 2)
        
        # Help text
        help_lines = [
            "Controls:",
            "P - Predict sentence from selected words",
            "C - Clear all words",
            "R - Remove deselected words",
            "Q - Quit",
            "Click words to select/deselect"
        ]
        
        for i, line in enumerate(help_lines):
            y = start_y + 15 + i * 16
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                       (200, 200, 200), 1)
                       
    def draw_wrapped_text(self, frame, text, pos, max_width, font, font_scale, color, thickness):
        """Draw text with word wrapping"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                
        if current_line:
            lines.append(current_line)
            
        x, y = pos
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x, y + i * 20), font, font_scale, color, thickness)
            
    def run(self):
        """Main demo loop"""
        print("Initializing MediaPipe...")
        try:
            mp = _import_mediapipe()
            holistic = mp.solutions.holistic.Holistic(
                model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            print("MediaPipe initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            return

        print("Opening camera...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Camera 1 not available, trying camera 0...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Cannot open any webcam")
                return

        print("Camera opened successfully")

        # Set up mouse callback
        cv2.namedWindow("ISL Gemini Demo")
        cv2.setMouseCallback("ISL Gemini Demo", self.handle_mouse_click)

        buf = collections.deque(maxlen=self.args.window)
        frame_idx = 0
        current_prediction = "..."
        
        print("Demo started! Controls: P=predict sentence, C=clear all, R=remove deselected, Q=quit")
        
        try:
            while True:
                ok, frame = cap.read()
                if not ok: 
                    print("Failed to read frame from camera")
                    break
                    
                # Flip frame for better user experience
                frame = cv2.flip(frame, 1)
                    
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    res = holistic.process(rgb)
                    feat = features_from_frame(res)
                    buf.append(feat)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

                # Predict every stride frames
                if len(buf) == self.args.window and frame_idx % self.args.stride == 0:
                    try:
                        x = torch.from_numpy(np.stack(buf)[None, ...]).to(self.device)
                        with torch.no_grad():
                            logits = self.model(x)
                            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                            pred_idx = int(prob.argmax())
                            confidence = float(prob[pred_idx])
                            
                        predicted_word = self.idx_to_class[pred_idx]
                        current_prediction = f"{predicted_word} ({confidence:.2f})"
                        
                        # Add to word buffer if confidence is high enough
                        if self.word_buffer.add_word(predicted_word, confidence):
                            print(f"Added word: {predicted_word} ({confidence:.2f})")
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        continue

                # Draw current prediction
                cv2.putText(frame, f"Current: {current_prediction}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Draw UI overlay
                try:
                    self.draw_ui(frame)
                except Exception as e:
                    print(f"Error drawing UI: {e}")

                cv2.imshow("ISL Gemini Demo", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') or key == ord('P'):
                    # Predict sentence
                    selected_words = self.word_buffer.get_selected_words()
                    if selected_words:
                        print(f"Predicting sentence from: {selected_words}")
                        try:
                            self.current_sentence = self.gemini.predict_sentence(
                                selected_words, self.gemini.sentence_history
                            )
                            print(f"Predicted: {self.current_sentence}")
                        except Exception as e:
                            print(f"Error predicting sentence: {e}")
                            self.current_sentence = " ".join(selected_words)  # Fallback
                    else:
                        print("No words selected for sentence prediction")
                        
                elif key == ord('c') or key == ord('C'):
                    # Clear words and sentence
                    self.word_buffer.clear()
                    self.current_sentence = ""
                    print("Cleared all words and sentence")
                    
                elif key == ord('r') or key == ord('R'):
                    # Remove only deselected words
                    before_count = len(self.word_buffer.words)
                    self.word_buffer.remove_deselected_words()
                    after_count = len(self.word_buffer.words)
                    removed_count = before_count - after_count
                    if removed_count > 0:
                        print(f"Removed {removed_count} deselected word(s)")
                    else:
                        print("No deselected words to remove")
                    
                elif key == ord('h') or key == ord('H'):
                    # Toggle help
                    self.show_help = not self.show_help
                    if self.show_help:
                        self.help_fade_time = time.time() + 10.0

                frame_idx += 1
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"Error during demo: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            holistic.close()
            print("Demo ended")


def main():
    p = argparse.ArgumentParser("Enhanced ISL Streaming Demo with Gemini Integration")
    p.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    p.add_argument("--window", type=int, default=32, help="Frames in sliding window")
    p.add_argument("--stride", type=int, default=4, help="Reclassify every N frames")
    p.add_argument("--hidden", type=int, default=256, help="Hidden size (if not in checkpoint)")
    p.add_argument("--layers", type=int, default=2, help="Number of layers (if not in checkpoint)")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (if not in checkpoint)")
    p.add_argument("--config", default="gemini_config.yaml", help="Path to Gemini config file")
    args = p.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found!")
        print("Please ensure gemini_config.yaml exists with your Gemini API key")
        return 1
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Validate Gemini API key
    api_key = config.get('gemini', {}).get('api_key', '')
    if not api_key or api_key == 'your_api_key_here':
        print("Please set your Gemini API key in gemini_config.yaml")
        return 1

    try:
        demo = EnhancedStreamingDemo(args, config)
        demo.run()
        return 0
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())