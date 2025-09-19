# 🤟 ISL Translation with Gemini AI

A Streamlit web application for Indian Sign Language (ISL) translation using deep learning and Google's Gemini AI for natural sentence formation.

## ✨ Features

- **Video Upload**: Upload ISL gesture videos for analysis
- **Real-time Prediction**: Extract words from ISL gestures using GRU neural network
- **Word Management**: Select/deselect predicted words with confidence scores
- **Natural Sentences**: Generate natural English sentences using Gemini AI
- **User-friendly Interface**: Clean, intuitive Streamlit web interface

## 🚀 Quick Start

### Local Development

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd ISL_Streamlit_Deploy
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run streamlit_gemini_demo.py
   ```

3. **Get Gemini API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key
   - Enter it in the app sidebar

### Streamlit Cloud Deployment

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial ISL Streamlit app"
   git branch -M main
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file: `streamlit_gemini_demo.py`
   - Deploy!

## 📁 Project Structure

```
ISL_Streamlit_Deploy/
├── streamlit_gemini_demo.py    # Main Streamlit application
├── common.py                   # Core ML utilities and model definition
├── requirements.txt            # Python dependencies
├── checkpoints/
│   └── best_gru.pt            # Pre-trained ISL recognition model (27MB)
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── README.md                   # This file
```

## 🎯 How to Use

1. **Enter API Key**: Add your Gemini API key in the sidebar
2. **Upload Video**: Choose an ISL video file (MP4, AVI, MOV, WebM)
3. **Analyze**: Click "Analyze Video" to extract predicted words
4. **Select Words**: Click on words to select/deselect them
5. **Generate Sentence**: Use Gemini AI to create natural sentences

## 📋 Requirements

- **Python 3.10** (recommended for Streamlit Cloud)
- Streamlit Cloud compatible dependencies
- Google Gemini API key (free tier available)

## 🐍 Python Version Setup

This app is optimized for **Python 3.10**. The deployment includes:

- `.python-version` - Specifies Python 3.10
- `runtime.txt` - Streamlit Cloud runtime configuration  
- `pyproject.toml` - Modern Python project configuration
- `requirements.txt` - Python 3.10 compatible dependencies

### For Local Development:
```bash
# Using pyenv (recommended)
pyenv install 3.10.12
pyenv local 3.10.12

# Using conda
conda create -n isl-env python=3.10
conda activate isl-env

# Using venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

## 🎥 Supported Video Formats

- MP4, AVI, MOV, WebM
- Recommended: 5-15 seconds duration
- Clear lighting and stable camera
- Hands visible in frame

## 🔧 Technical Details

- **Model**: GRU-based neural network (84 ISL classes)
- **Features**: MediaPipe Holistic landmarks (150 dimensions)
- **AI Integration**: Google Gemini 2.0 Flash for sentence formation
- **Framework**: Streamlit for web interface

## 📝 Model Information

- **Size**: ~27MB (optimized for cloud deployment)
- **Classes**: 84 common ISL words/phrases
- **Input**: 32-frame sequences of pose/hand landmarks
- **Accuracy**: Optimized for real-world ISL recognition

## 🌐 Deployment Notes

- **Memory Usage**: ~500MB (within Streamlit Cloud limits)
- **Processing**: CPU-only inference (no GPU required)
- **Caching**: Model and MediaPipe initialization cached
- **Scalability**: Suitable for personal/educational use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

This project is open source. Please ensure you have appropriate rights to use the pre-trained model for your intended purpose.

## 🆘 Troubleshooting

**Model Loading Issues:**
- Ensure `checkpoints/best_gru.pt` exists and is not corrupted
- Check file size is approximately 27MB

**API Key Issues:**
- Verify Gemini API key is valid
- Check API quota limits
- Ensure internet connectivity

**Video Processing Issues:**
- Use supported formats (MP4 recommended)
- Keep videos under 30 seconds
- Ensure clear lighting and hand visibility

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Streamlit logs for error details
3. Ensure all dependencies are correctly installed