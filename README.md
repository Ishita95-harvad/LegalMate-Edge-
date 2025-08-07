
![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA_Jetson-76B900?logo=nvidia&logoColor=white) ![Google Gemma 3n](https://img.shields.io/badge/Google_Gemma_3n-4285F4?logo=google&logoColor=white)    ![Hugging Face](https://img.shields.io/badge/🤗%20View%20on%20Hugging%20Face-grey?logo=huggingface)   
   ![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)     ![Live App Demo](https://img.shields.io/badge/Live-App_Demo-4285F4?logo=google)]    ![Share on LinkedIn](https://img.shields.io/badge/Share-LinkedIn%20Post-0A66C2?logo=linkedin)) ![Powered By Google Gemma 3n](https://img.shields.io/badge/Powered%20By-Google%20Gemma%203n-4285F4?logo=google&logoColor=white)   
   ![Optimized For NVIDIA Jetson](https://img.shields.io/badge/Optimized%20For-NVIDIA%20Jetson-76B900?logo=nvidia)     ![Multimodal AI](https://img.shields.io/badge/Multimodal-AI--Vision--Language-blueviolet)      ![Gemma Model](https://img.shields.io/badge/Gemma-3n%204B-multimodal)   ![Jetson Powered](https://img.shields.io/badge/Jetson-Nano%20%7C%20Xavier%20NX-blue)       ![Jetson Platform](https://img.shields.io/badge/Jetson-Nano--Ready-green)     ![Build](https://img.shields.io/badge/build-passing-brightgreen)   ![Jetson Powered](https://img.shields.io/badge/Jetson-Nano%20%7C%20Xavier%20NX-blue)     ![Jetson Platform](https://img.shields.io/badge/Jetson-Nano--Ready-green)    ![Build](https://img.shields.io/badge/build-passing-brightgreen)  ![Powered By Google Gemma 3n]( https://img.shields.io/badge/Powered%20By-Google%20Gemma%203n-4285F4?logo=google&logoColor=white)     ![Optimized For NVIDIA Jetson](https://img.shields.io/badge/Optimized%20For-NVIDIA%20Jetson-76B900?logo=nvidia)     ![Multimodal AI](https://img.shields.io/badge/Multimodal-AI--Vision--Language-blueviolet)     ![Gemma Model](https://img.shields.io/badge/Gemma-3n%204B-multimodal)    ![Share on LinkedIn](https://img.shields.io/badge/Share-LinkedIn%20Post-0A66C2?logo=linkedin)]    ![linkedIn Feed](https://www.linkedin.com/feed/)    ![LegalMate Edge – GitHub Repository](https://github.com/Ishita95-harvad/LegalMate-Edge-)
[LegalMate Edge – Hugging Face Spaces](https://huggingface.co/spaces/Ishita95-harvad/LegalMate-Edge-)   ![MTech Portfolio Website](https://ishita95-harvad.github.io/)

- ![LegalMate Edge – GitHub Repository](https://github.com/Ishita95-harvad/LegalMate-Edge-)      -[!LegalMate Edge – Hugging Face Spaces] https://huggingface.co/spaces/Ishita95-harvad/LegalMate-Edge-)
-![MTech Portfolio Website](https://ishita95-harvad.github.io/)


# 🚀[ **LegalMate-Edge-**](https://ishita95-harvad.github.io/Ishita-ai.mtech-portfolio.github.io/#)

## [**Google - The Gemma 3n Impact Challenge**](https://www.kaggle.com/competitions/google-gemma-3n-hackathon)

![Explore the newest Gemma model and build your best products for a better world](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F23623109%2Fe4c38d16ecc0580caf620235c7c6dc0a%2Fheader.png?generation=1754312063215711&alt=media)

**Explore the newest Gemma model and build your best products for a better world**

****

# [**🧠 Legal Mate Edge – Real-Time AI Legal Companion on NVIDIA Jetson**](https://www.kaggle.com/competitions/google-gemma-3n-hackathon)

### 🚀 **Overview**

***********Legal Mate Edge is a cutting-edge Gen AI-powered legal assistant designed to run entirely on-device using the NVIDIA Jetson platform and Gemma 3n LLM. It enables offline legal clause analysis, OCR-powered document scanning, and context-aware contract feedback—all on a lightweight edge device, making legal AI portable, secure, and cost-efficient.***********

-----------------------------------------------------------------------------------------------------------------------------------------------------

🔧 **Key Features**

✅ **Multimodal Interface**: Upload PDF/Images with contracts → Live clause detection via OCR + NLP  
✅ **On-Device Clause Extraction**: Leverages Gemma 3n + Jetson to parse NDAs, SLAs, rental agreements, etc.  
✅ **Privacy-First Legal AI**: Runs fully offline on Jetson, ensuring data never leaves the device  
✅ **Voice & Visual Interaction**: Input via microphone or camera, output via speech and highlights  
✅ **Realtime Feedback**: Suggests risk flags, missing clauses, and potential negotiation points  

----------------------------------------------------------------------------------------------------------------------

### 🔌 **Technologies Used**

| **Stack**          | **Tools/Frameworks**               |
|--------------------|-----------------------------------|
| **LLM**            | Gemma 3n (3B, via Ollama on-device) |
| **OCR**            | Tesseract + EasyOCR               |
| **Deployment**     | NVIDIA Jetson Nano / Xavier NX    |
| **Interface**      | React + Streamlit + Firebase      |
| **Finetuning**     | Unsloth + QLoRA (low-rank adapter)|
| **Backend**        | FastAPI + Torch Serve             |

-----------------------------------------------------------------------------------------------------------------------------------------------------

### Repositories

```
gemma3n-impact-app/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yaml
├── app/                          # Frontend: Android, Streamlit, or Web UI
│   ├── mobile/                   # Android App (Kotlin or Flutter)
│   │   ├── app/src/
│   │   └── GemmaModelLoader.kt
│   ├── streamlit_ui/            # Optional: Streamlit for web-based UI prototype
│   │   ├── app.py
│   │   └── utils.py
│   └── assets/                  # Logos, icons, etc.
│
├── gemma_model/                 # Model setup, loading, and inference
│   ├── inference.py             # Core Gemma 3n usage script
│   ├── ondevice_runner.py       # Handles model selection (2B, 4B)
│   ├── audio_processing.py      # Voice input → embeddings
│   ├── sentiment_engine.py      # Emotion/sentiment classifier
│   └── utils.py
│
├── notebooks/                   # Jupyter notebooks for exploration/training
│   ├── 01-data-exploration.ipynb
│   ├── 02-sentiment-training.ipynb
│   └── 03-gemma3n-integration.ipynb
│
├── data/                        # Sample input data (anonymized if needed)
│   ├── example_voice_samples/
│   └── sample_transcripts.csv
│
├── models/                      # Exported Gemma 3n models / fine-tuned weights
│   ├── gemma_3n_4b.bin
│   ├── gemma_3n_2b.bin
│   └── emotion_model.pt
│
├── api/                         # Optional REST API wrapper for demo
│   ├── main.py
│   ├── routes/
│   │   └── gemma_routes.py
│   └── config.py
│
├── docs/                        # For documentation, technical writeup, visuals
│   ├── architecture.md
│   ├── demo_storyboard.png
│   └── writeup.md
│
└── demo/                        # Final demo script or wrapper
    ├── run_demo.py
    ├── record_voice.py
    └── offline_test_cases/

  ```
-------------------------------------------
### 🔍 **Example Use Case**

- A lawyer scans a printed contract using the Jetson-powered Legal Maté device at a remote site with no internet  
- In real time, the device extracts legal clauses, flags missing indemnity terms, and suggests rephrasing—all securely, without any data ever leaving the device  

-------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Project Links!**

### 🔗 [Legal Mate Edge YouTube Demo](https://youtu.be/Z_ZmGqm3iow?si=s5LJzxm46K2CWZxE)

### 📹 [Live App Link](https://ishita95-harvad.github.io/legalmate-edge/)

 
