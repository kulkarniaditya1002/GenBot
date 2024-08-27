# 🎓 EduBot: Your AI-Powered Companion for Learning Coding and English Language Basics 📚

## Introduction

Welcome to EduBot, a generative AI chatbot designed to bridge the gaps in education and skill development, specifically targeting beginner-level coding and English language learning. EduBot provides clear and concise information, guiding users through fundamental concepts without promoting plagiarism. This project leverages cutting-edge AI models to support learners in building essential skills, ensuring accessibility and effectiveness in their educational journey.

## Project Goals

### 🎯 Objective-Driven Learning
EduBot is crafted to assist users in comprehending basic coding concepts and English grammar, ensuring that the information is both clear and original. The chatbot aims to:
- Provide introductory-level content in coding and English language basics.
- Encourage originality and understanding in learning processes by generating unique responses.
- Avoid in-depth or advanced topics to prioritize foundational understanding.

### 🛠️ Tailored Solution
EduBot is not just another generic chatbot. It's specifically designed to aid students who are beginners, providing targeted assistance that is both effective and easy to understand. Our unique approach includes:
- **Plagiarism Mitigation:** EduBot generates unique and tailored responses to help control plagiarism, promoting originality.
- **Custom Dataset Creation:** We compiled a bespoke dataset to ensure that the training material aligns closely with EduBot’s intended use case and audience.
- **Model Selection:** After extensive research, we selected three specific models to deliver optimal performance in generating accurate and helpful responses.

## Development Approach

### 📚 Research and Literature Review
We extensively reviewed research papers on the use of large language models (LLMs) and generative AI in education, revealing their significant benefits in enhancing student engagement and learning outcomes.

### 📊 Dataset Creation
Our dataset is meticulously structured, focusing on English grammar and coding, with over 80,000 records available for training. This dataset is crucial for EduBot's ability to generate well-informed responses across diverse domains.

### 🔍 Model Selection and Training
After careful consideration, we selected three transformer models—GPT-2 LM Head Model, BERT for Question Answering, and T5 Transformer Model—each trained to handle specific tasks such as pseudo-code generation and grammar assistance. The models were trained on a Kaggle GPU (P100 16GB) using PyTorch and sklearn, achieving impressive results with low training and validation losses.

### 🖥️ Application Development
EduBot was developed using Python, JavaScript, and Streamlit, creating an interactive and robust application. The app is designed to provide customized grammar help, with features like model selection and chat history saving.

## How It Works

EduBot interacts with users through a streamlined interface, processing user inputs and generating responses based on the trained models. The application is hosted on a Streamlit server, ensuring smooth and efficient performance.

1. **User Input and Configuration:** Users can interact with EduBot by entering their queries related to coding or English grammar.
2. **Model and Tokenizer Loading:** The selected model processes the input and generates a response.
3. **Response Generation:** EduBot provides an informative and original response, aiding the user in their learning journey.
4. **Output Display:** The response is displayed to the user, helping them grasp the fundamental concepts.

## Key Features

- **Customized Grammar Help:** Tailored responses that cater to specific grammar queries.
- **Plagiarism Control:** EduBot provides pseudo-code instead of complete solutions to minimize the risk of plagiarism.
- **Interactive Interface:** A user-friendly interface that enhances learning experiences.

## Model Training

| **Details**                            | **Description**                                                                           |
|----------------------------------------|-------------------------------------------------------------------------------------------|
| **Training Platform**                  | Kaggle GPU (P100 16GB)                                                                    |
| **Training Duration**                  | 20+ hours                                                                                 |
| **API Used**                           | Weights & Biases (W&B) API                                                                |
| **Frameworks & Libraries**             | PyTorch, sklearn                                                                          |
| **Training & Evaluation Description**  | The models were trained using PyTorch and sklearn, leveraging the capabilities of Kaggle's GPU environment. The W&B API was utilized for experiment tracking and monitoring. The training process involved over 20 hours of computation, ensuring that the transformer models were effectively trained and evaluated. |

## Evaluation Metrics

| **Model Name**              | **Metric Description**                                                                 | **BLEU Score** | **ROUGE-1**                                      | **ROUGE-2**                                      |
|-----------------------------|----------------------------------------------------------------------------------------|----------------|--------------------------------------------------|--------------------------------------------------|
| **GPT-2 LM Head Model**      | Used for assessing the quality of pseudo-code generation by comparing model output with a reference. | -              | \( f = 0.44 \), \( p = 0.46 \), \( r = 0.43 \)    | \( f = 0.28 \), \( p = 0.30 \), \( r = 0.27 \)    |
| **BERT for Question Answering** | Applied to evaluate the grammatical correctness and relevance of answers to English grammar questions. | -              | \( f = 0.42 \), \( p = 0.45 \), \( r = 0.40 \)    | \( f = 0.25 \), \( p = 0.27 \), \( r = 0.24 \)    |
| **T5 Transformer Model**     | Measured to gauge performance across multiple tasks including grammar assistance and pseudo-code generation. | 28            | \( f = 0.48 \), \( p = 0.51 \), \( r = 0.46 \)    | \( f = 0.33 \), \( p = 0.35 \), \( r = 0.32 \)    |

## Conclusion

EduBot showcases the power of generative AI in education, blending manual effort with AI assistance to foster genuine learning. By focusing on fundamental concepts, EduBot ensures that users build a strong foundation in coding and English grammar, all while maintaining academic integrity.

## Future Scope

- **Scalability:** Enhance scalability by upgrading GPU resources and optimizing performance for higher user concurrency.
- **Language and Framework Expansion:** Broaden EduBot's utility by adding support for additional languages and programming frameworks.
- **Advanced History-Saving:** Improve history-saving features with categorization and efficient searchability.
- **Anti-Plagiarism Enhancements:** Advance anti-plagiarism measures to align with evolving academic standards.

## Trained Model: 
The model files for running this application are uploaded here: [kulkarniaditya1002/ENG_Grammer_QA](https://huggingface.co/kulkarniaditya1002/ENG_Grammer_QA)

## Steps to Download, Pull, and Run the Code
   ```bash
   git clone https://github.com/GenBot/EduBot.git
   cd EduBot
   pip install -r requirements.txt
   streamlit run app.py
