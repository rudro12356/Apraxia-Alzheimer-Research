# Apraxia Alzheimer Research

# 🧠 Apraxia & Alzheimer Research Project

This repository contains code and resources for analyzing **Apraxia movements** using **skeleton-based pose estimation**. The project utilizes **MediaPipe**, **OpenCV**, and **machine learning** to extract pose landmarks from video sequences and process them for research purposes.

## 📖 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Example Workflow](#example-workflow)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 📝 Introduction
This project is designed for **motion analysis in Apraxia and Alzheimer’s research**. It processes **video data** of patients performing predefined actions, extracts **skeleton joints**, and stores them for further analysis. The extracted skeleton data can be used for **gesture classification**, **anomaly detection**, and **movement analysis**.

---

## ✨ Features
✅ **Extracts human pose landmarks** using **MediaPipe Pose**  
✅ **Processes medical video datasets** to analyze movement patterns  
✅ **Outputs structured CSV files** containing skeleton joint positions  
✅ **Supports batch processing** for multiple patients  
✅ **Automated Video-to-Skeleton Pipeline**  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```
git clone https://github.com/rudro12356/Apraxia-Alzheimer-Research.git
cd Apraxia-Alzheimer-Research
pip install -r requirements.txt
```

📌 Example Workflow  
	1.	Prepare dataset: Place patient videos in dataset/Patient_XX/RGB/  
	2.	Ensure annotations exist: Each patient should have an annotation CSV in annotations/  
	3.	Run main script: Execute main.py to extract skeleton data.  
	4.	Analyze output: The extracted skeleton data will be stored as CSVs.  

 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

⸻

📜 License

This project is licensed under the MIT License.

⸻

🙌 Acknowledgments

Special thanks to Professor Sumaiya’s Lab for guidance and the open-source community for amazing tools like MediaPipe, OpenCV, and Pandas.
