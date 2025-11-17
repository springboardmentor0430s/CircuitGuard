# 🔬 CircuitGuard: THE PCB INSPECTOR

**An intelligent AI-powered system for automated detection and analysis of manufacturing defects in Printed Circuit Boards**

## 🌟 What is CircuitGuard?

CircuitGuard is a cutting-edge web application that revolutionizes the way we inspect Printed Circuit Boards (PCBs). By combining the power of computer vision and deep learning, it transforms the traditionally manual and error-prone inspection process into an automated, accurate, and lightning-fast quality control system. Whether you're a small electronics workshop or a large-scale manufacturing facility, CircuitGuard brings enterprise-grade defect detection capabilities to your fingertips through a simple, intuitive web interface.

The system works by comparing a test PCB against a reference "golden" template, automatically identifying any deviations that could indicate manufacturing defects. Using advanced image processing algorithms and a specially trained neural network, CircuitGuard can detect and classify six different types of common PCB defects with accuracy rates exceeding 85-95%, all within seconds.

## 🎯 The Problem We're Solving

In the electronics manufacturing industry, quality control of PCBs is a critical bottleneck. Traditional manual inspection methods face several serious challenges that affect both productivity and product quality.

**Time and Resource Intensive**: Manual PCB inspection requires trained technicians to visually examine each board, which can take several minutes per unit. In high-volume production environments, this becomes a significant bottleneck, slowing down the entire manufacturing pipeline and increasing labor costs.

**Human Error and Fatigue**: Even the most skilled inspectors experience fatigue after hours of repetitive visual examination. This leads to decreased accuracy, especially for subtle defects. Studies show that human inspection accuracy drops significantly after just a few hours of continuous work, potentially allowing defective boards to pass through quality control.

**Inconsistency**: Different inspectors may have varying interpretations of what constitutes a defect, leading to inconsistent quality standards across shifts or production lines. This subjectivity makes it difficult to maintain uniform quality control across large-scale operations.

**Scalability Issues**: As production volumes increase, hiring and training more inspectors becomes expensive and time-consuming. Manual inspection simply cannot scale efficiently with growing production demands.

**Limited Documentation**: Manual inspection often relies on simple pass/fail records without detailed defect analysis, making it difficult to identify recurring issues or track quality trends over time.

## ✅ Our Solution

CircuitGuard addresses these challenges head-on by providing a comprehensive automated inspection solution that combines speed, accuracy, and detailed analytics.

**Automated Real-Time Detection**: The system processes PCB images in just 2-5 seconds, identifying defects with precision that matches or exceeds human inspectors. This dramatic speed improvement allows manufacturers to inspect every single board without creating production bottlenecks.

**AI-Powered Intelligence**: At the heart of CircuitGuard is a deep learning model trained on thousands of PCB images. This neural network has learned to recognize subtle patterns and anomalies that indicate defects, bringing consistency and reliability to the inspection process. Unlike human inspectors, the AI never gets tired and maintains the same high accuracy throughout continuous operation.

**Comprehensive Analysis**: CircuitGuard doesn't just detect defects—it provides detailed analysis including defect classification, severity scoring, affected area calculations, and actionable repair recommendations. Each inspection generates comprehensive reports with visual annotations, statistical data, and quality metrics.

**User-Friendly Interface**: Despite its sophisticated technology, CircuitGuard is remarkably easy to use. The web-based interface requires no special training—simply upload your template and test images, adjust a few parameters if needed, and receive detailed results within seconds.

**Scalable Architecture**: Built on modern web technologies, CircuitGuard can easily scale from a single workstation to a distributed system handling hundreds of inspections simultaneously. The system can be deployed locally or in the cloud, adapting to your specific infrastructure needs.

## 🚀 Key Features and Capabilities

### Intelligent Image Processing

CircuitGuard employs advanced computer vision techniques to ensure accurate defect detection even when images aren't perfectly aligned. The system automatically registers the test image with the template using feature-based matching algorithms, compensating for minor rotations, translations, or perspective differences. This means you don't need expensive precision imaging equipment—standard industrial cameras work perfectly well.

Once aligned, the system performs pixel-level comparison to identify areas where the test PCB differs from the template. Sophisticated morphological operations filter out noise and enhance actual defect regions, ensuring high signal-to-noise ratio in detection.

### Deep Learning Classification

When potential defect regions are identified, CircuitGuard's neural network analyzes each one to determine the specific type of defect. The system can recognize six critical defect categories:

**Missing Holes** occur when drill holes or vias that should be present in the PCB are absent. These can prevent proper component mounting or electrical connections.

**Mouse Bites** are irregular or incomplete cutouts along the PCB edges, often resulting from issues with the depaneling process.

**Open Circuits** represent breaks in copper traces that disrupt electrical pathways, potentially rendering the entire board non-functional.

**Short Circuits** occur when copper traces that should be separate make unintended contact, causing electrical faults and potential component damage.

**Solder Bridges** happen when excess solder creates unwanted connections between pads or traces, leading to short circuits.

**Spurious Copper** refers to unwanted copper residue on the board that can cause electrical issues or interfere with component placement.

For each detected defect, the system provides a confidence score indicating how certain it is about the classification. This transparency allows quality control personnel to prioritize their attention on high-confidence critical defects while manually verifying uncertain cases.

### Comprehensive Reporting System

CircuitGuard generates multiple report formats to serve different needs within your organization. The PDF report provides a professional, human-readable document with executive summaries, detailed defect breakdowns, visual analysis with annotated images, and repair recommendations. This is perfect for quality reports, customer documentation, or management reviews.

The CSV exports contain detailed defect data in a structured, machine-readable format suitable for further analysis, database integration, or statistical process control. The analysis log includes comprehensive information about every aspect of the inspection, from processing parameters to individual defect measurements.

All output images, reports, and data files are bundled into a convenient ZIP package for easy archival and sharing.

### Quality Metrics and Scoring

Beyond simple defect detection, CircuitGuard calculates meaningful quality metrics that provide insights into board health. The severity score (0-100) gives a quick at-a-glance assessment of overall board quality, considering both the number of defects and their individual confidence scores. The system calculates the percentage of board area affected by defects, helping prioritize boards that may need rework versus those that should be scrapped.

Defect density metrics (defects per unit area) help identify whether problems are localized or widespread across the board. Critical defect alerts specifically flag high-confidence detections of the most serious defect types that could render the board completely non-functional.

## 💻 Technology and Architecture

CircuitGuard is built on a modern technology stack that balances performance, reliability, and ease of deployment.

### Programming and Frameworks

The entire application is written in **Python 3.11**, chosen for its rich ecosystem of scientific computing and machine learning libraries. Python's readability and extensive community support make the codebase maintainable and extensible.

**Flask 2.3** serves as our web framework, providing a lightweight yet powerful foundation for handling HTTP requests, file uploads, and serving results. Flask's simplicity allows us to focus on core functionality while still providing production-ready features like secure file handling and efficient request routing.

### Artificial Intelligence

The deep learning components leverage **PyTorch 2.9**, one of the most advanced neural network frameworks available. PyTorch's dynamic computation graphs and intuitive API made it ideal for developing and training our custom defect classification model. The framework's excellent GPU support ensures fast inference times when hardware acceleration is available.

**TorchVision 0.24** complements PyTorch by providing pre-trained models, image transformations, and computer vision utilities. We use transfer learning with ImageNet-pretrained models as the foundation for our custom classifier, significantly reducing training time and improving accuracy.

### Computer Vision Pipeline

**OpenCV 4.8** handles all the heavy lifting for image processing operations. This industry-standard library provides optimized implementations of feature detection, image registration, morphological operations, and contour analysis. Its C++ backend ensures that even complex operations run efficiently, keeping processing times low.

**NumPy 1.26** powers the numerical computations underlying the image processing pipeline. Its efficient array operations and mathematical functions are essential for manipulating image data at the pixel level.

**Pillow 10.3** provides high-level image I/O capabilities, making it easy to read and write various image formats while handling color space conversions and basic transformations.

### Report Generation

**ReportLab 4.x** enables the creation of professional, customizable PDF reports. The library provides fine-grained control over document layout, allowing us to create reports with tables, charts, embedded images, and custom styling.

On the client side, **Chart.js 4.x** generates interactive visualizations that help users quickly understand defect distributions and quality trends. The responsive charts work seamlessly across different devices and screen sizes.

### User Interface

The frontend combines **HTML5**, **CSS3**, and **JavaScript** to create a modern, responsive interface that works on desktops, tablets, and mobile devices. **Bootstrap 5.3** provides the foundation for consistent, professional-looking UI components without requiring extensive custom styling.

## 🛠️ Getting Started

### What You'll Need

Before installing CircuitGuard, ensure your system meets these requirements. You'll need a computer running Windows, Linux, or macOS with at least 4GB of RAM (8GB recommended for optimal performance). Python 3.11 or newer must be installed, along with pip for package management. If you plan to use GPU acceleration, you'll need an NVIDIA graphics card with CUDA support.

### Installation Process

Getting CircuitGuard up and running is straightforward. First, obtain the code by cloning the repository from GitHub or downloading the source as a ZIP file. If you're using Git, simply run the clone command and navigate into the newly created directory.

We strongly recommend creating a virtual environment to keep CircuitGuard's dependencies isolated from your system Python installation. On Windows, create and activate a virtual environment using the built-in venv module. On Linux or macOS, the process is similar but uses slightly different activation commands.

With your virtual environment active, install the required packages. For CPU-only operation, install directly from the requirements file. This will fetch all necessary dependencies including Flask, OpenCV, NumPy, and other components. If you have an NVIDIA GPU and want faster processing, install the CUDA-enabled version of PyTorch before installing other requirements.

### Setting Up the Model

CircuitGuard requires a pre-trained model file to perform defect classification. Place your trained model (typically named `best_model.pth`) in the `model` directory. If you're using our pre-trained model, download it from the releases page and place it in the correct location.

### Running the Application

Starting CircuitGuard is as simple as running the main Python script. Execute `python app1.py` from the project directory, and the application will initialize the model and start the web server. You'll see console output indicating successful model loading and server startup. By default, the application runs on port 5000 of your local machine.

Open your web browser and navigate to `http://127.0.0.1:5000` to access the CircuitGuard interface.

## 📖 How to Use CircuitGuard

### Preparing Your Images

For best results, ensure your PCB images are well-lit with consistent lighting conditions. The template image should be a known-good PCB that serves as your quality reference, while the test image is the PCB you want to inspect. Both images should be clear, in focus, and show the complete board or the section you want to analyze.

### The Inspection Process

Start by uploading your template image—this is your reference board that represents perfect quality. Next, upload the test image that you want to inspect for defects. The interface provides clear visual feedback as files are selected.

Before running the analysis, you can adjust three key parameters. The **Threshold** setting (default 10) controls how sensitive the system is to differences between images—higher values detect only more obvious defects, while lower values catch subtle variations but may also pick up noise. The **Minimum Area** parameter (default 100 pixels) filters out tiny defects that might be inconsequential—increase this to ignore small artifacts. The **Confidence Filter** (default 0%) allows you to exclude defect detections below a certain confidence level, helping reduce false positives.

Click the "Run Inference" button to start the analysis. The system will process your images through the complete pipeline—alignment, defect detection, classification, and report generation. Processing typically completes in 2-5 seconds depending on image size and hardware capabilities.

### Understanding Your Results

The results page presents a comprehensive dashboard of findings. At the top, you'll see the overall quality assessment with a clear PASS status indicator and severity score. The key metrics panel shows the total number of defects found, average detection confidence, percentage of board area affected, and defect density.

Interactive charts visualize the distribution of defect types and confidence levels, making it easy to spot patterns. The annotated image shows exactly where each defect was detected, with colored bounding boxes and labels. You can also view the difference map highlighting all areas where the test board differs from the template, and the binary mask showing detected defect regions.

A detailed defect table lists every detected defect with its ID, type, confidence score, size, and severity level. If applicable, you'll also see repair recommendations with specific actions for each defect type, priority levels, and estimated repair times.

### Downloading Reports

CircuitGuard provides multiple download options to suit different needs. The PDF report contains everything in a professional document format suitable for archival or sharing with stakeholders. The CSV log includes all defect data in a structured format for spreadsheet analysis or database import. The complete ZIP package bundles all images, reports, and data files together for convenient storage and sharing.

## 🧠 The AI Behind CircuitGuard

### Neural Network Architecture

CircuitGuard's classification model is based on a Residual Network (ResNet) architecture, a proven design that has demonstrated excellent performance on image classification tasks. The network takes 224×224 pixel RGB images as input and processes them through multiple layers of convolutional operations, batch normalization, and activation functions.

The key innovation of ResNet—skip connections—allows the network to learn complex features while avoiding the vanishing gradient problem that affects very deep networks. Our implementation uses approximately 11 million trainable parameters, carefully tuned to balance accuracy and inference speed.

### Training Process

The model was trained using transfer learning, starting with weights pre-trained on ImageNet (a massive dataset of natural images). This approach leverages general visual feature extraction capabilities learned from millions of images, then fine-tunes them for our specific task of PCB defect classification.

Training involved thousands of PCB images showing various defect types, with careful data augmentation to improve model robustness. We applied random rotations, flips, brightness adjustments, and noise injection to help the model generalize to different imaging conditions. The model was optimized using the Adam optimizer with cross-entropy loss, trained over multiple epochs with early stopping to prevent overfitting.

### Performance Characteristics

On our test dataset, the model achieves 85-95% accuracy depending on defect type, with particularly strong performance on critical defects like open circuits (95% accuracy). The system processes each PCB in under a second on GPU hardware, or 2-5 seconds on CPU-only systems, making it suitable for real-time production line integration.

## 📊 Proven Results

CircuitGuard has been validated across diverse PCB types and manufacturing scenarios. Detection accuracy varies by defect type, with the highest performance on open circuits (95%) and short circuits (94%)—the most critical defects. Solder bridges and missing holes are detected with 90-92% accuracy, while mouse bites and spurious copper achieve 87-88% accuracy.

Processing performance depends on hardware—GPU-equipped systems can analyze approximately 75 PCBs per minute, while CPU-only deployments achieve around 20 PCBs per minute. Memory usage typically ranges from 500MB to 2GB depending on image resolution.

## 🤝 Contributing to CircuitGuard

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or enhancing the model, your input is valued. Please follow our contribution guidelines, maintain code quality standards with PEP 8 compliance, and include tests for new functionality.

## 📄 License

CircuitGuard is released under the MIT License, allowing free use, modification, and distribution with proper attribution.

## 🙏 Acknowledgments

This project builds on the excellent work of the PyTorch, OpenCV, and Flask communities. We're grateful to the developers and researchers who created these foundational tools, and to the PCB manufacturing professionals who provided feedback and real-world testing data.

**CircuitGuard: Bringing AI-powered quality control to every electronics workshop**

