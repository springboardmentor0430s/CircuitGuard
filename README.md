
INTRODUCTION 
	The Goal : The model will automatically finds the error in the PCB board circuit. 
	The Smart Part :  It is going to use deep learning and CNN process for the error detection.
	Working Step 1: The model will compare the picture of the golden image of PCB and the other image is having defects and spot the differences.
	Working Step 2 : After finding the differences it will detect the exact error type and location.
	The User Part : The user just simply have to upload the image of their pcb.
	Result : The model will give you the image you’ve uploaded with the detection of errors. You can easily find where the errors are occurred. 



ABSTRACT 
Think of PDB files as the 3D blueprints for all the tiny, complex machines (called proteins) in our bodies. The trouble is, these blueprints often have mistakes—like a smudge, a weird angle, or even a whole section that's just missing.
Our project was basically to become a quality checker for these blueprints. We created a way to automatically scan them and find all the common screw-ups, like:
	Atoms that are crashing into each other.
	Parts of the protein chain that are broken or missing.
	Measurements that just don't look right.
Why bother? Because scientists rely on these blueprints for all kinds of important research. We wanted to make sure those blueprints are accurate and trustworthy before they start building.


PROBLEM STATEMENT 
     Although PDB files are widely used in research, they may contain:
	Missing atoms or residues
	Incorrect bond angles or lengths
	Steric clashes between atoms
	Chain discontinuities
	Poorly resolved side-chain conformations
These errors lead to inaccurate interpretations and unreliable computational results. Thus, a structured method is required to detect and report errors effectively.


WORKING PROCESS 
	Uploading image
	Processing the image 
	Comparing the uploaded image with golden image 
	Finding the defects 
	Giving the uploaded image with identified defects 
	Displaying the parts of defects in image 
	Output in the Jason format 
INFO GRAPH OF WORKING PROCESS 
         

IMAGES OF THE PROJECT 
	Interface image :
         
 
     Above image is the interface of the uploading image. Here the golden image is already inbuilt and you just need to upload only the image of pcb which we have to find the defects and we have to give the confidence limit. Here the confidence limit refers to the minimum amount of accuracy we need in the output image. 

Image to find the defects:
 
This is the image of pcb where we have to find the defects. 

After finding the defects image: 
 
This is the image after finding the defects in the image and it will highlight the part where the defects are arrived.    
Jason output: 
 
This is the output where if the code run successfully it will show the number 200 and successful response. 

TRAINING PART 
 
This is the training part of the code where the code will run to 5 times for more accuracy for each time. In each step the accuracy is going to increase and loss is also  decreasing in each step. 
Graph of taring part 
 
The accuracy we got is 99.16%. 
Code for the Training and validation part: 
 



OBJECTIVES
	To analyze PDB structures and identify common structural errors.
	To detect missing residues and incomplete chain segments.
	To evaluate steric clashes and geometric inconsistencies.
	To document findings for improving downstream structure-related analyses.


METHODOLOGY 
Get the Data: We first downloaded all the 3D protein "blueprints" (PDB files) from the big online RCSB database.
Translate the Files: We used Python tools to "parse" those complex files. This let us pull out the exact 3D coordinates for every atom and understand the protein's structure.
Find the Flaws: This was the core of the project. We ran checks to spot common errors, specifically looking for:
Missing atoms or entire sections.
Breaks in the protein's "backbone."
"Steric clashes" (atoms bumping into each other).
Weird geometry (unnatural bond lengths or angles).
Make the Report: Finally, we recorded all the errors we found and categorized them. This created a "report card" for each protein, showing how severe its flaws were.

RESULTS AND FINDINGS 
The evaluation revealed:
Multiple structures contained missing loop regions and unresolved side chains.
Some proteins exhibited chain discontinuities likely caused by weak electron density.
Steric clashes were observed at crowded residue interfaces, indicating potential refinement issues.
Bond angle deviations were minimal but present in flexible regions.
These findings demonstrate that even published structures require validation prior to use.


CONCLUSION 
PDB error detection is essential for ensuring the accuracy and usability of protein structural data. The analysis performed in this project highlights the prevalence of missing residues, steric clashes, and chain discontinuities in protein structures. By identifying these issues early, researchers can refine models or select higher-quality alternatives, improving the reliability of downstream structural and computational analyses. 



This repository features the backend of a Machine Learning application, focusing on detecting defects in PCB images. 
The detection model used in this case is trained on a YOLOv5 model, using the dataset: **https://www.kaggle.com/datasets/akhatova/pcb-defects/data** 

In a gist, the **objectives** met by this project include:\
-Train an object detection model to detect different PCB defects using a public dataset.\
-Accept input images through an API and return predictions through bounding boxes and labels.\
-Provide visualized output with bounding boxes drawn on the images.\
-Store the prediction results in a PostgreSQL database. 

**Features:** 
1. Machine Learning Model: Trained on a public dataset using a YOLO-based object detection model.\
2. Backend: Built using FastAPI to expose the following endpoints: \
          **POST /predict:** \
          (POST): Accept image data and confidence limit, return predictions(bounding boxes and labels) as structured data. \
          Request: Image file (multipart/form-data) \
                  Confidence limit (float, optional) \
          Response: List of predictions with bounding boxes and their confidence scores. \
          \
          **POST /visualize**: \
          Accepts image data and a confidence limit, returning an image with drawn bounding boxes and labels. \
           Request: Image file (multipart/form-data) \
                  Confidence limit (float, optional) \
          Response: Image with bounding boxes drawn around detected defects. \
          
3. Database Integration: PostgreSQL stores the results of the predictions for convenient retrieval and logging. \


**Installation:** 
To get this running locally, make sure you meet these prerequisites: 
1. Python 3.8+
2. PostgreSQL (can be opted out of, if storing results is not a priority)

**Setup:**
1. Clone this repository: \
   git clone https://github.com/springboardmentor0430s/CircuitGuard.git \
   cd PCB-Defects-detection 

2. Create a virtual environment(if needed), and install the dependencies for the project: \
   python -m venv venv \
   venv/bin/activate \
   pip install -r requirements.txt 
   
3. Training your model:
   If you wish to train YOLOv5 according to new parameters, you can make use of the 'Dataprocess.py' and 'testing.py' codes;
   - 'Dataprocess.py' essentially does the required data preprocessing and makes a new dataset format appropriate for the YOLO format; \
     The code resizes the images and splits them into training, validation, and testing folders. 
   - 'testing.py' is a simple testing code to make sure the trained model is working, before moving on to deploying the model onto a backend. 
   - Whilst using these codes, make sure to change the file directories to their suitable locations on your device.
     
4. Setting up the backend using FASTAPI (& PostgreSQL):
   Make sure you have your trained model by this point, or you are welcome to use my pre-trained model- 'best.pt' 
   - Here, depending on whether one wishes to integrate PostgreSQL or wants to opt out, I have provided two codes; 
   - The 'main.py' integrates the trained model onto the backend using FASTAPI, with two main endpoints - /predict and /visualize, and then stores predictions in a database, such as: 
       1. Confidence Limit 
       2. Dimensions of Bounding Boxes 
       3. Image ID 
   - Before running 'main.py', make sure PostgreSQL has been set up, and create a new database, either through the command line or pgAdmin4;
       -Using the Command Line: \
         1. psql -U postgres (assuming you are under username 'postgres') \
         2. enter your password \
         3. CREATE DATABASE pcb_predict; 
    - After creating the database, make sure all the file directories and variable names are in order, and then use this line in the cmd line; \
      uvicorn main:app --reload 
    - Once the server is up, you should be able to interact with the endpoints at: **http://127.0.0.1:8000/docs**
      
    -  If you opt out of PostgreSQL, use the code 'app.py', and the rest of the steps should be the same skipping the database setting-up steps; \
      Use: uvicorn app:app --reload

 5. Access the stored data:
    From the command line, use the below- \
      psql -U postgres -d pcb_predict \
      enter your password \
      SELECT * FROM predictions;


   

   

   
